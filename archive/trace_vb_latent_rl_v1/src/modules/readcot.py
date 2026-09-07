import math
import random
import re
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


_NUMBER_PATTERN = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?")
_IDENTIFIER_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_WORD_PATTERN = re.compile(r"[A-Za-z]+")
_OPERATOR_PATTERN = re.compile(r"<=|>=|==|!=|[=+\-*/^%()]")
_COMMON_STOPWORDS = {
    "a",
    "an",
    "and",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "if",
    "in",
    "into",
    "is",
    "it",
    "of",
    "on",
    "or",
    "so",
    "that",
    "the",
    "then",
    "therefore",
    "thus",
    "to",
    "we",
    "with",
}
_SEQUENTIAL_CUES = (
    "above",
    "earlier",
    "from this",
    "plug",
    "previous",
    "substitute",
    "therefore",
    "thus",
    "using",
)


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower().strip())


def normalize_answer_text(text: str) -> str:
    return normalize_text(text).strip("# ").rstrip(".").replace(",", "")


def extract_numbers(text: str) -> List[str]:
    return _NUMBER_PATTERN.findall(text)


def extract_identifiers(text: str) -> List[str]:
    return [token.lower() for token in _IDENTIFIER_PATTERN.findall(text) if token.lower() not in _COMMON_STOPWORDS]


def extract_words(text: str) -> List[str]:
    return [token.lower() for token in _WORD_PATTERN.findall(text) if token.lower() not in _COMMON_STOPWORDS]


def extract_operators(text: str) -> List[str]:
    return _OPERATOR_PATTERN.findall(text)


def split_equation(text: str) -> Tuple[str, str]:
    if "=" not in text:
        return text, text
    left, right = text.split("=", 1)
    return left, right


def build_step_features(text: str, question_text: str) -> dict:
    left, right = split_equation(text)
    question_numbers = set(extract_numbers(question_text))
    question_identifiers = set(extract_identifiers(question_text))

    left_numbers = set(extract_numbers(left))
    right_numbers = set(extract_numbers(right))
    all_numbers = set(extract_numbers(text))

    left_identifiers = set(extract_identifiers(left))
    right_identifiers = set(extract_identifiers(right))
    all_identifiers = set(extract_identifiers(text))
    all_words = set(extract_words(text))

    operators = set(extract_operators(text))
    produced_numbers = right_numbers if right_numbers else (all_numbers - question_numbers)
    produced_identifiers = right_identifiers if right_identifiers else (all_identifiers - question_identifiers)
    produced_words = (
        set(extract_words(right))
        if "=" in text
        else (all_words - set(extract_words(question_text)))
    )

    return {
        "all_numbers": all_numbers,
        "all_identifiers": all_identifiers,
        "all_words": all_words,
        "left_numbers": left_numbers,
        "left_identifiers": left_identifiers,
        "produced_numbers": produced_numbers,
        "produced_identifiers": produced_identifiers,
        "produced_words": produced_words,
        "operators": operators,
    }


def _rule_confidence_from_score(score: float) -> float:
    return float(min(1.0, 0.35 + 0.15 * score))


def induce_dependency_supervision(
    question: str,
    steps: Sequence[str],
    answer: Optional[str] = None,
    threshold: float = 1.0,
    negative_confidence: float = 0.25,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build weak READ-CoT dependency skeleton D and confidence matrix C.

    The rule path is intentionally high-recall for arithmetic / symbolic CoT and
    acts as the fallback when no offline semantic judge annotations are cached.
    """
    n_steps = len(steps)
    if n_steps == 0:
        empty = torch.zeros(0, 0, dtype=torch.float32)
        return empty, empty

    dependency = torch.zeros(n_steps, n_steps, dtype=torch.float32)
    confidence = torch.zeros(n_steps, n_steps, dtype=torch.float32)
    if n_steps > 1 and negative_confidence > 0:
        confidence[torch.triu(torch.ones(n_steps, n_steps, dtype=torch.bool), diagonal=1)] = negative_confidence

    features = [build_step_features(step, question_text=question) for step in steps]
    score_matrix = torch.zeros(n_steps, n_steps, dtype=torch.float32)

    for j in range(1, n_steps):
        step_j = normalize_text(steps[j])
        has_sequential_cue = any(cue in step_j for cue in _SEQUENTIAL_CUES)

        best_i = None
        best_score = 0.0
        for i in range(j):
            score = 0.0
            produced_numbers = features[i]["produced_numbers"]
            produced_identifiers = features[i]["produced_identifiers"]
            produced_words = features[i]["produced_words"]

            number_overlap = produced_numbers & (
                features[j]["left_numbers"] | features[j]["all_numbers"] | features[j]["produced_numbers"]
            )
            identifier_overlap = produced_identifiers & (
                features[j]["left_identifiers"] | features[j]["all_identifiers"] | features[j]["produced_identifiers"]
            )
            word_overlap = produced_words & features[j]["all_words"]
            operator_overlap = features[i]["operators"] & features[j]["operators"]

            score += 2.0 * len(number_overlap)
            score += 1.0 * len(identifier_overlap)
            score += 0.5 * len(word_overlap)
            score += 0.25 * len(operator_overlap)
            if has_sequential_cue and i == j - 1:
                score += 1.0

            score_matrix[i, j] = score
            if score >= threshold:
                dependency[i, j] = 1.0
                confidence[i, j] = max(confidence[i, j].item(), _rule_confidence_from_score(score))

            if score > best_score:
                best_score = score
                best_i = i

        if dependency[:, j].sum() == 0 and best_i is not None and (has_sequential_cue or j == n_steps - 1):
            dependency[best_i, j] = 1.0
            fallback_conf = 0.55 if has_sequential_cue else 0.45
            confidence[best_i, j] = max(confidence[best_i, j].item(), fallback_conf)
        if dependency[:, j].sum() == 0 and j > 0 and n_steps <= 2:
            dependency[j - 1, j] = 1.0
            confidence[j - 1, j] = max(confidence[j - 1, j].item(), 0.5)

    # Answer-backtrace support: if a step states or touches the answer, connect it
    # to the strongest previous producer so final-answer steps are not isolated.
    if answer:
        answer_norm = normalize_answer_text(answer)
        answer_numbers = set(extract_numbers(answer_norm))
        for j, step in enumerate(steps):
            step_norm = normalize_answer_text(step)
            step_numbers = set(extract_numbers(step_norm))
            touches_answer = bool(answer_norm and answer_norm in step_norm) or bool(answer_numbers & step_numbers)
            if not touches_answer or j == 0:
                continue
            candidate_scores = score_matrix[:j, j]
            best_score = candidate_scores.max().item() if candidate_scores.numel() > 0 else 0.0
            best_i = int(candidate_scores.argmax().item()) if best_score > 0 else j - 1
            dependency[best_i, j] = 1.0
            confidence[best_i, j] = max(confidence[best_i, j].item(), 0.75)

    return dependency, confidence


def induce_dependency_matrix(question: str, steps: Sequence[str], answer: Optional[str] = None) -> torch.Tensor:
    dependency, _ = induce_dependency_supervision(question=question, steps=steps, answer=answer)
    return dependency


def sanitize_dependency_supervision(
    dependency_matrix,
    confidence_matrix,
    n_steps: int,
    device=None,
    negative_confidence: float = 0.25,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    if dependency_matrix is None:
        return None, None

    try:
        dependency = torch.tensor(dependency_matrix, dtype=torch.float32, device=device)
    except (TypeError, ValueError):
        return None, None
    if dependency.shape != (n_steps, n_steps):
        return None, None
    dependency = (dependency > 0).float()
    dependency = torch.triu(dependency, diagonal=1)

    if confidence_matrix is None:
        confidence = torch.ones_like(dependency) * negative_confidence
        confidence = torch.where(dependency > 0, torch.ones_like(confidence), confidence)
    else:
        try:
            confidence = torch.tensor(confidence_matrix, dtype=torch.float32, device=device)
        except (TypeError, ValueError):
            confidence = torch.ones_like(dependency) * negative_confidence
            confidence = torch.where(dependency > 0, torch.ones_like(confidence), confidence)
        if confidence.shape != (n_steps, n_steps):
            confidence = torch.ones_like(dependency) * negative_confidence
            confidence = torch.where(dependency > 0, torch.ones_like(confidence), confidence)

    confidence = torch.triu(confidence.clamp(0.0, 1.0), diagonal=1)
    return dependency, confidence


def fuse_dependency_supervision(
    rule_dependency: torch.Tensor,
    rule_confidence: torch.Tensor,
    model_dependency: Optional[torch.Tensor] = None,
    model_confidence: Optional[torch.Tensor] = None,
    rule_weight: float = 0.55,
    model_weight: float = 0.45,
    decision_threshold: float = 0.45,
    keep_high_conf_rule_edges: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fuse rule and offline-judge dependency evidence into final D/C."""
    if model_dependency is None or model_confidence is None:
        return rule_dependency.float(), rule_confidence.float().clamp(0.0, 1.0)

    model_dependency = model_dependency.to(rule_dependency.device).float()
    model_confidence = model_confidence.to(rule_confidence.device).float().clamp(0.0, 1.0)
    rule_dependency = rule_dependency.float()
    rule_confidence = rule_confidence.float().clamp(0.0, 1.0)

    weighted_support = rule_weight * rule_dependency * rule_confidence
    weighted_support = weighted_support + model_weight * model_dependency * model_confidence
    weighted_confidence = rule_weight * rule_confidence + model_weight * model_confidence
    dependency = (weighted_support >= decision_threshold).float()

    if keep_high_conf_rule_edges:
        dependency = torch.maximum(dependency, ((rule_dependency > 0) & (rule_confidence >= 0.75)).float())

    confidence = torch.maximum(weighted_confidence, weighted_support).clamp(0.0, 1.0)
    confidence = torch.triu(confidence, diagonal=1)
    dependency = torch.triu(dependency, diagonal=1)
    return dependency, confidence



def dependency_upper_triangle_mask(size: int, device) -> torch.Tensor:
    return torch.triu(torch.ones(size, size, device=device, dtype=torch.bool), diagonal=1)


def dependency_bce_loss(
    dep_logits: torch.Tensor,
    gold: torch.Tensor,
    confidence: Optional[torch.Tensor] = None,
    pos_weight_max: Optional[float] = None,
    loss_clamp: Optional[float] = None,
) -> torch.Tensor:
    size = gold.shape[0]
    mask = dependency_upper_triangle_mask(size=size, device=gold.device)
    if mask.sum() == 0:
        return dep_logits.new_zeros(())

    gold = gold.float()
    valid_gold = gold[mask]
    n_pos = valid_gold.sum()
    n_neg = mask.sum().float() - n_pos
    pos_weight = (n_neg / (n_pos + 1e-6)).clamp_min(1.0)
    if pos_weight_max is not None:
        pos_weight = pos_weight.clamp(max=float(pos_weight_max))

    loss = F.binary_cross_entropy_with_logits(dep_logits, gold, reduction="none", pos_weight=pos_weight)
    loss = loss.masked_select(mask)
    if confidence is None:
        dep_loss = loss.mean()
        if loss_clamp is not None:
            dep_loss = dep_loss.clamp(max=float(loss_clamp))
        return dep_loss

    weights = confidence.to(gold.device).float().clamp(0.0, 1.0).masked_select(mask)
    if weights.sum() <= 0:
        return dep_logits.new_zeros(())
    dep_loss = (loss * weights).sum() / weights.sum().clamp_min(1e-6)
    if loss_clamp is not None:
        dep_loss = dep_loss.clamp(max=float(loss_clamp))
    return dep_loss


def dependency_f1_score(dep_probs: torch.Tensor, gold: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    size = gold.shape[0]
    mask = dependency_upper_triangle_mask(size=size, device=gold.device)
    if mask.sum() == 0:
        return dep_probs.new_tensor(1.0)

    pred = dep_probs > threshold
    gold = gold.bool()
    pred = pred[mask]
    gold = gold[mask]
    tp = (pred & gold).sum().float()
    fp = (pred & ~gold).sum().float()
    fn = (~pred & gold).sum().float()
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    return 2 * precision * recall / (precision + recall + 1e-6)


def project_step_dependencies(assignment: torch.Tensor, relation_logits: torch.Tensor) -> torch.Tensor:
    # assignment: [M, K], relation_logits: [M, M]
    return torch.einsum("mi,mn,nj->ij", assignment, relation_logits, assignment)


def reconstruct_dependency_logits(
    assignment: torch.Tensor,
    relation_probs: torch.Tensor,
    center_relations: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # assignment: [M, K], relation_probs: [M, M]
    if center_relations:
        relation_values = relation_probs * 2 - 1
    else:
        relation_values = relation_probs
    relation_values = relation_values - torch.diag_embed(torch.diagonal(relation_values))
    dependency_logits = project_step_dependencies(assignment, relation_values)
    dependency_probs = torch.sigmoid(dependency_logits)
    return dependency_logits, dependency_probs


def aggregate_step_residuals(assignment: torch.Tensor, residuals: torch.Tensor) -> torch.Tensor:
    # assignment: [M, K], residuals: [K, H]
    return torch.einsum("mk,kh->mh", assignment, residuals)


def cosine_similarity_mean(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    if x.numel() == 0:
        return x.new_tensor(1.0)
    return F.cosine_similarity(x, y, dim=-1).mean()


def select_dependency_critical_anchors(
    steps: Sequence[str],
    dependency_matrix: torch.Tensor,
    answer: str,
    n_anchors: int,
    alpha: float = 1.0,
    beta: float = 1.0,
    gamma: float = 1.0,
    dependency_errors: Optional[torch.Tensor] = None,
    residual_errors: Optional[torch.Tensor] = None,
    anchor_bonus: Optional[torch.Tensor] = None,
    anchor_bonus_weight: float = 1.0,
    strategy: str = "dependency_critical",
    length_penalty: float = 0.0,
) -> Tuple[List[int], torch.Tensor]:
    n_steps = len(steps)
    if n_steps == 0 or n_anchors <= 0:
        return [], torch.zeros(0)

    if strategy == "random":
        indices = list(range(n_steps))
        random.shuffle(indices)
        indices = sorted(indices[: min(n_steps, n_anchors)])
        return indices, torch.ones(len(indices))

    degree = dependency_matrix.sum(dim=0) + dependency_matrix.sum(dim=1)
    answer_norm = normalize_answer_text(answer)
    answer_numbers = set(extract_numbers(answer_norm))
    answer_touch = torch.zeros(n_steps, dtype=torch.float32)
    for idx, step in enumerate(steps):
        step_norm = normalize_text(step)
        if answer_norm and answer_norm in normalize_answer_text(step_norm):
            answer_touch[idx] += 1.0
        step_numbers = set(extract_numbers(step_norm))
        if answer_numbers and step_numbers & answer_numbers:
            answer_touch[idx] += 0.5
        if idx == n_steps - 1:
            answer_touch[idx] += 0.5

    answer_support = answer_touch.clone()
    active = (answer_touch > 0).nonzero(as_tuple=False).flatten().tolist()
    if len(active) == 0 and n_steps > 0:
        active = [n_steps - 1]
        answer_support[n_steps - 1] += 0.5

    queue = list(active)
    visited = set(active)
    while queue:
        current = queue.pop(0)
        parents = (dependency_matrix[:, current] > 0).nonzero(as_tuple=False).flatten().tolist()
        for parent in parents:
            answer_support[parent] = torch.maximum(
                answer_support[parent],
                answer_support[current] * 0.75,
            )
            if parent not in visited:
                visited.add(parent)
                queue.append(parent)

    total_error = torch.zeros(n_steps, dtype=torch.float32)
    if dependency_errors is not None:
        total_error += dependency_errors.float()
    if residual_errors is not None:
        total_error += residual_errors.float()

    scores = alpha * degree.float() + beta * answer_support + gamma * total_error
    if anchor_bonus is not None and anchor_bonus.numel() == n_steps:
        scores = scores + float(anchor_bonus_weight) * anchor_bonus.float()
    if length_penalty > 0:
        step_lengths = torch.tensor(
            [max(1, len(step.split())) for step in steps],
            dtype=torch.float32,
        )
        scores = scores - float(length_penalty) * step_lengths
    n_select = min(n_steps, n_anchors)
    top_indices = torch.topk(scores, k=n_select).indices.tolist()
    return sorted(top_indices), scores


class ReadStepCompressor(nn.Module):
    def __init__(self, hidden_size: int, n_latents: int, dropout: float = 0.0):
        super().__init__()
        self.n_latents = n_latents
        self.latent_queries = nn.Parameter(torch.randn(n_latents, hidden_size) / math.sqrt(hidden_size))
        self.query_proj = nn.Linear(hidden_size, hidden_size)
        self.key_proj = nn.Linear(hidden_size, hidden_size)
        self.value_proj = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, step_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # step_states: [K, H]
        if step_states.dim() != 2:
            raise ValueError(f"step_states should have shape [K, H], but got {tuple(step_states.shape)}")
        keys = self.key_proj(step_states)  # [K, H]
        values = self.value_proj(step_states)  # [K, H]
        queries = self.query_proj(self.latent_queries)  # [M, H]
        scores = torch.matmul(queries, keys.transpose(0, 1)) / math.sqrt(step_states.shape[-1])  # [M, K]
        assignment = torch.softmax(scores, dim=-1)
        assignment = self.dropout(assignment)
        latents = torch.matmul(assignment, values)
        return latents, assignment


class LatentRelationHead(nn.Module):
    def __init__(self, hidden_size: int, relation_hidden_size: Optional[int] = None):
        super().__init__()
        relation_hidden_size = relation_hidden_size or hidden_size
        self.proj = nn.Linear(hidden_size * 2, relation_hidden_size)
        self.score = nn.Linear(relation_hidden_size, 1)

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        # latents: [M, H]
        n_latents = latents.shape[0]
        lhs = latents.unsqueeze(1).expand(n_latents, n_latents, -1)
        rhs = latents.unsqueeze(0).expand(n_latents, n_latents, -1)
        pair = torch.cat([lhs, rhs], dim=-1)
        hidden = torch.tanh(self.proj(pair))
        relation_logits = self.score(hidden).squeeze(-1)
        relation_logits = relation_logits - torch.diag_embed(torch.diagonal(relation_logits))
        return relation_logits

    def forward_probs(self, latents: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        relation_logits = self.forward(latents)
        relation_probs = torch.sigmoid(relation_logits)
        relation_probs = relation_probs - torch.diag_embed(torch.diagonal(relation_probs))
        return relation_logits, relation_probs
