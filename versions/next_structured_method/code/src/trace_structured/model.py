"""Question-only causal roles with matched raw categorical rollout/replay."""
from copy import deepcopy
from dataclasses import dataclass
import math
import torch
from torch import nn
import torch.nn.functional as F

from .data import output_parts, question_prompt
from .policy import RolePolicy
from .targets import build_targets, process_scores, structure_loss


@dataclass
class RoleTrace:
    question_ids: torch.Tensor
    inputs: torch.Tensor
    states: torch.Tensor
    pre_states: torch.Tensor
    actions: torch.Tensor
    log_probs: torch.Tensor
    means: torch.Tensor
    log_stds: torch.Tensor


@dataclass
class Completion:
    tokens: torch.Tensor
    log_probs: torch.Tensor
    mask: torch.Tensor
    text: str
    stopped_on_eos: bool


def token_mask(tokens, eos_ids, length=None):
    """Explicit length plus first EOS; PAD and EOS may share an ID."""
    if tokens.ndim != 1:
        raise ValueError("one-dimensional completion required")
    length = len(tokens) if length is None else length
    if not 0 <= length <= len(tokens):
        raise ValueError("invalid completion length")
    mask = torch.arange(len(tokens), device=tokens.device) < length
    for i in range(length):
        if int(tokens[i]) in eos_ids:
            mask[i + 1:] = False
            break
    return mask


class StructuredModel(nn.Module):
    def __init__(self, language_model, tokenizer, config):
        super().__init__()
        self.language_model = language_model
        self.tokenizer = tokenizer
        self.config = config
        hidden = int(language_model.config.hidden_size)
        self.policy = RolePolicy(hidden, config)
        self.bridge = nn.Sequential(nn.Linear(hidden, hidden), nn.Tanh())
        self.queries = nn.Parameter(torch.randn(len(config.role_names), hidden) / math.sqrt(hidden))
        generator = torch.Generator().manual_seed(config.projection_seed)
        projection = torch.randn(hidden, config.semantic_dim, generator=generator) / math.sqrt(hidden)
        self.register_buffer("semantic_projection", projection)
        self.plan_head = nn.Linear(config.semantic_dim, config.solve_roles * config.semantic_dim)
        # Always present in the schema; initialized from Stage1 exactly once.
        self.reference_policy = deepcopy(self.policy).requires_grad_(False)
        self.score_plan_head = deepcopy(self.plan_head).requires_grad_(False)
        self.register_buffer("reference_ready", torch.tensor(False))
        self.train(False)

    @property
    def device(self):
        return self.queries.device

    @property
    def eos_ids(self):
        values = self.language_model.config.eos_token_id
        values = values if isinstance(values, (tuple, list)) else [values]
        return {int(x) for x in [*values, self.tokenizer.eos_token_id] if x is not None}

    def train(self, mode=True):
        super().train(mode)
        # Gradients are enabled independently of eval mode. No hidden dropout
        # draw may alter the rollout/replay conditional distribution.
        self.language_model.eval()
        self.reference_policy.eval()
        self.score_plan_head.eval()
        return self

    def initialize_stage2_reference(self):
        if bool(self.reference_ready):
            raise ValueError("reference already exists; restore it, do not rebuild it")
        self.reference_policy.load_state_dict(self.policy.state_dict())
        self.score_plan_head.load_state_dict(self.plan_head.state_dict())
        self.reference_ready.fill_(True)

    def configure_stage(self, stage):
        if stage not in ("stage0", "stage1", "stage2"):
            raise ValueError("unknown training stage")
        for name, param in self.named_parameters():
            if name.startswith(("reference_policy.", "score_plan_head.")):
                param.requires_grad_(False)
            elif not name.startswith("language_model."):
                param.requires_grad_(stage != "stage0")
        if stage == "stage2" and not bool(self.reference_ready):
            raise ValueError("Stage2 requires an initialized, persisted Stage1 reference")

    def encode(self, text):
        return self.tokenizer.encode(text, add_special_tokens=False)

    def _tensor(self, ids):
        return torch.tensor(ids, dtype=torch.long, device=self.device)

    def question_ids(self, question):
        ids = self.encode(question_prompt(question))
        if not ids or len(ids) > self.config.max_question_tokens:
            raise ValueError("question token budget exceeded; no silent truncation")
        return self._tensor(ids)

    @staticmethod
    def normalize_state(value):
        return F.layer_norm(value.float(), value.shape[-1:])

    def _forward_embeds(self, embeds, *, cache=None, total_length=None, hidden=False):
        n = embeds.shape[0]
        total = n if total_length is None else total_length
        mask = torch.ones(1, total, device=self.device, dtype=torch.long)
        positions = torch.arange(total - n, total, device=self.device).unsqueeze(0)
        return self.language_model(inputs_embeds=embeds.unsqueeze(0), attention_mask=mask,
                                   position_ids=positions, past_key_values=cache, use_cache=True,
                                   output_hidden_states=hidden)

    def roles(self, question, *, stochastic=False, forced_actions=None, sft_std=None):
        """The ONLY latent input entry point: no gold text, view IDs or graph."""
        n = len(self.config.role_names)
        if forced_actions is not None and forced_actions.shape != (n, self.config.action_dim):
            raise ValueError("saved actions do not match role schema")
        ids = self.question_ids(question)
        embedding = self.language_model.get_input_embeddings()
        question_embeds = embedding(ids)
        initial = self._forward_embeds(question_embeds, hidden=True)
        previous = self.normalize_state(initial.hidden_states[-1][0, -1])
        base = self.bridge(previous)
        cache = initial.past_key_values
        inputs, states, pres, actions, logprobs, means, stds = [], [], [], [], [], [], []
        for index in range(n):
            pres.append(previous)
            action, lp, mean, log_std = self.policy.realize(
                previous, index, stochastic=stochastic, sft_std=sft_std,
                forced=None if forced_actions is None else forced_actions[index])
            current = (base + self.config.query_scale * self.queries[index]
                       + self.config.action_scale * self.policy.action_projection(action)).to(question_embeds.dtype)
            output = self._forward_embeds(current.unsqueeze(0), cache=cache,
                                          total_length=len(ids) + index + 1, hidden=True)
            cache = output.past_key_values
            previous = self.normalize_state(output.hidden_states[-1][0, -1])
            for collection, value in ((inputs, current), (states, previous), (actions, action),
                                      (logprobs, lp), (means, mean), (stds, log_std)):
                collection.append(value)
        # Never expose a mutable KV cache for reuse by multiple consumers.
        return RoleTrace(ids, *(torch.stack(x) for x in (inputs, states, pres, actions, logprobs, means, stds)))

    def projected_states(self, trace):
        return trace.states.float() @ self.semantic_projection

    def plan_prediction(self, states, *, fixed=False):
        head = self.score_plan_head if fixed else self.plan_head
        if fixed and not bool(self.reference_ready):
            raise ValueError("fixed process scorer has not been initialized")
        return head(states[0]).reshape(self.config.solve_roles, self.config.semantic_dim)

    def structure(self, trace, targets):
        states = self.projected_states(trace)
        return structure_loss(states, self.plan_prediction(states), targets, self.config)

    @torch.no_grad()
    def score_process(self, trace, targets):
        states = self.projected_states(trace)
        return process_scores(states, self.plan_prediction(states, fixed=True), targets)

    def _output_prefix(self, trace):
        text = "\nAnchors:\n" if self.config.anchor_count else "\n"
        embeds = self.language_model.get_input_embeddings()
        suffix = self._tensor(self.encode(text))
        return torch.cat([embeds(trace.question_ids), trace.inputs, embeds(suffix)], dim=0)

    def answer_logprobs(self, trace, tokens, *, length=None):
        if tokens.ndim != 1 or len(tokens) == 0:
            raise ValueError("completion must contain tokens")
        prefix = self._output_prefix(trace)
        embedded = self.language_model.get_input_embeddings()(tokens[:-1])
        # Fresh forward prevents in-place DynamicCache reuse from changing the prefix.
        outputs = self._forward_embeds(torch.cat([prefix, embedded], 0))
        logits = outputs.logits[0, len(prefix) - 1:len(prefix) - 1 + len(tokens)].float()
        log_probs = F.log_softmax(logits, -1).gather(-1, tokens[:, None]).squeeze(-1)
        return log_probs, token_mask(tokens, self.eos_ids, length)

    @torch.no_grad()
    def generate(self, trace, *, sample=False):
        """Raw softmax: temperature=1, no top-k/top-p/other logits processors."""
        prefix = self._output_prefix(trace)
        output = self._forward_embeds(prefix)
        cache, total = output.past_key_values, len(prefix)
        tokens, probs, stopped = [], [], False
        for _ in range(self.config.max_new_tokens):
            log_probs = F.log_softmax(output.logits[0, -1].float(), -1)
            token = torch.multinomial(log_probs.exp(), 1).squeeze(0) if sample else log_probs.argmax()
            tokens.append(token)
            probs.append(log_probs[token])
            if int(token) in self.eos_ids:
                stopped = True
                break
            total += 1
            output = self._forward_embeds(self.language_model.get_input_embeddings()(token.view(1)),
                                          cache=cache, total_length=total)
            cache = output.past_key_values
        ids = torch.stack(tokens)
        return Completion(ids, torch.stack(probs), token_mask(ids, self.eos_ids),
                          self.tokenizer.decode(ids.tolist(), skip_special_tokens=True), stopped)

    def sft_loss(self, example, targets, *, noisy=False):
        trace = self.roles(example.question, stochastic=noisy,
                           sft_std=self.config.sft_noise_std if noisy else None)
        _, anchor_text, answer_text = output_parts(example, self.config)
        anchors = self.encode(anchor_text)
        answer = self.encode(answer_text) + [int(self.tokenizer.eos_token_id)]
        ids = self._tensor(anchors + answer)
        if len(ids) > self.config.max_target_tokens:
            raise ValueError("SFT target exceeds configured budget; no silent truncation")
        log_probs, mask = self.answer_logprobs(trace, ids)
        anchor_mask = torch.arange(len(ids), device=self.device) < len(anchors)
        anchor_loss = -(log_probs * mask * anchor_mask).sum() / self.config.answer_scale
        answer_loss = -(log_probs * mask * ~anchor_mask).sum() / self.config.answer_scale
        structural, terms = self.structure(trace, targets)
        total = answer_loss + self.config.anchor_weight * anchor_loss + structural
        return total, {"answer": answer_loss, "anchor": anchor_loss, "structure": structural, **terms}

    def teacher_tokens(self, example):
        ids = self.question_ids(example.question).tolist()
        boundaries = [len(ids) - 1]
        for step in example.steps:
            piece = self.encode(step)
            if not piece:
                raise ValueError("a CoT step tokenizes to an empty sequence")
            ids.extend(piece)
            boundaries.append(len(ids) - 1)
            ids.extend(self.encode("\n"))
        if len(ids) > self.config.max_teacher_tokens:
            raise ValueError("teacher CoT token budget exceeded; no silent truncation")
        return self._tensor(ids), boundaries

    @torch.no_grad()
    def teacher_target(self, example):
        ids, boundaries = self.teacher_tokens(example)
        outputs = self.language_model(input_ids=ids.unsqueeze(0), output_hidden_states=True, use_cache=False)
        states = self.normalize_state(outputs.hidden_states[-1][0, boundaries])
        return build_targets(states @ self.semantic_projection, self.config.solve_roles)

    def stage0_loss(self, example):
        ids, _ = self.teacher_tokens(example)
        question_length = len(self.question_ids(example.question))
        end = self._tensor(self.encode(f"Answer: {example.answer}") + [int(self.tokenizer.eos_token_id)])
        full = torch.cat([ids, end])
        out = self.language_model(input_ids=full[:-1].unsqueeze(0), use_cache=False)
        log_probs = F.log_softmax(out.logits[0].float(), -1).gather(-1, full[1:, None]).squeeze(-1)
        valid = torch.arange(len(log_probs), device=self.device) >= question_length - 1
        return -(log_probs * valid).sum() / self.config.max_target_tokens

    def checkpoint_state(self):
        """LoRA deltas + all new modules/references, never frozen base shards."""
        trainable_llm = {"language_model." + n for n, p in self.language_model.named_parameters() if p.requires_grad}
        return {n: value.detach().cpu().clone() for n, value in self.state_dict().items()
                if not n.startswith("language_model.") or n in trainable_llm}

    def load_checkpoint_state(self, state):
        expected = set(self.checkpoint_state())
        if set(state) != expected:
            raise ValueError("checkpoint tensors do not match model/adapter schema")
        result = self.load_state_dict(state, strict=False)
        if result.unexpected_keys or any(not k.startswith("language_model.") for k in result.missing_keys):
            raise ValueError("checkpoint is missing structured model state")


def load_model(model_path, config, device):
    """Offline local weights only. No change to the user's existing environment."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
    if tokenizer.eos_token_id is None:
        raise ValueError("tokenizer must define EOS")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.bfloat16 if torch.device(device).type == "cuda" else torch.float32
    llm = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True,
                                             trust_remote_code=False, dtype=dtype, attn_implementation="eager")
    llm = get_peft_model(llm, LoraConfig(r=config.lora_rank, lora_alpha=config.lora_alpha,
        lora_dropout=0.0, target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none", task_type="CAUSAL_LM"))
    return StructuredModel(llm, tokenizer, config).to(device)
