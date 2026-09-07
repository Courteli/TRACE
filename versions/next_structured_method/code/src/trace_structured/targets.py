"""One teacher-coordinate contract for role SFT, SOLVE geometry and reward."""
from dataclasses import dataclass
import torch
import torch.nn.functional as F


def contiguous_spans(length, count):
    if length < 1 or count < 1:
        raise ValueError("positive step and role counts required")
    base, extra = divmod(length, count)
    start, result = 0, []
    for i in range(count):
        end = start + base + int(i < extra)
        result.append((start, end))
        start = end
    return tuple(result)


@dataclass
class Targets:
    solve: torch.Tensor
    span_mask: torch.Tensor
    cumulative: torch.Tensor
    end: torch.Tensor

    def to(self, device):
        return Targets(*(x.to(device) for x in (self.solve, self.span_mask, self.cumulative, self.end)))

    def payload(self):
        return {k: getattr(self, k).detach().cpu() for k in ("solve", "span_mask", "cumulative", "end")}

    @classmethod
    def from_payload(cls, value):
        return cls(**value)


def build_targets(boundaries, solve_roles):
    """Boundaries already projected with the SAME fixed teacher/student map."""
    if boundaries.ndim != 2 or boundaries.shape[0] < 2 or not torch.isfinite(boundaries).all():
        raise ValueError("finite [initial + step boundaries, semantic] tensor required")
    boundaries = boundaries.detach().float()
    delta = boundaries[1:] - boundaries[:-1]
    spans = contiguous_spans(len(delta), solve_roles)
    solve = torch.stack([delta[a:b].sum(0) for a, b in spans])
    mask = torch.tensor([b > a for a, b in spans], device=boundaries.device)
    cumulative = solve.cumsum(0)
    if not torch.allclose(cumulative[-1], boundaries[-1] - boundaries[0], atol=2e-5, rtol=2e-5):
        raise ValueError("teacher partition does not close")
    return Targets(solve, mask, cumulative, boundaries[-1])


def distance(prediction, target):
    """Dimension-mean Huber + valid-direction cosine; zero targets retain Huber."""
    if prediction.shape != target.shape:
        raise ValueError("coordinate shapes do not match")
    prediction, target = prediction.float(), target.detach().float()
    huber = F.smooth_l1_loss(prediction, target, reduction="none").mean(-1)
    valid = target.norm(dim=-1) > 1e-6
    cosine = (1 - F.cosine_similarity(prediction, target, dim=-1)) * valid
    return huber + 0.1 * cosine


def masked_mean(value, mask):
    return (value * mask).sum() / mask.sum().clamp_min(1)


def role_quantities(projected_states, predicted_plan, targets):
    k = len(targets.solve)
    if projected_states.shape != (k + 2, targets.solve.shape[-1]) or predicted_plan.shape != targets.solve.shape:
        raise ValueError("role/semantic schema mismatch")
    # SOLVE starts after PLAN; PLAN is not an extra text-CoT transition.
    solve = projected_states[1:k + 1] - projected_states[:k]
    cumulative = projected_states[1:k + 1] - projected_states[0]
    plan_dist = masked_mean(distance(predicted_plan, targets.solve), targets.span_mask)
    solve_dist = distance(solve, targets.solve)
    path_dist = distance(cumulative, targets.cumulative)
    readout_dist = distance(projected_states[-1], targets.end)
    return plan_dist, solve_dist, path_dist, readout_dist


def structure_loss(projected_states, predicted_plan, targets, config):
    plan, solve, path, readout = role_quantities(projected_states, predicted_plan, targets)
    terms = {"plan": plan, "solve": masked_mean(solve, targets.span_mask),
             "path": masked_mean(path, targets.span_mask), "readout": readout}
    total = sum(getattr(config, f"{key}_weight") * value for key, value in terms.items())
    return total, terms


@torch.no_grad()
def process_scores(projected_states, fixed_plan_prediction, targets):
    plan, solve, path, readout = role_quantities(projected_states, fixed_plan_prediction, targets)
    scores = torch.cat([torch.exp(-plan).reshape(1), torch.exp(-0.5 * (solve + path)),
                        torch.exp(-readout).reshape(1)])
    valid = torch.cat([targets.span_mask.any().reshape(1), targets.span_mask,
                       torch.ones(1, dtype=torch.bool, device=scores.device)])
    return scores * valid, valid
