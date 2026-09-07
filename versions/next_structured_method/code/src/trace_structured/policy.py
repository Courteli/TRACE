"""Shared SOLVE Gaussian policy and bounded group-relative credit."""
import math
import torch
from torch import nn


def gaussian_logprob(action, mean, log_std):
    return (-0.5 * (((action - mean) * (-log_std).exp()).square()
                   + 2 * log_std + math.log(2 * math.pi))).sum(-1)


def gaussian_kl(mean, log_std, ref_mean, ref_log_std):
    return (ref_log_std - log_std + 0.5 * (
        (2 * (log_std - ref_log_std)).exp()
        + (mean - ref_mean).square() * (-2 * ref_log_std).exp() - 1)).mean(-1)


class RolePolicy(nn.Module):
    def __init__(self, hidden, config):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(len(config.role_names), config.role_embedding_dim)
        self.trunk = nn.Sequential(nn.LayerNorm(hidden + config.role_embedding_dim),
                                   nn.Linear(hidden + config.role_embedding_dim, config.policy_width),
                                   nn.GELU(), nn.Linear(config.policy_width, config.policy_width), nn.LayerNorm(config.policy_width))
        self.means = nn.ModuleDict({k: nn.Linear(config.policy_width, config.action_dim) for k in ("plan", "solve", "readout")})
        self.log_stds = nn.ModuleDict({k: nn.Linear(config.policy_width, config.action_dim) for k in ("plan", "solve")})
        self.action_projection = nn.Sequential(nn.Linear(config.action_dim, hidden), nn.Tanh())
        for head in self.means.values():
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
        for head in self.log_stds.values():
            nn.init.zeros_(head.weight)
            nn.init.constant_(head.bias, config.initial_log_std)

    def distribution(self, state, index):
        if not 0 <= index < len(self.config.role_names):
            raise ValueError("invalid role index")
        key = "plan" if index == 0 else "readout" if index == self.config.solve_roles + 1 else "solve"
        features = self.trunk(torch.cat([state.float(), self.embedding.weight[index]], -1))
        mean = self.means[key](features)
        log_std = torch.zeros_like(mean) if key == "readout" else self.log_stds[key](features).clamp(
            self.config.log_std_min, self.config.log_std_max)
        return mean, log_std

    def realize(self, state, index, *, stochastic=False, forced=None, sft_std=None):
        mean, log_std = self.distribution(state, index)
        if index == self.config.solve_roles + 1:
            # Recompute even when a saved READOUT action is supplied.
            return mean, mean.sum() * 0, mean, log_std
        if sft_std is not None:
            if sft_std <= 0:
                raise ValueError("SFT noise std must be positive")
            log_std = torch.full_like(log_std, math.log(sft_std))
        if forced is not None:
            action = forced.detach().float()
        elif stochastic:
            action = mean + log_std.exp() * torch.randn_like(mean)
        else:
            action = mean
        return action, gaussian_logprob(action, mean, log_std), mean, log_std


def group_advantage(values, group_size, *, std_floor=1e-4, bound=None):
    if group_size < 2 or values.shape[0] % group_size:
        raise ValueError("complete same-question groups required")
    grouped = values.detach().reshape(-1, group_size, *values.shape[1:])
    centered = grouped - grouped.mean(1, keepdim=True)
    scaled = centered / grouped.std(1, keepdim=True, unbiased=False).clamp_min(std_floor)
    return (scaled.clamp(-bound, bound) if bound is not None else scaled).reshape_as(values)


def process_returns(scores, valid, gamma):
    """Normalized discounted process return, including terminal READOUT once.

    Terminal ANSWER reward is a separate channel, never added here.
    """
    if scores.shape != valid.shape or scores.ndim != 2 or not 0 <= gamma <= 1:
        raise ValueError("invalid process-return inputs")
    out = torch.zeros_like(scores)
    running = torch.zeros_like(scores[:, 0])
    weight = running.clone()
    for index in range(scores.shape[1] - 1, -1, -1):
        running = scores[:, index] * valid[:, index] + gamma * running
        weight = valid[:, index].float() + gamma * weight
        out[:, index] = running / weight.clamp_min(1e-8)
    # READOUT has no stochastic likelihood, but contributes to earlier returns.
    return out[:, :-1]


def surrogate(current, old, advantage, mask, epsilon, fixed_scale):
    """Sum valid likelihood terms; never divide by a response's own length."""
    if current.shape != old.shape or current.shape != mask.shape or fixed_scale <= 0:
        raise ValueError("invalid policy reduction inputs")
    ratio = (current - old.detach()).clamp(-20, 20).exp()
    terms = torch.minimum(ratio * advantage.detach(), ratio.clamp(1 - epsilon, 1 + epsilon) * advantage.detach())
    return -(terms * mask).sum() / fixed_scale
