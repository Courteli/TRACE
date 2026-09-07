"""Single-update group-relative optimization and explicitly reduced gradients."""
from dataclasses import dataclass
import math
import torch
import torch.distributed as dist

from .data import answer_matches
from .policy import gaussian_kl, group_advantage, process_returns, surrogate


@dataclass
class Rollout:
    actions: torch.Tensor
    action_log_probs: torch.Tensor
    completion: object


def add_metrics(destination, values, weight=1.0):
    for key, value in values.items():
        destination[key] = destination.get(key, 0.0) + float(value.detach() if torch.is_tensor(value) else value) * weight


def check_finite(loss):
    if not torch.isfinite(loss).all():
        raise FloatingPointError("nonfinite loss; keep the last validated checkpoint")


def gradient_probe(loss, model):
    """Small named head probe, NOT a claim about full-model gradient norms."""
    parameters = [model.policy.means["plan"].weight, model.policy.means["solve"].weight,
                  model.policy.means["readout"].weight, model.plan_head.weight]
    grads = torch.autograd.grad(loss, parameters, allow_unused=True, retain_graph=True)
    return sum(float(g.detach().float().square().sum()) for g in grads if g is not None) ** 0.5


def rl_backward(model, example, target, completed_step, total_steps, *, audit_gradients=False):
    """Accumulate this question's mean-over-G gradient; caller averages questions.

    No parameter update occurs between collecting and replaying this group.
    """
    config = model.config
    model.train()
    rows, rewards, scores, valid = [], [], [], []
    with torch.no_grad():
        for _ in range(config.group_size):
            trace = model.roles(example.question, stochastic=True)
            completion = model.generate(trace, sample=True)
            score, mask = model.score_process(trace, target)
            rows.append(Rollout(trace.actions.detach(), trace.log_probs.detach(), completion))
            rewards.append(float(answer_matches(completion.text, example.answer)))
            scores.append(score)
            valid.append(mask)
    reward = torch.tensor(rewards, device=model.device)
    answer_adv = group_advantage(reward, config.group_size)
    returns = process_returns(torch.stack(scores), torch.stack(valid), config.discount)
    process_adv = group_advantage(returns, config.group_size, std_floor=config.process_std_floor,
                                  bound=config.advantage_bound)
    coefficient = config.process_coefficient(completed_step, total_steps)
    metrics = {"exact_accuracy": float(reward.mean()), "process_score": float(torch.stack(scores).sum() / torch.stack(valid).sum().clamp_min(1)),
               "process_weight": coefficient, "answer_adv_std": float(answer_adv.std(unbiased=False)),
               "process_adv_max": float(process_adv.abs().max()), "sampled_answer_tokens": sum(len(x.completion.tokens) for x in rows) / len(rows)}
    for index, row in enumerate(rows):
        trace = model.roles(example.question, forced_actions=row.actions)
        current, mask = model.answer_logprobs(trace, row.completion.tokens)
        latent_adv = answer_adv[index] + coefficient * process_adv[index]
        answer_loss = surrogate(current, row.completion.log_probs, answer_adv[index], mask,
                                config.clip_epsilon, config.answer_scale)
        latent_loss = surrogate(trace.log_probs[:-1], row.action_log_probs[:-1], latent_adv,
                                torch.ones_like(latent_adv, dtype=torch.bool), config.clip_epsilon,
                                config.solve_roles + 1)
        latent_answer = surrogate(trace.log_probs[:-1], row.action_log_probs[:-1], answer_adv[index],
                                  torch.ones_like(latent_adv, dtype=torch.bool), config.clip_epsilon,
                                  config.solve_roles + 1)
        # Exact incremental effect of adding process advantage to the combined
        # clipped surrogate; do not assume clipping is algebraically additive.
        process_increment = latent_loss - latent_answer
        kl_terms = []
        for role in range(config.solve_roles + 1):
            with torch.no_grad():
                ref_mean, ref_std = model.reference_policy.distribution(trace.pre_states[role].detach(), role)
            kl_terms.append(gaussian_kl(trace.means[role], trace.log_stds[role], ref_mean, ref_std))
        kl = torch.stack(kl_terms).mean()
        entropy = (trace.log_stds[:-1] + 0.5 * (1 + math.log(2 * math.pi))).sum(-1).mean()
        total = answer_loss + latent_loss + config.kl_weight * kl - config.entropy_weight * entropy
        check_finite(total)
        if audit_gradients and index == 0:
            for name, term in (("answer", answer_loss), ("latent", latent_loss),
                               ("latent_answer", latent_answer), ("process_increment", process_increment),
                               ("kl", config.kl_weight * kl), ("entropy", -config.entropy_weight * entropy)):
                metrics[f"head_probe_grad/{name}"] = gradient_probe(term, model)
        (total / config.group_size).backward()
        add_metrics(metrics, {"answer_policy_loss": answer_loss, "latent_policy_loss": latent_loss,
                              "latent_answer_loss": latent_answer, "process_increment_loss": process_increment,
                              "role_kl": kl, "entropy": entropy, "total_loss": total}, 1 / config.group_size)
        metrics["old_current_logprob_max_error"] = max(metrics.get("old_current_logprob_max_error", 0.0),
            float((current.detach() - row.completion.log_probs).abs().max()),
            float((trace.log_probs[:-1].detach() - row.action_log_probs[:-1]).abs().max()))
    if config.replay_weight:
        replay, _ = model.sft_loss(example, target, noisy=False)
        weighted = config.replay_weight * replay
        check_finite(weighted)
        if audit_gradients:
            metrics["head_probe_grad/replay"] = gradient_probe(weighted, model)
        weighted.backward()
        add_metrics(metrics, {"sft_replay_loss": replay, "total_loss": weighted})
    return metrics


def sft_backward(model, example, target, completed_step, total_steps):
    config = model.config
    enabled = completed_step / max(total_steps, 1) >= config.sft_noise_start
    noisy = enabled and bool(torch.rand((), device=model.device) < config.sft_noise_fraction)
    loss, metrics = model.sft_loss(example, target, noisy=noisy)
    check_finite(loss)
    loss.backward()
    return {**{k: float(v.detach()) for k, v in metrics.items()}, "total_loss": float(loss.detach()), "noisy_path": float(noisy)}


def world():
    return (dist.get_rank(), dist.get_world_size()) if dist.is_initialized() else (0, 1)


def reduce_gradients(model, local_questions):
    """SUM gradients / actual global question count; no padded train examples.

    Parameters unused on every rank keep grad=None (no accidental Adam decay).
    """
    parameters = [p for p in model.parameters() if p.requires_grad]
    count = torch.tensor(float(local_questions), device=model.device)
    present = torch.tensor([p.grad is not None for p in parameters], dtype=torch.int32, device=model.device)
    if dist.is_initialized():
        dist.all_reduce(count, op=dist.ReduceOp.SUM)
        dist.all_reduce(present, op=dist.ReduceOp.MAX)
    if count.item() <= 0:
        raise ValueError("empty global batch")
    for parameter, active in zip(parameters, present.tolist()):
        if not active:
            continue
        if parameter.grad is None:
            parameter.grad = torch.zeros_like(parameter)
        if dist.is_initialized():
            dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
        parameter.grad.div_(count)
        if not torch.isfinite(parameter.grad).all():
            raise FloatingPointError("nonfinite globally reduced gradient")
    return int(count.item())


def epoch_indices(size, seed, epoch, limit=None):
    generator = torch.Generator().manual_seed(seed + epoch)
    order = torch.randperm(size, generator=generator).tolist()
    return order[:min(size, limit)] if limit is not None else order


def make_scheduler(optimizer, warmup, total):
    def multiplier(step):
        if warmup and step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(total - warmup, 1)
        return 0.5 * (1 + math.cos(math.pi * min(max(progress, 0), 1)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)
