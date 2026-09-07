"""Explicit candidate settings, not claims of tuned hyperparameters."""
from dataclasses import asdict, dataclass, fields
import hashlib
import json
import math
from pathlib import Path


@dataclass(frozen=True)
class Config:
    solve_roles: int = 6
    action_dim: int = 16
    policy_width: int = 512
    role_embedding_dim: int = 64
    semantic_dim: int = 256
    projection_seed: int = 1701
    action_scale: float = 0.05
    query_scale: float = 0.1
    log_std_min: float = -2.5
    log_std_max: float = 0.5
    initial_log_std: float = -2.120263536200091  # log(0.12): match the default noisy-SFT path
    sft_noise_start: float = 0.5
    sft_noise_fraction: float = 0.25
    sft_noise_std: float = 0.12
    group_size: int = 8
    clip_epsilon: float = 0.12
    discount: float = 0.9
    process_weight: float = 0.15
    process_std_floor: float = 0.1
    advantage_bound: float = 2.0
    kl_weight: float = 0.02
    entropy_weight: float = 0.001
    replay_weight: float = 0.05
    plan_weight: float = 0.08
    solve_weight: float = 0.12
    path_weight: float = 0.14
    readout_weight: float = 0.1
    anchor_weight: float = 1.0
    anchor_count: int = 2
    anchor_max_chars: int = 72
    answer_scale: int = 96
    max_new_tokens: int = 160
    max_question_tokens: int = 768
    max_teacher_tokens: int = 4096
    max_target_tokens: int = 768
    seed: int = 1701
    global_batch_size: int = 4
    stage0_epochs: int = 3
    stage1_epochs: int = 10
    stage2_epochs: int = 10
    stage2_samples_per_epoch: int = 2048
    sft_lr: float = 1e-5
    rl_lr: float = 5e-7
    weight_decay: float = 0.01
    warmup_steps: int = 75
    grad_clip: float = 1.0
    save_every: int = 100
    lora_rank: int = 64
    lora_alpha: int = 32

    def __post_init__(self):
        for f in fields(self):
            v = getattr(self, f.name)
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
                raise ValueError(f"invalid numerical configuration: {f.name}")
            if isinstance(f.default, int) and not isinstance(v, int):
                raise ValueError(f"{f.name} must be an integer")
        positive = ("solve_roles", "action_dim", "policy_width", "role_embedding_dim", "semantic_dim",
                    "answer_scale", "max_new_tokens", "max_question_tokens", "max_teacher_tokens",
                    "max_target_tokens", "global_batch_size", "stage0_epochs", "stage1_epochs",
                    "stage2_epochs", "stage2_samples_per_epoch", "save_every", "lora_rank", "lora_alpha",
                    "process_std_floor", "advantage_bound", "sft_noise_std", "sft_lr", "rl_lr", "grad_clip")
        if any(getattr(self, k) <= 0 for k in positive) or self.group_size < 2:
            raise ValueError("positive sizes/scales and group_size >= 2 are required")
        if self.anchor_count not in (0, 1, 2) or self.anchor_max_chars <= 0:
            raise ValueError("anchor_count must be 0, 1 or 2")
        for k in ("sft_noise_start", "sft_noise_fraction", "discount"):
            if not 0 <= getattr(self, k) <= 1:
                raise ValueError(f"{k} must be in [0, 1]")
        if not self.log_std_min <= self.initial_log_std <= self.log_std_max:
            raise ValueError("initial log std must lie within policy bounds")
        if not 0 < self.clip_epsilon < 1:
            raise ValueError("clip_epsilon must be in (0, 1)")
        for k in ("action_scale", "query_scale", "process_weight", "kl_weight", "entropy_weight",
                  "replay_weight", "plan_weight", "solve_weight", "path_weight", "readout_weight",
                  "anchor_weight", "weight_decay", "warmup_steps", "seed", "projection_seed"):
            if getattr(self, k) < 0:
                raise ValueError(f"{k} cannot be negative")

    @property
    def role_names(self):
        return ("PLAN", *(f"SOLVE{i + 1}" for i in range(self.solve_roles)), "READOUT")

    def to_dict(self):
        return asdict(self)

    def fingerprint(self):
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()

    @classmethod
    def load(cls, path=None):
        return cls(**json.loads(Path(path).read_text())) if path else cls()

    def process_coefficient(self, completed, total):
        return self.process_weight * max(0.0, 1.0 - completed / max(total - 1, 1))
