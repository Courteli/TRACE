import ast
import copy
import ctypes
import gc
import hashlib
import json
import math
import re
from contextlib import contextmanager, nullcontext
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils.checkpoint import checkpoint as activation_checkpoint

from .read_stable_efficient import LitREADCoTStableEfficient
from ..modules.readcot import (
    aggregate_step_residuals,
    dependency_bce_loss,
    dependency_f1_score,
    reconstruct_dependency_logits,
)
from ..modules.trace_policy import (
    CoTConditionedTrajectoryPosterior,
    GaussianTrajectoryPolicy,
    HardPathPair,
    action_conditioned_progress_centers,
    action_transition_identifiability_loss,
    action_transition_retrieval_accuracy,
    build_transition_advantages,
    clipped_policy_loss,
    counterfactual_action_batch,
    counterfactual_transition_credits,
    diagonal_gaussian_kl,
    gaussian_log_prob,
    group_standardize,
    group_standardize_with_floor,
    minimum_action_entropy_loss,
    mine_question_local_hard_pairs,
    pairwise_action_path_correlation,
    path_noncollapse_loss,
    sampled_forward_kl,
    stochastic_monotone_assignment,
    trajectory_distance,
    trajectory_distance_components,
)


_ARITHMETIC_EXPRESSION_NODES = (
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    ast.Constant,
    ast.Name,
    ast.Load,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.UAdd,
    ast.USub,
)
_ARITHMETIC_TOKEN_PATTERN = re.compile(
    r"\d+(?:\.\d+)?|\.\d+|[+\-*/=()]"
)


def _validated_arithmetic_ast(expression: str):
    normalized = expression.replace("$", "").replace(",", "")
    try:
        tree = ast.parse(normalized, mode="eval")
    except SyntaxError:
        return None
    if any(
        not isinstance(node, _ARITHMETIC_EXPRESSION_NODES)
        for node in ast.walk(tree)
    ):
        return None
    return tree


def _evaluate_arithmetic_ast(tree) -> Optional[Decimal]:
    """Evaluate the restricted arithmetic grammar without using ``eval``."""

    def evaluate(node):
        if isinstance(node, ast.Expression):
            return evaluate(node.body)
        if isinstance(node, ast.Constant) and isinstance(
            node.value,
            (int, float),
        ):
            return Decimal(str(node.value))
        if isinstance(node, ast.UnaryOp) and isinstance(
            node.op,
            (ast.UAdd, ast.USub),
        ):
            value = evaluate(node.operand)
            return value if isinstance(node.op, ast.UAdd) else -value
        if isinstance(node, ast.BinOp):
            left = evaluate(node.left)
            right = evaluate(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                return left / right
        raise ValueError("expression is not fully numeric")

    try:
        return evaluate(tree).normalize()
    except (ArithmeticError, InvalidOperation, ValueError, ZeroDivisionError):
        return None


def is_numerically_valid_equation(
    equation: str,
    *,
    absolute_tolerance: Decimal = Decimal("1e-8"),
    relative_tolerance: Decimal = Decimal("1e-6"),
) -> bool:
    """Require an extracted numeric equation to be arithmetically true."""
    if equation.count("=") != 1:
        return False
    lhs, rhs = equation.split("=", 1)
    tree = _validated_arithmetic_ast(lhs)
    expected = _decimal_key(rhs)
    if tree is None or expected is None:
        return False
    observed = _evaluate_arithmetic_ast(tree)
    if observed is None:
        return False
    tolerance = max(
        absolute_tolerance,
        abs(expected) * relative_tolerance,
    )
    return abs(observed - expected) <= tolerance


def _decimal_key(text: str) -> Optional[Decimal]:
    normalized = str(text).replace("$", "").replace(",", "").strip()
    try:
        return Decimal(normalized).normalize()
    except InvalidOperation:
        return None


def _arithmetic_literals(expression: str) -> List[Decimal]:
    """Read numeric operands from a validated arithmetic AST."""
    tree = _validated_arithmetic_ast(expression)
    if tree is None:
        return []

    values: List[Decimal] = []

    class LiteralVisitor(ast.NodeVisitor):
        def visit_UnaryOp(self, node):
            if (
                isinstance(node.op, (ast.USub, ast.UAdd))
                and isinstance(node.operand, ast.Constant)
                and isinstance(node.operand.value, (int, float))
            ):
                value = Decimal(str(node.operand.value))
                if isinstance(node.op, ast.USub):
                    value = -value
                values.append(value.normalize())
                return
            self.generic_visit(node)

        def visit_Constant(self, node):
            if isinstance(node.value, (int, float)):
                values.append(Decimal(str(node.value)).normalize())

    LiteralVisitor().visit(tree)
    return values


def extract_unit_normalized_equations(
    steps: Sequence[str],
) -> List[str]:
    """Extract ordered arithmetic while ignoring natural-language units."""
    equations = []
    for raw_step in steps:
        text = (
            str(raw_step)
            .replace("×", "*")
            .replace("÷", "/")
            .replace("–", "-")
            .replace("−", "-")
        )
        text = re.sub(
            r"(?<=[A-Za-z])[-‐‑](?=[A-Za-z])",
            " ",
            text,
        )
        # A slash inside a unit, such as snakes/jaguar, is not division.
        text = re.sub(
            r"(?<=[A-Za-z])\s*/\s*(?=[A-Za-z])",
            " ",
            text,
        )
        text = re.sub(
            r"\bdivided\s+by\b",
            "/",
            text,
            flags=re.IGNORECASE,
        )
        text = re.sub(
            r"\b(?:multiplied\s+by|times)\b",
            "*",
            text,
            flags=re.IGNORECASE,
        )
        text = re.sub(
            r"\bplus\b",
            "+",
            text,
            flags=re.IGNORECASE,
        )
        text = re.sub(
            r"\bminus\b",
            "-",
            text,
            flags=re.IGNORECASE,
        )
        text = re.sub(
            r"\bequals\b",
            "=",
            text,
            flags=re.IGNORECASE,
        )
        text = re.sub(
            r"(?<=\s)[xX](?=\s*(?:\$?\d|\.\d))",
            "*",
            text,
        )
        text = re.sub(
            r"(\$?(?:\d[\d,]*(?:\.\d+)?|\.\d+))\s*%",
            r"(\1/100)",
            text,
        )
        text = text.replace("%", "")
        text = text.replace("$", "").replace(",", "")
        text = re.sub(r"[A-Za-z]+", " ", text)
        text = re.sub(r"(?<=\d)\s*(?=\()", "*", text)
        tokens = _ARITHMETIC_TOKEN_PATTERN.findall(text)
        current = []
        index = 0
        while index < len(tokens):
            token = tokens[index]
            if token == "=":
                rhs_index = index + 1
                rhs_sign = ""
                if (
                    rhs_index < len(tokens)
                    and tokens[rhs_index] in {"+", "-"}
                ):
                    rhs_sign = tokens[rhs_index]
                    rhs_index += 1
                if (
                    rhs_index < len(tokens)
                    and (
                        tokens[rhs_index][0].isdigit()
                        or tokens[rhs_index].startswith(".")
                    )
                    and any(part in {"+", "-", "*", "/"} for part in current)
                ):
                    rhs = tokens[rhs_index]
                    if rhs.startswith("."):
                        rhs = "0" + rhs
                    rhs = rhs_sign + rhs
                    lhs = "".join(current)
                    equation = f"{lhs}={rhs}"
                    if (
                        is_numerically_valid_equation(equation)
                        and equation not in equations
                    ):
                        equations.append(equation)
                    current = [rhs]
                    index = rhs_index + 1
                    continue
                current = []
                index += 1
                continue
            is_number = token[0].isdigit() or token.startswith(".")
            if is_number:
                if token.startswith("."):
                    token = "0" + token
                previous_is_operand = bool(current) and (
                    current[-1][0].isdigit()
                    or current[-1].startswith(".")
                    or current[-1] == ")"
                )
                if previous_is_operand:
                    current = []
            current.append(token)
            index += 1
    return equations


def answer_causal_equation_slice(
    equations: Sequence[str],
    answer: str,
) -> List[str]:
    """Keep the equation DAG needed to produce the registered answer.

    Equations are traversed backwards. A result is retained when it is the
    answer or appears as an arithmetic operand of a retained later equation.
    This removes explanatory side calculations without dropping intermediate
    operations that the final answer actually depends on.
    """
    records = []
    for equation in equations:
        if "=" not in equation:
            continue
        lhs, rhs = equation.rsplit("=", 1)
        result = _decimal_key(rhs)
        if result is None:
            continue
        records.append(
            {
                "equation": equation,
                "result": result,
                "operands": _arithmetic_literals(lhs),
            }
        )
    desired = _decimal_key(answer)
    if desired is None or not records:
        return list(equations)

    needed = {desired}
    selected = []
    for index in range(len(records) - 1, -1, -1):
        record = records[index]
        if record["result"] not in needed:
            continue
        selected.append(index)
        needed.discard(record["result"])
        earlier_results = {
            earlier["result"] for earlier in records[:index]
        }
        needed.update(
            operand
            for operand in record["operands"]
            if operand in earlier_results
        )
    if not selected:
        return list(equations)
    return [
        records[index]["equation"] for index in sorted(selected)
    ]


@contextmanager
def selective_saved_activation_offload(
    *,
    minimum_bytes: int,
    pin_memory: bool,
):
    """Offload large dense activations while leaving SDPA metadata in place."""

    def pack(tensor: torch.Tensor):
        nbytes = tensor.numel() * tensor.element_size()
        should_offload = (
            tensor.device.type == "cuda"
            and tensor.is_floating_point()
            and not tensor.is_leaf
            and tensor.ndim in (2, 3, 4)
            and nbytes >= minimum_bytes
        )
        if not should_offload:
            return tensor
        cpu_tensor = tensor.detach().to(
            device="cpu",
            non_blocking=False,
            copy=True,
        )
        if pin_memory:
            cpu_tensor = cpu_tensor.pin_memory()
        return (
            "trace_saved_activation_cpu",
            tensor.device,
            cpu_tensor,
            pin_memory,
        )

    def unpack(packed):
        if (
            not isinstance(packed, tuple)
            or len(packed) != 4
            or packed[0] != "trace_saved_activation_cpu"
        ):
            return packed
        _, device, cpu_tensor, was_pinned = packed
        return cpu_tensor.to(
            device=device,
            non_blocking=was_pinned,
        )

    with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
        yield


def build_path_bottleneck_mask(
    question_attention_mask: torch.Tensor,
    latent_attention_mask: torch.Tensor,
    answer_attention_mask: torch.Tensor,
    *,
    include_question: bool = False,
) -> torch.Tensor:
    """Build the answer decoder's source mask.

    TRACE-Policy v3 uses ``include_question=False``: question K/V remains in
    the frozen prefix cache for latent formation but is masked from answer
    queries, which may read only the complete latent path and prior answers.
    """
    if not (
        question_attention_mask.ndim
        == latent_attention_mask.ndim
        == answer_attention_mask.ndim
        == 2
    ):
        raise ValueError("all attention masks must be two-dimensional")
    if not (
        question_attention_mask.shape[0]
        == latent_attention_mask.shape[0]
        == answer_attention_mask.shape[0]
    ):
        raise ValueError("all attention masks must share a batch dimension")
    return torch.cat(
        [
            (
                question_attention_mask
                if include_question
                else torch.zeros_like(question_attention_mask)
            ),
            latent_attention_mask,
            answer_attention_mask,
        ],
        dim=1,
    )


def _right_pad(
    tensors: Sequence[torch.Tensor],
    *,
    value: float,
) -> torch.Tensor:
    if not tensors:
        raise ValueError("cannot pad an empty tensor sequence")
    max_length = max(tensor.shape[1] for tensor in tensors)
    padded = []
    for tensor in tensors:
        if tensor.shape[1] == max_length:
            padded.append(tensor)
            continue
        shape = list(tensor.shape)
        shape[1] = max_length - tensor.shape[1]
        extension = torch.full(
            shape,
            fill_value=value,
            device=tensor.device,
            dtype=tensor.dtype,
        )
        padded.append(torch.cat([tensor, extension], dim=1))
    return torch.cat(padded, dim=0)


def extract_stage2_policy_reference(checkpoint: dict):
    """Return the immutable prior only from a genuine Stage-2 checkpoint."""
    checkpoint_stage = int(
        checkpoint.get("trace_policy_training_stage", 1)
    )
    if checkpoint_stage != 2:
        return None
    reference = checkpoint.get("trace_stage1_policy_reference")
    if reference is None:
        raise RuntimeError(
            "Stage-2 checkpoint is missing its immutable Stage-1 "
            "policy reference"
        )
    return reference


def summarize_unique_validation_records(
    shards: Sequence[Sequence[Tuple[int, float, int]]],
    *,
    expected_count: int,
) -> Dict[str, float]:
    """Deduplicate DDP padding and require one deterministic result per item."""
    records: Dict[int, Tuple[float, int]] = {}
    for shard in shards:
        for index, accuracy, output_length in shard:
            value = (float(accuracy), int(output_length))
            if index in records and records[index] != value:
                raise RuntimeError(
                    f"validation result for index {index} is nondeterministic"
                )
            records[int(index)] = value
    if len(records) != int(expected_count):
        raise RuntimeError(
            "full validation contract failed: "
            f"found {len(records)} unique questions, "
            f"expected {expected_count}"
        )
    return {
        "accuracy": float(
            np.mean([value[0] for value in records.values()])
        ),
        "output_length": float(
            np.mean([value[1] for value in records.values()])
        ),
        "unique_questions": float(len(records)),
    }


class LitTRACEPolicy(LitREADCoTStableEfficient):
    """TRACE-Policy v3 with deployment-consistent trajectory refinement.

    Stage 1 samples latent paths from a single-CoT-conditioned posterior and
    distills them into a question-only autoregressive Gaussian prior. Stage 2
    treats each prior action as the optimized policy action and assigns
    transition-specific credit through causal counterfactual suffixes.
    """

    path_adapter_name = "default"
    cot_encoder_adapter_name = "trace_cot_encoder"
    answer_adapter_name = "trace_answer"
    trace_policy_version = "TRACE-Policy-v3"

    @staticmethod
    def _trace_position_ids(
        attention_mask: torch.Tensor,
        current_length: int,
    ) -> torch.Tensor:
        """Position current tokens from the unmasked causal source sequence."""
        positions = attention_mask.long().cumsum(dim=-1) - 1
        positions = positions.masked_fill(attention_mask == 0, 0)
        return positions[:, -int(current_length) :]

    def __init__(self, model_kwargs, training_kwargs, all_config=None):
        super().__init__(
            model_kwargs=model_kwargs,
            training_kwargs=training_kwargs,
            all_config=all_config,
        )
        self.trace_config = model_kwargs.get("trace_policy_config", {})
        self.trace_rl_config = model_kwargs.get("trace_rl_config", {})
        self.do_trace_rl = bool(model_kwargs.get("do_trace_rl", False))
        # TRACE does not use the inherited deterministic bridge or its random
        # residual projection. Keeping those dormant parameters would make the
        # compact-target selector depend on an untrained legacy branch.
        self.latent_bridge = torch.nn.Identity()
        self.residual_projector = torch.nn.Identity()
        # Dynamic TRACE corridors query with the realized student path, never a
        # table of fixed latent slots.
        self.step_compressor.latent_queries.requires_grad_(False)
        if not self.model_kwargs.get("do_lora", False):
            raise ValueError(
                "TRACE requires LoRA to isolate teacher, path, and answer roles"
            )
        if self.cot_encoder_adapter_name not in self.llm.peft_config:
            cot_encoder_config = copy.deepcopy(
                self.llm.peft_config[self.path_adapter_name]
            )
            self.llm.add_adapter(
                self.cot_encoder_adapter_name,
                cot_encoder_config,
            )
        self._match_adapter_storage_to_path(
            self.cot_encoder_adapter_name
        )
        self._set_adapter_parameter_trainability()
        self.n_trace_steps = int(self.readcot_config.n_latents)
        if self.n_trace_steps <= 0:
            raise ValueError("TRACE requires at least one latent transition")
        if self.readcot_config.get("implicit_latent_mode") == "block":
            # TRACE's action distribution is autoregressive even if the source
            # BRIDGE config used a block latent implementation.
            self.readcot_config.implicit_latent_mode = "autoregressive-policy"
        if self.readcot_config.get("use_anchor_gate", False):
            raise ValueError("TRACE policy does not use a route gate")
        answer_context_mode = str(
            self.trace_config.get(
                "answer_context_mode",
                "question_and_path",
            )
        )
        if answer_context_mode not in {
            "question_and_path",
            "path_only",
        }:
            raise ValueError(
                "answer_context_mode must be question_and_path or path_only"
            )
        self.answer_context_mode = answer_context_mode
        self.answer_reads_question = (
            answer_context_mode == "question_and_path"
        )
        self.stage1_target_mode = str(
            self.trace_config.get("stage1_target_mode", "")
        )
        if self.stage1_target_mode != (
            "complete_answer_causal_arithmetic_trace"
        ):
            raise ValueError(
                "TRACE-Policy v3 requires one complete answer-causal "
                "arithmetic target per question"
            )
        if bool(
            self.trace_config.get("require_path_bottleneck", False)
        ) and self.answer_reads_question:
            raise ValueError(
                "require_path_bottleneck=true forbids direct question K/V "
                "at the answer decoder"
            )
        self.trajectory_policy = GaussianTrajectoryPolicy(
            hidden_size=self.hidden_size,
            action_dim=int(self.trace_config.get("action_dim", 16)),
            n_steps=self.n_trace_steps,
            policy_hidden_size=int(
                self.trace_config.get("policy_hidden_size", 512)
            ),
            step_embedding_size=int(
                self.trace_config.get("policy_step_embedding_size", 64)
            ),
            initial_log_std=float(
                self.trace_config.get("initial_log_std", -0.7)
            ),
            min_log_std=float(
                self.trace_config.get("min_log_std", -2.5)
            ),
            max_log_std=float(
                self.trace_config.get("max_log_std", 0.5)
            ),
            minimum_action_gate=float(
                self.trace_config.get("minimum_action_gate", 0.08)
            ),
            maximum_action_gate=float(
                self.trace_config.get("maximum_action_gate", 0.40)
            ),
            initial_action_gate=float(
                self.trace_config.get("initial_action_gate", 0.20)
            ),
        )
        self.trajectory_posterior = CoTConditionedTrajectoryPosterior(
            hidden_size=self.hidden_size,
            action_dim=self.trajectory_policy.action_dim,
            n_steps=self.n_trace_steps,
            posterior_hidden_size=int(
                self.trace_config.get("posterior_hidden_size", 512)
            ),
            step_embedding_size=int(
                self.trace_config.get(
                    "posterior_step_embedding_size",
                    64,
                )
            ),
            min_log_std=float(
                self.trace_config.get("posterior_min_log_std", -1.5)
            ),
            max_log_std=float(
                self.trace_config.get("posterior_max_log_std", 0.5)
            ),
        )
        self.posterior_context_norm = torch.nn.LayerNorm(
            self.hidden_size
        )
        self.stage1_policy_reference = copy.deepcopy(
            self.trajectory_policy
        )
        for parameter in self.stage1_policy_reference.parameters():
            parameter.requires_grad_(False)

        self.transition_action_decoder = torch.nn.Sequential(
            torch.nn.LayerNorm(self.hidden_size),
            torch.nn.Linear(
                self.hidden_size,
                self.trajectory_policy.action_dim,
            ),
        )
        torch.nn.init.zeros_(self.transition_action_decoder[-1].bias)
        self._reference_restored = False
        self._loaded_stage2_state = False
        self._cot_encoder_adapter_loaded = False
        self._stage2_initialized = False
        self._last_trace_metrics: Dict[str, torch.Tensor] = {}
        self._trace_visual_records: List[dict] = []
        self._validation_question_records: List[
            Tuple[int, float, int]
        ] = []
        self.strict_loading = False

        if self.do_trace_rl:
            if self.answer_reads_question:
                raise ValueError(
                    "TRACE-Policy v3 Stage 2 requires "
                    "answer_context_mode=path_only"
                )
            required_stage2_objectives = (
                "use_trajectory_policy_loss",
                "use_answer_policy_loss",
            )
            disabled = [
                key
                for key in required_stage2_objectives
                if not bool(self.trace_rl_config.get(key, False))
            ]
            if disabled:
                raise ValueError(
                    "Final TRACE Stage 2 requires both latent-action and "
                    f"answer-token policy objectives; disabled: {disabled}"
                )
            if int(
                self.trace_rl_config.get("policy_update_epochs", 1)
            ) < 1:
                raise ValueError(
                    "Final TRACE requires at least one on-policy update per "
                    "rollout"
                )
            positive_stage2_weights = (
                "dense_outcome_weight",
                "answer_policy_weight",
                "stage1_policy_kl_weight",
                "stage1_answer_kl_weight",
                "minimum_gold_score_std",
                "minimum_gold_score_gap",
            )
            invalid_weights = [
                key
                for key in positive_stage2_weights
                if float(self.trace_rl_config.get(key, 0.0)) <= 0.0
            ]
            if invalid_weights:
                raise ValueError(
                    "Final TRACE requires dense path credit and a joint "
                    "Stage-1 trust region; non-positive weights: "
                    f"{invalid_weights}"
                )
            steps_per_pair = int(
                self.trace_rl_config.get(
                    "counterfactual_steps_per_pair",
                    self.n_trace_steps,
                )
            )
            if not 1 <= steps_per_pair <= self.n_trace_steps:
                raise ValueError(
                    "counterfactual_steps_per_pair must lie within the path"
                )
            self._initialize_stage2_modules()
            self.automatic_optimization = False

    def _initialize_stage2_modules(self):
        if not self.model_kwargs.get("do_lora", False):
            raise ValueError(
                "Stage 2 answer-token GRPO requires phase-isolated LoRA"
            )
        if self.answer_adapter_name not in self.llm.peft_config:
            adapter_config = copy.deepcopy(
                self.llm.peft_config[self.path_adapter_name]
            )
            self.llm.add_adapter(
                self.answer_adapter_name,
                adapter_config,
            )
        self._match_adapter_storage_to_path(self.answer_adapter_name)

        for parameter in self.parameters():
            parameter.requires_grad_(False)
        self.trajectory_policy.set_stage2_trainability()
        self._set_adapter_parameter_trainability()
        self._activate_answer_adapter()

    def _set_adapter_parameter_trainability(self):
        path_marker = f".{self.path_adapter_name}."
        cot_encoder_marker = f".{self.cot_encoder_adapter_name}."
        answer_marker = f".{self.answer_adapter_name}."
        for name, parameter in self.llm.named_parameters():
            if cot_encoder_marker in name:
                parameter.requires_grad_(False)
            elif path_marker in name:
                parameter.requires_grad_(not self.do_trace_rl)
            elif answer_marker in name:
                parameter.requires_grad_(self.do_trace_rl)

    def _activate_path_adapter(self):
        self.llm.set_adapter(self.path_adapter_name)
        self._set_adapter_parameter_trainability()

    def _activate_cot_encoder_adapter(self):
        self.llm.set_adapter(self.cot_encoder_adapter_name)
        self._set_adapter_parameter_trainability()

    def _activate_answer_adapter(self):
        if not self.do_trace_rl:
            return
        self.llm.set_adapter(self.answer_adapter_name)
        self._set_adapter_parameter_trainability()

    def _copy_path_adapter(self, target_adapter_name: str):
        self._match_adapter_storage_to_path(target_adapter_name)
        parameters = dict(self.llm.named_parameters())
        path_marker = f".{self.path_adapter_name}."
        target_marker = f".{target_adapter_name}."
        copied = 0
        with torch.no_grad():
            for name, value in list(parameters.items()):
                if path_marker not in name:
                    continue
                target_name = name.replace(path_marker, target_marker)
                if (
                    target_name in parameters
                    and parameters[target_name].shape == value.shape
                ):
                    parameters[target_name].copy_(value.detach())
                    copied += 1
        if copied == 0:
            raise RuntimeError(
                f"could not map the path adapter to {target_adapter_name}"
            )
        return copied

    def _match_adapter_storage_to_path(self, target_adapter_name: str):
        """Match added PEFT adapters to the Stage-0 path adapter dtype."""
        parameters = dict(self.llm.named_parameters())
        path_marker = f".{self.path_adapter_name}."
        target_marker = f".{target_adapter_name}."
        matched = 0
        with torch.no_grad():
            for name, source in list(parameters.items()):
                if path_marker not in name:
                    continue
                target_name = name.replace(path_marker, target_marker)
                target = parameters.get(target_name)
                if target is None or target.shape != source.shape:
                    continue
                if (
                    target.dtype != source.dtype
                    or target.device != source.device
                ):
                    target.data = target.data.to(
                        device=source.device,
                        dtype=source.dtype,
                    )
                matched += 1
        if matched == 0:
            raise RuntimeError(
                f"could not align adapter storage for {target_adapter_name}"
            )
        return matched

    def _copy_path_adapter_to_answer_adapter(self):
        if self.do_trace_rl:
            self._copy_path_adapter(self.answer_adapter_name)

    def _copy_path_adapter_to_cot_encoder_adapter(self):
        self._copy_path_adapter(self.cot_encoder_adapter_name)
        self._cot_encoder_adapter_loaded = True
        self._set_adapter_parameter_trainability()

    def _snapshot_stage1_policy(self):
        self.stage1_policy_reference.load_state_dict(
            self.trajectory_policy.state_dict(),
            strict=True,
        )
        self.stage1_policy_reference.eval()
        for parameter in self.stage1_policy_reference.parameters():
            parameter.requires_grad_(False)

    def load_state_dict(self, state_dict, strict=True, assign=False):
        answer_marker = f".{self.answer_adapter_name}."
        cot_encoder_marker = f".{self.cot_encoder_adapter_name}."
        self._loaded_stage2_state = any(
            answer_marker in name for name in state_dict
        )
        self._cot_encoder_adapter_loaded = any(
            cot_encoder_marker in name for name in state_dict
        )
        return super().load_state_dict(
            state_dict,
            strict=strict,
            assign=assign,
        )

    def on_load_checkpoint(self, checkpoint):
        checkpoint_stage = int(
            checkpoint.get("trace_policy_training_stage", 1)
        )
        checkpoint_version = checkpoint.get("trace_policy_version")
        if (
            checkpoint.get("trace_policy_training_stage") is not None
            and checkpoint_version != self.trace_policy_version
        ):
            raise RuntimeError(
                "Refusing to mix a legacy TRACE checkpoint with "
                f"{self.trace_policy_version}: found {checkpoint_version!r}"
            )
        reference = extract_stage2_policy_reference(checkpoint)
        self._reference_restored = False
        if reference is not None:
            self.stage1_policy_reference.load_state_dict(
                reference,
                strict=True,
            )
            self._reference_restored = True
        self._loaded_stage2_state = checkpoint_stage == 2
        return super().on_load_checkpoint(checkpoint)

    def on_save_checkpoint(self, checkpoint):
        super().on_save_checkpoint(checkpoint)
        full_state = self.state_dict()
        preserve_prefixes = (
            "trajectory_policy.",
            "trajectory_posterior.",
            "posterior_context_norm.",
            "transition_action_decoder.",
            "step_compressor.",
            "latent_relation.",
            "state_norm.",
        )
        path_marker = f".{self.path_adapter_name}."
        cot_encoder_marker = f".{self.cot_encoder_adapter_name}."
        for name, value in full_state.items():
            if (
                name.startswith(preserve_prefixes)
                or path_marker in name
                or cot_encoder_marker in name
            ):
                checkpoint["state_dict"][name] = value
        checkpoint_stage = 2 if self.do_trace_rl else 1
        checkpoint["trace_policy_training_stage"] = checkpoint_stage
        checkpoint["trace_policy_version"] = self.trace_policy_version
        if checkpoint_stage == 2:
            checkpoint["trace_stage1_policy_reference"] = {
                name: value.detach().cpu()
                for name, value in (
                    self.stage1_policy_reference.state_dict().items()
                )
            }
        else:
            # A Stage-1 checkpoint must never publish the constructor-time
            # reference as if it were the trained Stage-1 policy.
            checkpoint.pop("trace_stage1_policy_reference", None)

    def on_fit_start(self):
        if not self._cot_encoder_adapter_loaded:
            if self.do_trace_rl:
                raise RuntimeError(
                    "Stage 2 requires a Stage-1 checkpoint containing the "
                    "frozen single-CoT encoder adapter"
                )
            self._copy_path_adapter_to_cot_encoder_adapter()
        if self.do_trace_rl:
            if self._loaded_stage2_state and not self._reference_restored:
                raise RuntimeError(
                    "A Stage-2 state was loaded without its immutable Stage-1 "
                    "policy reference. Resume from the full Lightning "
                    "checkpoint instead of loading weights only."
                )
            if not self._loaded_stage2_state:
                self._copy_path_adapter_to_answer_adapter()
            if not self._reference_restored:
                self._snapshot_stage1_policy()
            self._stage2_initialized = True
            self._activate_answer_adapter()
            self._validate_trace_rl_epoch_budget()
        return super().on_fit_start()

    def _validate_trace_rl_epoch_budget(self):
        """Validate a four-rank sampled budget over the intact source split."""
        target_questions = int(
            self.trace_rl_config.get(
                "n_train_samples_per_epoch",
                2048,
            )
        )
        if target_questions <= 0:
            raise RuntimeError(
                "Stage-2 n_train_samples_per_epoch must be positive"
            )
        limit_batches = self.trainer.limit_train_batches
        if isinstance(limit_batches, bool) or not isinstance(
            limit_batches,
            int,
        ):
            raise RuntimeError(
                "Stage 2 requires an integer trainer.limit_train_batches"
            )
        local_batch_size = int(self.all_config.dataloader.batch_size)
        world_size = int(self.trainer.world_size)
        synchronized_batch_size = local_batch_size * world_size
        expected_batches = math.ceil(
            target_questions / float(synchronized_batch_size)
        )
        if int(limit_batches) != expected_batches:
            raise RuntimeError(
                "Stage-2 sampled-budget batch mismatch: "
                f"limit_train_batches={limit_batches}, expected "
                f"ceil({target_questions}/{synchronized_batch_size})="
                f"{expected_batches}"
            )
        realized = int(limit_batches) * local_batch_size * world_size
        padding = realized - target_questions
        if padding < 0 or padding >= synchronized_batch_size:
            raise RuntimeError(
                "Stage-2 DDP padding mismatch: "
                f"{limit_batches} batches x {local_batch_size} local batch "
                f"x {world_size} ranks = {realized}, target unique questions "
                f"={target_questions}"
            )
        if int(self.trainer.accumulate_grad_batches) != 1:
            raise RuntimeError(
                "Stage-2 manual policy updates require "
                "accumulate_grad_batches=1"
            )
        all_indices = list(
            self.trainer.datamodule.get_all_train_indices()
        )
        train_set = self.trainer.datamodule.train_set
        if target_questions > len(all_indices):
            raise RuntimeError(
                "Stage-2 sampled budget exceeds the registered source split: "
                f"budget={target_questions}, dataset={len(all_indices)}"
            )
        if len(train_set) != len(all_indices):
            raise RuntimeError(
                "Stage 2 must sample from the full training split; "
                "do not mutate dataset indices to impose the epoch budget"
            )

    def training_step(
        self,
        batch,
        batch_idx=None,
        dataloader_idx=0,
    ):
        if self.do_trace_rl:
            if not self._stage2_initialized:
                raise RuntimeError("Stage 2 policy reference was not initialized")
            return self.trace_rl_training_step(
                batch=batch,
                batch_idx=batch_idx,
                dataloader_idx=dataloader_idx,
            )
        return super().training_step(
            batch=batch,
            batch_idx=batch_idx,
            dataloader_idx=dataloader_idx,
        )

    def on_train_batch_end(
        self,
        outputs,
        batch,
        batch_idx: int,
    ) -> None:
        """Return released offload/cache blocks to the host and CUDA driver."""
        interval = int(
            self.trace_config.get(
                "stage1_offload_cache_release_interval",
                0,
            )
        )
        should_release = (
            not self.do_trace_rl
            and bool(
                self.trace_config.get(
                    "stage1_posterior_activation_offload",
                    False,
                )
            )
            and interval > 0
            and (batch_idx + 1) % interval == 0
        )
        if should_release:
            gc.collect()
            try:
                ctypes.CDLL(None).malloc_trim(0)
            except AttributeError:
                pass
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return super().on_train_batch_end(outputs, batch, batch_idx)

    def _decode_single_gold_cots(self, batch) -> List[dict]:
        gold_steps = self._decode_step_lists(batch)
        gold_dependencies = self._decode_cached_matrices(
            batch,
            "dependency_matrix",
        )
        gold_confidences = self._decode_cached_matrices(
            batch,
            "confidence_matrix",
        )
        decoded = []
        for sample_index, steps in enumerate(gold_steps):
            decoded.append(
                {
                    "steps": list(steps),
                    "dependency_matrix": gold_dependencies[sample_index],
                    "confidence_matrix": gold_confidences[sample_index],
                    "source": "original_gsm8k_gold_cot",
                    "annotation_status": (
                        "dataset_gold_not_independently_runtime_verified"
                    ),
                }
            )
        return decoded

    def _distance_kwargs(self) -> dict:
        return {
            "anchor_count": int(
                self.trace_config.get("path_anchor_count", 3)
            ),
            "position_weight": float(
                self.trace_config.get("path_position_weight", 0.45)
            ),
            "direction_weight": float(
                self.trace_config.get("path_direction_weight", 0.35)
            ),
            "step_weight": float(
                self.trace_config.get("path_step_weight", 0.15)
            ),
        }

    def _stage1_posterior_activation_context(self):
        """Offload saved posterior activations without changing gradients."""
        enabled = bool(
            self.trace_config.get(
                "stage1_posterior_activation_offload",
                False,
            )
        )
        if (
            not enabled
            or self.do_trace_rl
            or not self.training
            or not torch.is_grad_enabled()
            or not torch.cuda.is_available()
        ):
            return nullcontext()
        return selective_saved_activation_offload(
            minimum_bytes=int(
                self.trace_config.get(
                    "stage1_posterior_activation_offload_min_bytes",
                    1_048_576,
                )
            ),
            pin_memory=bool(
                self.trace_config.get(
                    "stage1_posterior_activation_offload_pin_memory",
                    False,
                )
            )
        )

    def _stage1_trajectory_latents(
        self,
        questions: Sequence[str],
        *,
        deterministic: bool,
        posterior_context: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Run Stage-1 trajectory formation with optional GPU recomputation."""
        checkpoint_trajectory = bool(
            self.trace_config.get(
                "stage1_trajectory_activation_checkpoint",
                False,
            )
        )
        if (
            not checkpoint_trajectory
            or self.do_trace_rl
            or not self.training
            or not torch.is_grad_enabled()
        ):
            return self._trajectory_latents(
                questions,
                deterministic=deterministic,
                posterior_context=posterior_context,
            )

        if posterior_context is None:
            def form_without_context():
                return self._trajectory_latents(
                    questions,
                    deterministic=deterministic,
                )

            return activation_checkpoint(
                form_without_context,
                use_reentrant=False,
            )

        def form_with_context(context):
            return self._trajectory_latents(
                questions,
                deterministic=deterministic,
                posterior_context=context,
            )

        return activation_checkpoint(
            form_with_context,
            posterior_context,
            use_reentrant=False,
        )

    def _trajectory_latents(
        self,
        questions: Sequence[str],
        *,
        deterministic: bool = False,
        posterior_context: Optional[torch.Tensor] = None,
        innovations: Optional[torch.Tensor] = None,
        forced_actions: Optional[torch.Tensor] = None,
        forced_action_mask: Optional[torch.Tensor] = None,
        compute_reference: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Generate one causal latent trajectory per question."""
        self._activate_path_adapter()
        base_model = self.llm.get_base_model()
        backbone = getattr(base_model, "model", None)
        if backbone is None:
            raise RuntimeError(
                "TRACE requires a causal LM exposing its hidden-state backbone "
                "as get_base_model().model"
            )
        batch_size = len(questions)
        action_dim = self.trajectory_policy.action_dim
        expected = (batch_size, self.n_trace_steps, action_dim)
        if posterior_context is not None and tuple(
            posterior_context.shape
        ) != (batch_size, self.hidden_size):
            raise ValueError(
                "posterior_context must have shape "
                f"{(batch_size, self.hidden_size)}"
            )
        if innovations is not None and tuple(innovations.shape) != expected:
            raise ValueError(
                f"innovations have shape {tuple(innovations.shape)}, "
                f"expected {expected}"
            )
        if forced_actions is not None and tuple(forced_actions.shape) != expected:
            raise ValueError(
                f"forced_actions have shape {tuple(forced_actions.shape)}, "
                f"expected {expected}"
            )
        if forced_actions is not None and forced_action_mask is None:
            forced_action_mask = torch.ones(
                batch_size,
                self.n_trace_steps,
                device=self.device,
                dtype=torch.bool,
            )
        if forced_action_mask is not None and tuple(
            forced_action_mask.shape
        ) != (batch_size, self.n_trace_steps):
            raise ValueError("forced_action_mask has an invalid shape")

        question_ids, question_mask = self.prepare_inputs(
            questions,
            padding_side="left",
            part="question",
            suffix=(
                self.speed_template.format(1)
                + self.thinking_separator
            ),
        )
        question_embeds = self.embedding(question_ids)
        question_outputs = backbone(
            inputs_embeds=question_embeds,
            attention_mask=question_mask,
            position_ids=self._trace_position_ids(
                question_mask,
                question_embeds.shape[1],
            ),
            output_hidden_states=False,
            use_cache=True,
            return_dict=True,
        )
        cache = question_outputs.past_key_values
        context_mask = question_mask
        previous_state = self.state_norm(
            question_outputs.last_hidden_state[:, -1, :]
        )
        latent_states = []
        policy_states = []
        residuals = []
        actions = []
        sampled_innovations = []
        log_probs = []
        means = []
        log_stds = []
        reference_means = []
        reference_log_stds = []
        prior_means = []
        prior_log_stds = []
        action_scale = float(
            self.trace_config.get("action_embedding_scale", 1.0)
        )
        step_scale = float(
            self.trace_config.get("dynamics_step_scale", 0.10)
        )

        for step_index in range(self.n_trace_steps):
            policy_states.append(previous_state)
            prior_mean, prior_log_std = (
                self.trajectory_policy.distribution_parameters(
                previous_state,
                step_index,
            )
            )
            if posterior_context is None:
                mean, log_std = prior_mean, prior_log_std
            else:
                mean, log_std = (
                    self.trajectory_posterior.distribution_parameters(
                        previous_state,
                        posterior_context,
                        step_index,
                        prior_mean,
                        prior_log_std,
                    )
                )
            std = torch.exp(log_std)
            if innovations is None:
                epsilon = (
                    torch.zeros_like(mean)
                    if deterministic
                    else torch.randn_like(mean)
                )
            else:
                epsilon = innovations[:, step_index].to(mean.dtype)
            sampled_action = mean + std * epsilon
            if forced_actions is not None:
                mask = forced_action_mask[:, step_index].unsqueeze(-1)
                action = torch.where(
                    mask,
                    forced_actions[:, step_index].to(mean.dtype).detach(),
                    sampled_action,
                )
                realized_epsilon = torch.where(
                    mask,
                    (action - mean) / std.clamp_min(1e-8),
                    epsilon,
                )
            else:
                action = sampled_action
                realized_epsilon = epsilon
            current_input = self.trajectory_policy.latent_input(
                previous_state,
                action,
                step_index,
                action_scale=action_scale,
                step_scale=step_scale,
            ).to(question_embeds.dtype)
            current_mask = torch.ones(
                batch_size,
                1,
                device=self.device,
                dtype=context_mask.dtype,
            )
            context_mask = torch.cat([context_mask, current_mask], dim=1)
            outputs = backbone(
                inputs_embeds=current_input.unsqueeze(1),
                attention_mask=context_mask,
                position_ids=self._trace_position_ids(
                    context_mask,
                    1,
                ),
                past_key_values=cache,
                output_hidden_states=False,
                use_cache=True,
                return_dict=True,
            )
            cache = outputs.past_key_values
            current_state = self.state_norm(
                outputs.last_hidden_state[:, -1, :]
            )
            latent_states.append(current_state)
            residuals.append(current_state - previous_state)
            actions.append(action)
            sampled_innovations.append(realized_epsilon)
            log_probs.append(gaussian_log_prob(action, mean, log_std))
            means.append(mean)
            log_stds.append(log_std)
            prior_means.append(prior_mean)
            prior_log_stds.append(prior_log_std)
            if compute_reference:
                with torch.no_grad():
                    ref_mean, ref_log_std = (
                        self.stage1_policy_reference
                        .distribution_parameters(
                            previous_state.detach(),
                            step_index,
                        )
                    )
                reference_means.append(ref_mean)
                reference_log_stds.append(ref_log_std)
            previous_state = current_state

        latent_mask = torch.ones(
            batch_size,
            self.n_trace_steps,
            device=self.device,
            dtype=question_mask.dtype,
        )
        with torch.no_grad():
            gate_values = self.trajectory_policy.action_gate_values(
                torch.stack(policy_states, dim=1).reshape(
                    -1,
                    self.hidden_size,
                )
            ).view(batch_size, self.n_trace_steps)
        result = {
            "question_attention_mask": question_mask,
            "latent_attention_mask": latent_mask,
            "context_attention_mask": context_mask,
            "past_key_values": cache,
            "latent_states": torch.stack(latent_states, dim=1),
            "policy_states": torch.stack(policy_states, dim=1).detach(),
            "implicit_residuals": torch.stack(residuals, dim=1),
            "actions": torch.stack(actions, dim=1),
            "innovations": torch.stack(sampled_innovations, dim=1),
            "action_log_probs": torch.stack(log_probs, dim=1),
            "action_means": torch.stack(means, dim=1),
            "action_log_stds": torch.stack(log_stds, dim=1),
            "prior_action_means": torch.stack(prior_means, dim=1),
            "prior_action_log_stds": torch.stack(
                prior_log_stds,
                dim=1,
            ),
            "action_gate_values": gate_values,
        }
        if compute_reference:
            result["reference_action_means"] = torch.stack(
                reference_means,
                dim=1,
            )
            result["reference_action_log_stds"] = torch.stack(
                reference_log_stds,
                dim=1,
            )
        return result

    def _teacher_force_bottleneck(
        self,
        trajectory_outputs: Dict[str, torch.Tensor],
        target_texts: Sequence[str],
        *,
        include_hybrid_header: Optional[bool] = None,
        micro_batch_size: Optional[int] = None,
    ) -> torch.Tensor:
        batch_size = len(target_texts)
        if trajectory_outputs["latent_states"].shape[0] != batch_size:
            raise ValueError(
                "trajectory rows and teacher-forcing targets must match"
            )
        if micro_batch_size is None or micro_batch_size >= batch_size:
            loss, _ = self._teacher_force_bottleneck_chunk(
                trajectory_outputs,
                target_texts,
                include_hybrid_header=include_hybrid_header,
            )
            return loss
        if micro_batch_size <= 0:
            raise ValueError("micro_batch_size must be positive")

        weighted_losses = []
        token_counts = []
        for start in range(0, batch_size, micro_batch_size):
            end = min(start + micro_batch_size, batch_size)
            indices = torch.arange(start, end, device=self.device)
            selected_outputs = self._select_trajectory_rows(
                trajectory_outputs,
                indices,
            )
            chunk_loss, chunk_tokens = (
                self._teacher_force_bottleneck_chunk(
                    selected_outputs,
                    target_texts[start:end],
                    include_hybrid_header=include_hybrid_header,
                )
            )
            weighted_losses.append(chunk_loss * chunk_tokens)
            token_counts.append(chunk_tokens)
        return self._combine_token_weighted_losses(
            weighted_losses,
            token_counts,
        )

    @staticmethod
    def _combine_token_weighted_losses(
        weighted_losses: Sequence[torch.Tensor],
        token_counts: Sequence[torch.Tensor],
    ) -> torch.Tensor:
        if not weighted_losses or len(weighted_losses) != len(token_counts):
            raise ValueError("loss and token-count chunks must align")
        return torch.stack(list(weighted_losses)).sum() / torch.stack(
            list(token_counts)
        ).sum().clamp_min(1.0)

    def _teacher_force_bottleneck_chunk(
        self,
        trajectory_outputs: Dict[str, torch.Tensor],
        target_texts: Sequence[str],
        *,
        include_hybrid_header: Optional[bool] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        checkpoint_answer = bool(
            self.trace_config.get(
                "stage1_answer_activation_checkpoint",
                False,
            )
        )
        if (
            checkpoint_answer
            and not self.do_trace_rl
            and self.training
            and torch.is_grad_enabled()
        ):
            def decode(outputs):
                return self._teacher_force_bottleneck_chunk_impl(
                    outputs,
                    target_texts,
                    include_hybrid_header=include_hybrid_header,
                )

            return activation_checkpoint(
                decode,
                trajectory_outputs,
                use_reentrant=False,
            )
        return self._teacher_force_bottleneck_chunk_impl(
            trajectory_outputs,
            target_texts,
            include_hybrid_header=include_hybrid_header,
        )

    def _teacher_force_bottleneck_chunk_impl(
        self,
        trajectory_outputs: Dict[str, torch.Tensor],
        target_texts: Sequence[str],
        *,
        include_hybrid_header: Optional[bool] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        self._activate_answer_adapter()
        target_ids, target_mask = self._prepare_raw_texts(
            target_texts,
            padding_side="right",
            suffix=self.tokenizer.eos_token,
        )
        separator_ids, separator_mask = self._prompt_ids_for_answer(
            len(target_texts),
            include_hybrid_header=include_hybrid_header,
        )
        current_ids = torch.cat([separator_ids, target_ids], dim=1)
        current_mask = torch.cat([separator_mask, target_mask], dim=1)
        loss_mask = torch.cat(
            [torch.zeros_like(separator_mask), target_mask],
            dim=1,
        )
        attention_mask = build_path_bottleneck_mask(
            trajectory_outputs["question_attention_mask"],
            trajectory_outputs["latent_attention_mask"],
            current_mask,
            include_question=self.answer_reads_question,
        )
        source_position_mask = torch.cat(
            [
                trajectory_outputs["context_attention_mask"],
                current_mask,
            ],
            dim=1,
        )
        outputs = self.llm.forward(
            input_ids=current_ids,
            attention_mask=attention_mask,
            position_ids=self._trace_position_ids(
                source_position_mask,
                current_ids.shape[1],
            ),
            past_key_values=self._fork_past_key_values(
                trajectory_outputs["past_key_values"]
            ),
            output_hidden_states=False,
        )
        loss = self._masked_causal_ce(
            outputs.logits,
            current_ids,
            loss_mask,
        )
        token_count = loss_mask[:, 1:].float().sum().clamp_min(1.0)
        return loss, token_count

    def _prompt_ids_for_answer(
        self,
        batch_size: int,
        *,
        include_hybrid_header: Optional[bool] = None,
    ):
        prompt_ids = torch.full(
            (batch_size, 1),
            fill_value=self.thinking_separator_id,
            device=self.device,
            dtype=torch.long,
        )
        prompt_mask = torch.ones_like(prompt_ids)
        if include_hybrid_header is None:
            include_hybrid_header = bool(
                self.readcot_config.get("use_hybrid", False)
                and self.readcot_config.get(
                    "hybrid_seed_anchor_header",
                    False,
                )
            )
        if include_hybrid_header:
            header_ids, header_mask = self._prepare_raw_texts(
                [self.anchor_header + "\n"] * batch_size,
                padding_side="right",
            )
            prompt_ids = torch.cat([prompt_ids, header_ids], dim=1)
            prompt_mask = torch.cat([prompt_mask, header_mask], dim=1)
        return prompt_ids, prompt_mask

    @staticmethod
    def _resolve_latent_read_mask(
        trajectory_outputs: Dict[str, torch.Tensor],
        latent_read_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        available = trajectory_outputs["latent_attention_mask"]
        if latent_read_mask is None:
            return available
        if tuple(latent_read_mask.shape) != tuple(available.shape):
            raise ValueError(
                "latent_read_mask must have shape "
                f"{tuple(available.shape)}, got "
                f"{tuple(latent_read_mask.shape)}"
            )
        return latent_read_mask.to(
            device=available.device,
            dtype=available.dtype,
        ) * available

    @staticmethod
    def _fork_past_key_values(past_key_values):
        """Create a mutable cache shell over the same read-only prefix K/V."""
        if hasattr(past_key_values, "layers"):
            # DynamicCache(ddp_cache_data=...) rebuilds every layer with
            # torch.cat, eagerly duplicating the complete prefix. A shallow
            # cache/layer fork shares the immutable prefix tensors; each
            # layer's update() assigns newly concatenated tensors to the fork
            # and therefore leaves the source cache untouched.
            fork = copy.copy(past_key_values)
            fork.layers = [
                copy.copy(layer) for layer in past_key_values.layers
            ]
            return fork
        if isinstance(past_key_values, tuple):
            return tuple(past_key_values)
        try:
            return type(past_key_values)(list(past_key_values))
        except (TypeError, ValueError):
            return copy.deepcopy(past_key_values)

    def _select_trajectory_rows(
        self,
        trajectory_outputs: Dict[str, torch.Tensor],
        indices: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Select exchangeable paths and their cache rows without recompute."""
        indices = indices.to(device=self.device, dtype=torch.long)
        selected = {}
        batch_size = trajectory_outputs["latent_states"].shape[0]
        for name, value in trajectory_outputs.items():
            if name == "past_key_values":
                cache = self._fork_past_key_values(value)
                if not hasattr(cache, "batch_select_indices"):
                    raise RuntimeError(
                        "trajectory cache cannot select exchangeable rows"
                    )
                cache.batch_select_indices(indices)
                selected[name] = cache
            elif torch.is_tensor(value) and value.shape[0] == batch_size:
                selected[name] = value.index_select(0, indices)
            else:
                selected[name] = value
        return selected

    def _generate_answers_from_trajectory(
        self,
        trajectory_outputs: Dict[str, torch.Tensor],
        *,
        do_sample: bool,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        latent_read_mask: Optional[torch.Tensor] = None,
        include_hybrid_header: Optional[bool] = None,
    ) -> torch.Tensor:
        self._activate_answer_adapter()
        batch_size = trajectory_outputs["latent_states"].shape[0]
        prompt_ids, prompt_mask = self._prompt_ids_for_answer(
            batch_size,
            include_hybrid_header=include_hybrid_header,
        )
        attention_mask = build_path_bottleneck_mask(
            trajectory_outputs["question_attention_mask"],
            self._resolve_latent_read_mask(
                trajectory_outputs,
                latent_read_mask,
            ),
            prompt_mask,
            include_question=self.answer_reads_question,
        )
        source_position_mask = torch.cat(
            [
                trajectory_outputs["context_attention_mask"],
                prompt_mask,
            ],
            dim=1,
        )
        generation_config = dict(self._get_generation_config())
        generation_config["do_sample"] = bool(do_sample)
        if do_sample:
            generation_config["temperature"] = float(
                temperature if temperature is not None else 0.95
            )
            generation_config["top_p"] = float(
                top_p if top_p is not None else 0.97
            )
        generated = self.llm.generate(
            input_ids=prompt_ids,
            attention_mask=attention_mask,
            position_ids=self._trace_position_ids(
                source_position_mask,
                prompt_ids.shape[1],
            ),
            past_key_values=self._fork_past_key_values(
                trajectory_outputs["past_key_values"]
            ),
            **generation_config,
        )
        if (
            generated.shape[1] >= prompt_ids.shape[1]
            and torch.equal(
                generated[:, : prompt_ids.shape[1]],
                prompt_ids,
            )
        ):
            generated = generated[:, prompt_ids.shape[1] :]
        return generated

    def _answer_token_log_probs(
        self,
        trajectory_outputs: Dict[str, torch.Tensor],
        answer_input_ids: torch.Tensor,
        answer_attention_mask: torch.Tensor,
        latent_read_mask: Optional[torch.Tensor] = None,
        include_hybrid_header: Optional[bool] = None,
        decoder_role: str = "deployed",
    ) -> torch.Tensor:
        if decoder_role == "deployed":
            self._activate_answer_adapter()
        elif decoder_role == "stage1_path_value":
            self._activate_path_adapter()
        else:
            raise ValueError(f"unknown decoder_role: {decoder_role}")
        batch_size = answer_input_ids.shape[0]
        prompt_ids, prompt_mask = self._prompt_ids_for_answer(
            batch_size,
            include_hybrid_header=include_hybrid_header,
        )
        current_ids = torch.cat([prompt_ids, answer_input_ids], dim=1)
        current_mask = torch.cat(
            [prompt_mask, answer_attention_mask],
            dim=1,
        )
        attention_mask = build_path_bottleneck_mask(
            trajectory_outputs["question_attention_mask"],
            self._resolve_latent_read_mask(
                trajectory_outputs,
                latent_read_mask,
            ),
            current_mask,
            include_question=self.answer_reads_question,
        )
        source_position_mask = torch.cat(
            [
                trajectory_outputs["context_attention_mask"],
                current_mask,
            ],
            dim=1,
        )
        outputs = self.llm.forward(
            input_ids=current_ids,
            attention_mask=attention_mask,
            position_ids=self._trace_position_ids(
                source_position_mask,
                current_ids.shape[1],
            ),
            past_key_values=self._fork_past_key_values(
                trajectory_outputs["past_key_values"]
            ),
            output_hidden_states=False,
        )
        answer_length = answer_input_ids.shape[1]
        logits = outputs.logits[
            :,
            prompt_ids.shape[1] - 1 : prompt_ids.shape[1] - 1 + answer_length,
            :,
        ]
        flat_logits = logits.reshape(-1, logits.shape[-1])
        flat_targets = answer_input_ids.reshape(-1)
        selected = []
        for start in range(0, flat_logits.shape[0], 16):
            end = min(start + 16, flat_logits.shape[0])
            selected.append(
                F.log_softmax(flat_logits[start:end], dim=-1)
                .gather(
                    dim=-1,
                    index=flat_targets[start:end].unsqueeze(-1),
                )
                .squeeze(-1)
            )
        log_probs = torch.cat(selected, dim=0).reshape_as(
            answer_input_ids
        )
        return torch.nan_to_num(
            log_probs,
            nan=-30.0,
            neginf=-30.0,
            posinf=30.0,
        ).clamp(min=-30.0, max=30.0)

    def _gold_answer_scores(
        self,
        trajectory_outputs: Dict[str, torch.Tensor],
        answers: Sequence[str],
        latent_read_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Score interventions with the frozen Stage-1 path-only channel.

        Stage 1 calibrates this direct-answer protocol on one uniformly drawn
        exchangeable path per question. The answer decoder cannot attend to
        question K/V, so changing the path is the only changed causal source.
        """
        target_ids, target_mask = self._prepare_raw_texts(
            [self.answer_template.format(answer) for answer in answers],
            padding_side="right",
            suffix=self.tokenizer.eos_token,
        )
        log_probs = self._answer_token_log_probs(
            trajectory_outputs,
            target_ids,
            target_mask,
            latent_read_mask=latent_read_mask,
            include_hybrid_header=False,
            decoder_role="stage1_path_value",
        )
        return (
            log_probs * target_mask
        ).sum(dim=-1) / target_mask.sum(dim=-1).clamp_min(1)

    def _target_token_count(self, target: str) -> int:
        return len(
            self.tokenizer.encode(
                target + self.tokenizer.eos_token,
                add_special_tokens=False,
            )
        )

    def _fit_compact_target_to_generation_budget(
        self,
        target: str,
        answer: str,
    ) -> str:
        """Keep whole compact equations within the deployed token budget."""
        budget = int(
            self.trace_config.get(
                "compact_target_max_new_tokens",
                self.model_kwargs.hybrid_generation_config.max_new_tokens,
            )
        )
        if budget <= 0:
            raise ValueError("compact target token budget must be positive")
        if target.startswith(self.anchor_header + "\n"):
            target = target[len(self.anchor_header) + 1 :]
        answer_suffix = (
            self.thinking_separator + self.answer_template.format(answer)
        )
        marker_index = target.rfind(answer_suffix)
        if marker_index < 0:
            raise ValueError("compact target is missing its protected answer")
        if self._target_token_count(answer_suffix) > budget:
            raise ValueError(
                "answer suffix alone exceeds the deployed generation budget"
            )
        if self._target_token_count(target) <= budget:
            return target

        raw_lines = [
            line.strip()
            for line in target[:marker_index].splitlines()
            if line.strip().startswith("- ")
        ]
        candidates = []
        for line in raw_lines:
            text = line[2:].strip()
            clauses = [
                clause.strip()
                for clause in text.split(";")
                if clause.strip()
            ]
            candidates.extend(f"- {clause}" for clause in clauses)

        kept = []
        for line in candidates:
            candidate = "\n".join(kept + [line, answer_suffix])
            if self._target_token_count(candidate) <= budget:
                kept.append(line)
        fitted = "\n".join(kept + [answer_suffix])
        fitted_length = self._target_token_count(fitted)
        if fitted_length > budget:
            raise RuntimeError(
                f"compact target has {fitted_length} tokens, budget={budget}"
            )
        return fitted

    def _cot_posterior_context(
        self,
        explicit_item: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        states = explicit_item["step_states"].float().detach()
        direction = states[-1] - states[0]
        summary = states.mean(dim=0) + 0.5 * direction
        return self.posterior_context_norm(summary)

    def _single_cot_corridor(
        self,
        explicit_item: Dict[str, torch.Tensor],
        student_path: torch.Tensor,
        student_actions: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Align one sampled path to the single observed CoT corridor.

        The CoT is not resampled and does not define a synthetic target set.
        Each sampled action path defines its own strictly ordered compression
        schedule, then selects a nondecreasing alignment to the same observed
        step sequence using detached path semantics.
        """
        states = explicit_item["step_states"].float()
        queries = self.step_compressor.query_proj(
            student_path.detach().float()
        )
        keys = self.step_compressor.key_proj(states)
        values = self.step_compressor.value_proj(states)
        semantic_scores = (
            F.normalize(queries, dim=-1)
            @ F.normalize(keys, dim=-1).transpose(0, 1)
        )
        centers = action_conditioned_progress_centers(
            student_actions.detach().float(),
            progress_dim=int(
                self.trace_config.get("corridor_progress_action_dim", 0)
            ),
            action_scale=float(
                self.trace_config.get("corridor_progress_action_scale", 1.0)
            ),
        ).to(semantic_scores.dtype)
        assignment = stochastic_monotone_assignment(
            semantic_scores,
            centers,
            sigma=float(
                self.trace_config.get("corridor_progress_sigma", 0.20)
            ),
            progress_strength=float(
                self.trace_config.get(
                    "corridor_progress_strength",
                    1.0,
                )
            ),
        )
        compressed = assignment @ values
        relation_logits, relation_probs = self.latent_relation.forward_probs(
            compressed
        )
        dependency_logits, dependency_probs = reconstruct_dependency_logits(
            assignment=assignment,
            relation_probs=relation_probs,
            center_relations=self.readcot_config.get(
                "center_relation_probs",
                True,
            ),
        )
        dependency_loss = dependency_bce_loss(
            dependency_logits,
            explicit_item["dependency_matrix"],
            confidence=explicit_item.get("confidence_matrix"),
            pos_weight_max=self.readcot_config.get(
                "dep_pos_weight_max",
                None,
            ),
            loss_clamp=self.readcot_config.get("dep_loss_clamp", None),
        )
        dependency_f1 = dependency_f1_score(
            dep_probs=dependency_probs,
            gold=explicit_item["dependency_matrix"],
            threshold=self.readcot_config.get(
                "dependency_threshold",
                0.5,
            ),
        )
        return {
            "path": aggregate_step_residuals(
                assignment,
                explicit_item["step_residuals"].float(),
            ),
            "assignment": assignment,
            "progress_centers": centers,
            "dependency_loss": dependency_loss,
            "dependency_f1": dependency_f1,
            "dependency_probs": dependency_probs,
            "relation_probs": relation_probs,
        }

    def _collect_single_cot_features(
        self,
        questions: Sequence[str],
        gold_cots: Sequence[dict],
    ) -> Tuple[List[Dict[str, torch.Tensor]], torch.Tensor]:
        self._activate_cot_encoder_adapter()
        explicit_features = self._collect_explicit_batch_features(
            questions=list(questions),
            step_lists=[cot["steps"] for cot in gold_cots],
            # Do not append a separate answer field to the posterior encoder.
            # The original annotated CoT itself may state the final answer.
            answers=[""] * len(gold_cots),
            dependency_matrices=[
                cot.get("dependency_matrix") for cot in gold_cots
            ],
            confidence_matrices=[
                cot.get("confidence_matrix") for cot in gold_cots
            ],
        )
        self._activate_path_adapter()
        contexts = torch.stack(
            [
                self._cot_posterior_context(item)
                for item in explicit_features
            ],
            dim=0,
        )
        return explicit_features, contexts

    def _build_single_cot_corridors(
        self,
        explicit_features: Sequence[Dict[str, torch.Tensor]],
        model_paths: torch.Tensor,
        model_actions: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        if model_paths.ndim != 4:
            raise ValueError(
                "Stage-1 paths must have shape [batch, sample, step, hidden]"
            )
        if len(explicit_features) != model_paths.shape[0]:
            raise ValueError("explicit CoTs and sampled path groups must align")
        if model_actions.ndim != 4:
            raise ValueError(
                "Stage-1 actions must have shape [batch, sample, step, action]"
            )
        if model_actions.shape[:3] != model_paths.shape[:3]:
            raise ValueError("sampled paths and actions must align")
        paths = []
        assignments = []
        centers = []
        relation_probs = []
        dependency_probs = []
        dependency_losses = []
        dependency_f1s = []
        assignment_diversities = []
        for sample_index, explicit_item in enumerate(explicit_features):
            sample_paths = []
            sample_assignments = []
            sample_centers = []
            sample_relations = []
            sample_dependency_probs = []
            for student_path, student_actions in zip(
                model_paths[sample_index],
                model_actions[sample_index],
            ):
                corridor = self._single_cot_corridor(
                    explicit_item,
                    student_path,
                    student_actions,
                )
                sample_paths.append(corridor["path"])
                sample_assignments.append(corridor["assignment"])
                sample_centers.append(corridor["progress_centers"])
                sample_relations.append(corridor["relation_probs"])
                sample_dependency_probs.append(
                    corridor["dependency_probs"]
                )
                dependency_losses.append(corridor["dependency_loss"])
                dependency_f1s.append(corridor["dependency_f1"])
            paths.append(torch.stack(sample_paths, dim=0))
            assignments.append(sample_assignments)
            centers.append(torch.stack(sample_centers, dim=0))
            relation_probs.append(sample_relations)
            dependency_probs.append(sample_dependency_probs)
            assignment_diversities.append(
                torch.stack(sample_assignments, dim=0)
                .float()
                .std(dim=0, unbiased=False)
                .mean()
            )

        stacked_centers = torch.stack(centers, dim=0)
        return {
            "paths": torch.stack(paths, dim=0),
            "assignments": assignments,
            "progress_centers": stacked_centers,
            "relation_probs": relation_probs,
            "dependency_probs": dependency_probs,
            "dependency_loss": torch.stack(dependency_losses).mean(),
            "dependency_f1": torch.stack(dependency_f1s).mean(),
            "assignment_diversity": torch.stack(
                assignment_diversities
            ).mean(),
            "progress_schedule_diversity": stacked_centers.float()
            .std(dim=1, unbiased=False)
            .mean(),
        }

    def _single_cot_computation_equations(
        self,
        gold_cot: dict,
        answer: str,
    ) -> List[str]:
        equations = extract_unit_normalized_equations(gold_cot["steps"])
        return answer_causal_equation_slice(equations, answer)

    def _complete_computation_trace_target(
        self,
        gold_cot: dict,
        answer: str,
    ) -> str:
        equations = self._single_cot_computation_equations(
            gold_cot,
            answer,
        )
        if equations:
            operations = [
                equation.rsplit("=", 1)[0] for equation in equations
            ]
            trace = ";".join(operations) + "\n"
        else:
            trace = ""
        target = (
            trace
            + self.thinking_separator
            + self.answer_template.format(answer)
        )
        budget = int(
            self.trace_config.get(
                "compact_target_max_new_tokens",
                self.model_kwargs.hybrid_generation_config.max_new_tokens,
            )
        )
        target_length = self._target_token_count(target)
        if target_length > budget:
            raise RuntimeError(
                "complete computation trace exceeds its deployment budget: "
                f"tokens={target_length}, budget={budget}"
            )
        return target

    def _complete_trace_targets_for_paths(
        self,
        *,
        gold_cots: Sequence[dict],
        answers: Sequence[str],
        path_count: int,
        trajectory_outputs: Dict[str, torch.Tensor],
    ) -> Tuple[List[str], torch.Tensor]:
        """Give every hidden path the same complete, compact task target.

        The sampled paths differ only through latent actions and structural
        alignment. Visible target selection therefore cannot create a hidden
        route identity or omit a required arithmetic transition.
        """
        batch_size = len(answers)
        if path_count <= 0:
            raise ValueError("path_count must be positive")
        if trajectory_outputs["implicit_residuals"].shape[0] != (
            batch_size * path_count
        ):
            raise ValueError(
                "trajectory rows do not match complete-trace targets"
            )
        base_targets = [
            self._complete_computation_trace_target(cot, answer)
            for cot, answer in zip(gold_cots, answers)
        ]
        compact_targets = [
            target
            for target in base_targets
            for _ in range(path_count)
        ]
        lengths = torch.tensor(
            [self._target_token_count(target) for target in compact_targets],
            device=self.device,
            dtype=torch.float32,
        )
        return compact_targets, lengths

    def forward(self, batch):
        """Form exchangeable paths while training the deployed policy path."""
        if self.do_trace_rl:
            raise RuntimeError(
                "Stage-1 forward is not used as replay during Stage 2"
            )
        questions = list(batch["question"])
        answers = list(batch["answer"])
        gold_cots = self._decode_single_gold_cots(batch)
        explicit_features, cot_contexts = self._collect_single_cot_features(
            questions,
            gold_cots,
        )
        sample_count = int(
            self.trace_config.get("stage1_posterior_samples", 4)
        )
        if sample_count < 2:
            raise ValueError(
                "Stage 1 requires at least two IID posterior paths"
            )
        repeated_questions = [
            question
            for question in questions
            for _ in range(sample_count)
        ]
        repeated_answers = [
            answer for answer in answers for _ in range(sample_count)
        ]
        repeated_contexts = cot_contexts.repeat_interleave(
            sample_count,
            dim=0,
        )
        with self._stage1_posterior_activation_context():
            model_outputs = self._stage1_trajectory_latents(
                repeated_questions,
                deterministic=False,
                posterior_context=repeated_contexts,
            )
            # Validation and deployment use exactly this question-only,
            # conditional-mean rollout. It is not a member of the exchangeable
            # sampled set and does not define a reasoning-mode identity.
            deployment_outputs = self._stage1_trajectory_latents(
                questions,
                deterministic=True,
            )
        model_paths = model_outputs["implicit_residuals"].view(
            len(questions),
            sample_count,
            self.n_trace_steps,
            self.hidden_size,
        )
        action_groups = model_outputs["actions"].view(
            len(questions),
            sample_count,
            self.n_trace_steps,
            self.trajectory_policy.action_dim,
        )
        corridors = self._build_single_cot_corridors(
            explicit_features,
            model_paths,
            action_groups,
        )
        deployment_paths = deployment_outputs[
            "implicit_residuals"
        ].unsqueeze(1)
        deployment_actions = deployment_outputs["actions"].unsqueeze(1)
        deployment_corridors = self._build_single_cot_corridors(
            explicit_features,
            deployment_paths,
            deployment_actions,
        )
        corridor_paths = corridors["paths"]
        sampled_distance = trajectory_distance_components(
            model_paths,
            corridor_paths.detach(),
            **self._distance_kwargs(),
        )
        deployment_distance = trajectory_distance_components(
            deployment_paths,
            deployment_corridors["paths"].detach(),
            **self._distance_kwargs(),
        )
        sampled_noncollapse = path_noncollapse_loss(
            model_paths,
            margin=float(
                self.trace_config.get("path_noncollapse_margin", 0.02)
            ),
        )
        deployment_noncollapse = path_noncollapse_loss(
            deployment_paths,
            margin=float(
                self.trace_config.get("path_noncollapse_margin", 0.02)
            ),
        )
        predicted_actions = self.transition_action_decoder(
            model_paths.float()
        )
        action_identifiability = action_transition_identifiability_loss(
            predicted_actions,
            action_groups,
            temperature=float(
                self.trace_config.get(
                    "stage1_action_contrastive_temperature",
                    0.10,
                )
            ),
        )

        sampled_compact_targets, sampled_compact_lengths = (
            self._complete_trace_targets_for_paths(
                gold_cots=gold_cots,
                answers=answers,
                path_count=sample_count,
                trajectory_outputs=model_outputs,
            )
        )
        deployment_compact_targets, deployment_compact_lengths = (
            self._complete_trace_targets_for_paths(
                gold_cots=gold_cots,
                answers=answers,
                path_count=1,
                trajectory_outputs=deployment_outputs,
            )
        )
        deployment_risk_mix = float(
            self.trace_config.get("stage1_deployment_risk_mix", 0.50)
        )
        if not 0.0 < deployment_risk_mix < 1.0:
            raise ValueError(
                "stage1_deployment_risk_mix must lie strictly between 0 and 1"
            )
        with self._stage1_posterior_activation_context():
            sampled_compact_loss = self._teacher_force_bottleneck(
                model_outputs,
                sampled_compact_targets,
                include_hybrid_header=True,
                micro_batch_size=int(
                    self.trace_config.get(
                        "stage1_sampled_answer_micro_batch_size",
                        1,
                    )
                ),
            )
            deployment_compact_loss = self._teacher_force_bottleneck(
                deployment_outputs,
                deployment_compact_targets,
                include_hybrid_header=True,
            )
            # One uniformly sampled member per question also calibrates the
            # frozen direct-answer value channel used in Stage 2. Selection is
            # redrawn every batch, so no persistent route receives a special
            # role and the four-path objective remains exchangeable in law.
            direct_local_indices = torch.randint(
                sample_count,
                (len(questions),),
                device=self.device,
            )
            direct_indices = (
                torch.arange(len(questions), device=self.device)
                * sample_count
                + direct_local_indices
            )
            direct_outputs = self._select_trajectory_rows(
                model_outputs,
                direct_indices,
            )
            direct_answer_loss = self._teacher_force_bottleneck(
                direct_outputs,
                [
                    self.answer_template.format(answer)
                    for answer in answers
                ],
                include_hybrid_header=False,
            )
        compact_weight = self._scheduled_loss_weight(
            "anchor",
            float(
                self.trace_config.get("stage1_compact_weight", 1.0)
            ),
        )
        direct_answer_weight = float(
            self.trace_config.get("stage1_direct_answer_weight", 0.35)
        )
        compact_loss = (
            (1.0 - deployment_risk_mix) * sampled_compact_loss
            + deployment_risk_mix * deployment_compact_loss
        )
        answer_loss = (
            compact_weight * compact_loss
            + direct_answer_weight * direct_answer_loss
        )
        noncollapse_weight = float(
            self.trace_config.get("stage1_noncollapse_weight", 0.05)
        )
        action_identifiability_weight = float(
            self.trace_config.get(
                "stage1_action_identifiability_weight",
                0.10,
            )
        )
        distance = {
            name: (
                (1.0 - deployment_risk_mix) * sampled_distance[name].mean()
                + deployment_risk_mix
                * deployment_distance[name].mean()
            )
            for name in ("total", "position", "direction", "step")
        }
        noncollapse = (
            (1.0 - deployment_risk_mix) * sampled_noncollapse
            + deployment_risk_mix * deployment_noncollapse
        )
        formation = (
            distance["total"]
            + noncollapse_weight * noncollapse
            + action_identifiability_weight * action_identifiability
        )
        formation_weight = float(
            self.trace_config.get("stage1_formation_weight", 0.14)
        )
        dependency_weight = self._scheduled_loss_weight(
            "dep",
            self.readcot_config.get("lambda_dep", 0.04),
        )
        posterior_kl = diagonal_gaussian_kl(
            model_outputs["action_means"],
            model_outputs["action_log_stds"],
            model_outputs["prior_action_means"],
            model_outputs["prior_action_log_stds"],
        ).mean()
        posterior_kl_weight = float(
            self.trace_config.get("stage1_posterior_kl_weight", 0.05)
        )
        minimum_action_std = float(
            self.trace_config.get("stage1_minimum_action_std", 0.20)
        )
        action_entropy_floor = 0.5 * (
            minimum_action_entropy_loss(
                model_outputs["action_log_stds"],
                minimum_std=minimum_action_std,
            )
            + minimum_action_entropy_loss(
                model_outputs["prior_action_log_stds"],
                minimum_std=minimum_action_std,
            )
        )
        entropy_weight = float(
            self.trace_config.get(
                "stage1_entropy_floor_weight",
                0.02,
            )
        )
        dependency_loss = (
            (1.0 - deployment_risk_mix) * corridors["dependency_loss"]
            + deployment_risk_mix
            * deployment_corridors["dependency_loss"]
        )
        dependency_f1 = (
            (1.0 - deployment_risk_mix) * corridors["dependency_f1"]
            + deployment_risk_mix * deployment_corridors["dependency_f1"]
        )
        total_loss = (
            answer_loss
            + formation_weight * formation
            + posterior_kl_weight * posterior_kl
            + entropy_weight * action_entropy_floor
            + dependency_weight * dependency_loss
        )
        posterior_std = torch.exp(
            model_outputs["action_log_stds"].float()
        )
        prior_std = torch.exp(
            model_outputs["prior_action_log_stds"].float()
        )
        return {
            "total_loss": total_loss,
            "answer_loss": answer_loss,
            "trace_stage1_direct_answer_loss": direct_answer_loss,
            "trace_stage1_exchangeable_compact_loss": sampled_compact_loss,
            "trace_stage1_sampled_answer_micro_batch_size": (
                total_loss.new_tensor(
                    float(
                        self.trace_config.get(
                            "stage1_sampled_answer_micro_batch_size",
                            1,
                        )
                    )
                )
            ),
            "trace_stage1_deployment_compact_loss": deployment_compact_loss,
            "trace_stage1_mixture_compact_loss": compact_loss,
            "trace_stage1_compact_target_tokens": (
                sampled_compact_lengths.mean().detach()
            ),
            "trace_stage1_compact_target_tokens_max": (
                torch.maximum(
                    sampled_compact_lengths.max(),
                    deployment_compact_lengths.max(),
                ).detach()
            ),
            "trace_stage1_deployment_compact_target_tokens": (
                deployment_compact_lengths.mean().detach()
            ),
            "dep_loss": dependency_loss,
            "dep_f1": dependency_f1,
            "trace_stage1_formation_loss": formation,
            "trace_stage1_corridor_loss": distance["total"],
            "trace_stage1_sampled_corridor_loss": (
                sampled_distance["total"].mean()
            ),
            "trace_stage1_deployment_corridor_loss": (
                deployment_distance["total"].mean()
            ),
            "trace_stage1_position_loss": distance["position"],
            "trace_stage1_direction_loss": distance["direction"],
            "trace_stage1_step_loss": distance["step"],
            "trace_stage1_noncollapse_loss": noncollapse,
            "trace_stage1_action_identifiability_loss": (
                action_identifiability
            ),
            "trace_stage1_action_retrieval_accuracy": (
                action_transition_retrieval_accuracy(
                    predicted_actions.detach(),
                    action_groups.detach(),
                )
            ),
            "trace_stage1_action_entropy_floor_loss": (
                action_entropy_floor
            ),
            "trace_stage1_posterior_prior_kl": posterior_kl,
            "trace_stage1_action_path_correlation": (
                pairwise_action_path_correlation(
                    action_groups.detach(),
                    model_paths.detach(),
                )
            ),
            "trace_stage1_posterior_std": posterior_std.mean().detach(),
            "trace_stage1_posterior_std_min": posterior_std.min().detach(),
            "trace_stage1_prior_std": prior_std.mean().detach(),
            "trace_stage1_single_cot_contract": total_loss.new_ones(()),
            "trace_stage1_iid_posterior_samples": total_loss.new_tensor(
                float(sample_count)
            ),
            "trace_stage1_privileged_sampled_paths": total_loss.new_zeros(()),
            "trace_stage1_deployment_calibration_paths": (
                total_loss.new_ones(())
            ),
            "trace_stage1_deployment_is_sampled_mode": (
                total_loss.new_zeros(())
            ),
            "trace_stage1_structure_supervised_paths": (
                total_loss.new_tensor(float(sample_count + 1))
            ),
            "trace_stage1_exchangeable_task_supervised_paths": (
                total_loss.new_tensor(float(sample_count))
            ),
            "trace_stage1_corridor_progress_span": (
                corridors["progress_centers"][..., -1]
                - corridors["progress_centers"][..., 0]
            ).mean().detach(),
            "trace_stage1_alignment_diversity": corridors[
                "assignment_diversity"
            ].detach(),
            "trace_stage1_progress_schedule_diversity": corridors[
                "progress_schedule_diversity"
            ].detach(),
            "trace_answer_question_access": total_loss.new_tensor(
                float(self.answer_reads_question)
            ),
            "trace_stage1_action_gate": model_outputs[
                "action_gate_values"
            ].mean().detach(),
            "trace_stage1_deployment_action_gate": deployment_outputs[
                "action_gate_values"
            ].mean().detach(),
            "lambda_trace_formation_eff": total_loss.new_tensor(
                formation_weight
            ),
            "lambda_dep_eff": total_loss.new_tensor(dependency_weight),
            "lambda_posterior_kl_eff": total_loss.new_tensor(
                posterior_kl_weight
            ),
            "lambda_entropy_floor_eff": total_loss.new_tensor(
                entropy_weight
            ),
            "lambda_direct_answer_eff": total_loss.new_tensor(
                direct_answer_weight
            ),
            "lambda_compact_eff": total_loss.new_tensor(
                compact_weight
            ),
            "lambda_deployment_risk_mix": total_loss.new_tensor(
                deployment_risk_mix
            ),
            "minimum_action_std": total_loss.new_tensor(
                minimum_action_std
            ),
        }

    def _rollout_micro_batch_size(self) -> int:
        return max(
            1,
            int(self.trace_rl_config.get("rollout_micro_batch_size", 1)),
        )

    def _optimization_micro_batch_size(self) -> int:
        return max(
            1,
            int(self.trace_rl_config.get("exp_batch_size", 1)),
        )

    def _current_rollout_group_size(self) -> int:
        """Use a synchronized 4-to-8 rollout curriculum across DDP ranks."""
        minimum = int(
            self.trace_rl_config.get("minimum_group_size", 4)
        )
        maximum = int(
            self.trace_rl_config.get(
                "maximum_group_size",
                self.trace_rl_config.get("group_size", 8),
            )
        )
        warmup_epochs = int(
            self.trace_rl_config.get("small_group_warmup_epochs", 1)
        )
        if minimum < 2 or maximum < minimum:
            raise ValueError("invalid rollout group-size curriculum")
        return minimum if int(self.current_epoch) < warmup_epochs else maximum

    def _answers_to_accuracy(
        self,
        output_ids: torch.Tensor,
        answers: Sequence[str],
    ) -> torch.Tensor:
        output_strings = self.tokenizer.batch_decode(
            output_ids,
            skip_special_tokens=True,
        )
        accuracy = []
        for output, answer in zip(output_strings, answers):
            prediction = self.extract_answer_from_output(output)
            accuracy.append(
                self.verify_answer(
                    gt_answer=answer,
                    pred_answer=prediction,
                )
            )
        return torch.tensor(
            accuracy,
            device=self.device,
            dtype=torch.float32,
        )

    @torch.no_grad()
    def _score_fixed_action_paths(
        self,
        questions: Sequence[str],
        answers: Sequence[str],
        actions: torch.Tensor,
    ) -> torch.Tensor:
        scores = []
        micro_batch = self._rollout_micro_batch_size()
        full_mask = torch.ones(
            actions.shape[:2],
            device=actions.device,
            dtype=torch.bool,
        )
        for start in range(0, len(questions), micro_batch):
            end = min(start + micro_batch, len(questions))
            trajectory = self._trajectory_latents(
                questions[start:end],
                forced_actions=actions[start:end],
                forced_action_mask=full_mask[start:end],
            )
            scores.append(
                self._gold_answer_scores(
                    trajectory,
                    answers[start:end],
                )
            )
            del trajectory
        return torch.cat(scores, dim=0)

    @staticmethod
    def _hard_pair_path_indices(
        pairs: Sequence[HardPathPair],
    ) -> List[int]:
        return sorted(
            {
                path_index
                for pair in pairs
                for path_index in (
                    pair.correct_index,
                    pair.wrong_index,
                )
            }
        )

    @torch.no_grad()
    def _score_hard_pair_base_paths(
        self,
        group_questions: Sequence[str],
        group_answers: Sequence[str],
        actions: torch.Tensor,
        pairs: Sequence[HardPathPair],
    ) -> torch.Tensor:
        """Score only original paths used by the selected interventions."""
        selected = self._hard_pair_path_indices(pairs)
        if not selected:
            return actions.new_zeros(actions.shape[0])
        selected_tensor = torch.tensor(
            selected,
            device=actions.device,
            dtype=torch.long,
        )
        selected_scores = self._score_fixed_action_paths(
            [group_questions[index] for index in selected],
            [group_answers[index] for index in selected],
            actions.index_select(0, selected_tensor),
        )
        base_scores = selected_scores.new_zeros(actions.shape[0])
        base_scores.index_copy_(0, selected_tensor, selected_scores)
        return base_scores

    @torch.no_grad()
    def _score_counterfactual_paths(
        self,
        group_questions: Sequence[str],
        group_answers: Sequence[str],
        pairs: Sequence[HardPathPair],
        metadata: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        if metadata["forced_actions"].shape[0] == 0:
            return torch.empty(0, device=self.device)
        counterfactual_questions = []
        counterfactual_answers = []
        for row in range(metadata["forced_actions"].shape[0]):
            pair = pairs[int(metadata["pair_indices"][row].item())]
            direction = int(metadata["directions"][row].item())
            recipient = (
                pair.correct_index if direction == 0 else pair.wrong_index
            )
            counterfactual_questions.append(group_questions[recipient])
            counterfactual_answers.append(group_answers[recipient])

        scores = []
        micro_batch = self._rollout_micro_batch_size()
        for start in range(
            0,
            len(counterfactual_questions),
            micro_batch,
        ):
            end = min(
                start + micro_batch,
                len(counterfactual_questions),
            )
            trajectory = self._trajectory_latents(
                counterfactual_questions[start:end],
                innovations=metadata["innovations"][start:end],
                forced_actions=metadata["forced_actions"][start:end],
                forced_action_mask=metadata["forced_mask"][start:end],
            )
            scores.append(
                self._gold_answer_scores(
                    trajectory,
                    counterfactual_answers[start:end],
                )
            )
            del trajectory
        return torch.cat(scores, dim=0)

    @torch.no_grad()
    def trace_policy_rollout(
        self,
        questions: Sequence[str],
        answers: Sequence[str],
    ) -> Dict[str, torch.Tensor]:
        """Collect IID policy paths and separate path and token outcomes."""
        group_size = self._current_rollout_group_size()
        if group_size < 2:
            raise ValueError(
                "group_size must allow a mixed correct/wrong rollout group"
            )
        group_questions = [
            question for question in questions for _ in range(group_size)
        ]
        group_answers = [
            answer for answer in answers for _ in range(group_size)
        ]
        micro_batch = self._rollout_micro_batch_size()
        action_chunks = []
        innovation_chunks = []
        action_log_prob_chunks = []
        policy_state_chunks = []
        path_chunks = []
        greedy_accuracy_chunks = []
        greedy_length_chunks = []
        sampled_accuracy_chunks = []
        sampled_id_chunks = []
        sampled_mask_chunks = []
        old_answer_log_prob_chunks = []
        stage1_answer_log_prob_chunks = []
        frozen_gold_score_chunks = []

        for start in range(0, len(group_questions), micro_batch):
            end = min(start + micro_batch, len(group_questions))
            chunk_questions = group_questions[start:end]
            chunk_answers = group_answers[start:end]

            sampled_path = self._trajectory_latents(
                chunk_questions,
                deterministic=False,
            )
            actions = sampled_path["actions"].detach()
            innovations = sampled_path["innovations"].detach()
            greedy_ids = self._generate_answers_from_trajectory(
                sampled_path,
                do_sample=False,
            )
            greedy_accuracy = self._answers_to_accuracy(
                greedy_ids,
                chunk_answers,
            )
            greedy_lengths = greedy_ids.ne(
                self.tokenizer.pad_token_id
            ).float().sum(dim=-1)
            action_chunks.append(actions)
            innovation_chunks.append(innovations)
            action_log_prob_chunks.append(
                sampled_path["action_log_probs"].detach()
            )
            policy_state_chunks.append(
                sampled_path["policy_states"].float().detach()
            )
            path_chunks.append(
                sampled_path["implicit_residuals"].float().detach()
            )
            greedy_accuracy_chunks.append(greedy_accuracy)
            greedy_length_chunks.append(greedy_lengths)
            sampled_ids = self._generate_answers_from_trajectory(
                sampled_path,
                do_sample=True,
                temperature=float(
                    self.trace_rl_config.get("answer_temperature", 0.95)
                ),
                top_p=float(
                    self.trace_rl_config.get("answer_top_p", 0.97)
                ),
            )
            sampled_mask = sampled_ids.ne(
                self.tokenizer.pad_token_id
            ).long()
            sampled_accuracy = self._answers_to_accuracy(
                sampled_ids,
                chunk_answers,
            )
            sampled_accuracy_chunks.append(sampled_accuracy)
            sampled_id_chunks.append(sampled_ids)
            sampled_mask_chunks.append(sampled_mask)
            old_answer_log_prob_chunks.append(
                self._answer_token_log_probs(
                    sampled_path,
                    sampled_ids,
                    sampled_mask,
                ).detach()
            )
            stage1_answer_log_prob_chunks.append(
                self._answer_token_log_probs(
                    sampled_path,
                    sampled_ids,
                    sampled_mask,
                    decoder_role="stage1_path_value",
                ).detach()
            )
            frozen_gold_score_chunks.append(
                self._gold_answer_scores(
                    sampled_path,
                    chunk_answers,
                ).detach()
            )
            del sampled_path, greedy_ids

        actions = torch.cat(action_chunks, dim=0)
        innovations = torch.cat(innovation_chunks, dim=0)
        old_action_log_probs = torch.cat(
            action_log_prob_chunks,
            dim=0,
        )
        policy_states = torch.cat(policy_state_chunks, dim=0)
        rollout_paths = torch.cat(path_chunks, dim=0)
        greedy_accuracy = torch.cat(greedy_accuracy_chunks, dim=0)
        greedy_lengths = torch.cat(greedy_length_chunks, dim=0)
        sampled_accuracy = torch.cat(sampled_accuracy_chunks, dim=0)
        sampled_ids = _right_pad(
            sampled_id_chunks,
            value=float(self.tokenizer.pad_token_id),
        )
        sampled_mask = _right_pad(sampled_mask_chunks, value=0.0)
        old_answer_log_probs = _right_pad(
            old_answer_log_prob_chunks,
            value=0.0,
        )
        stage1_answer_log_probs = _right_pad(
            stage1_answer_log_prob_chunks,
            value=0.0,
        )
        frozen_gold_scores = torch.cat(
            frozen_gold_score_chunks,
            dim=0,
        )
        output_lengths = sampled_mask.float().sum(dim=-1)
        length_weight = float(
            self.trace_rl_config.get(
                "output_length_penalty_weight",
                0.03,
            )
        )
        target_length = float(
            self.trace_rl_config.get("target_output_length", 33.5)
        )
        trajectory_length_penalty = length_weight * F.relu(
            greedy_lengths / max(target_length, 1.0) - 1.0
        )
        answer_length_penalty = length_weight * F.relu(
            output_lengths / max(target_length, 1.0) - 1.0
        )
        (
            dense_outcome_advantages,
            frozen_score_stds,
            dense_active_groups,
        ) = group_standardize_with_floor(
            frozen_gold_scores.unsqueeze(-1),
            group_size=group_size,
            minimum_std=float(
                self.trace_rl_config.get(
                    "minimum_gold_score_std",
                    1.0e-3,
                )
            ),
        )
        dense_outcome_advantages = dense_outcome_advantages.squeeze(-1)
        trajectory_rewards = (
            greedy_accuracy
            - trajectory_length_penalty
            + float(
                self.trace_rl_config.get("dense_outcome_weight", 0.25)
            )
            * dense_outcome_advantages
        )
        answer_rewards = sampled_accuracy - answer_length_penalty

        pairs = mine_question_local_hard_pairs(
            rollout_paths,
            greedy_accuracy,
            group_size=group_size,
            margin=float(
                self.trace_rl_config.get("local_ranking_margin", 0.08)
            ),
            max_pairs_per_group=int(
                self.trace_rl_config.get(
                    "max_hard_pairs_per_group",
                    1,
                )
            ),
            distance_kwargs=self._distance_kwargs(),
            outcome_scores=frozen_gold_scores,
            minimum_score_gap=float(
                self.trace_rl_config.get(
                    "minimum_gold_score_gap",
                    2.0e-3,
                )
            ),
        )
        counterfactual_credits = rollout_paths.new_zeros(
            (0, self.n_trace_steps)
        )
        if pairs:
            steps_per_pair = min(
                self.n_trace_steps,
                max(
                    1,
                    int(
                        self.trace_rl_config.get(
                            "counterfactual_steps_per_pair",
                            self.n_trace_steps,
                        )
                    ),
                ),
            )
            step_offset = int(self.global_step) % self.n_trace_steps
            selected_steps = sorted(
                {
                    (step_offset + index * self.n_trace_steps // steps_per_pair)
                    % self.n_trace_steps
                    for index in range(steps_per_pair)
                }
            )
            counterfactual_metadata = counterfactual_action_batch(
                actions,
                innovations,
                pairs,
                step_indices=selected_steps,
            )
            counterfactual_scores = self._score_counterfactual_paths(
                group_questions,
                group_answers,
                pairs,
                counterfactual_metadata,
            )
            counterfactual_credits = counterfactual_transition_credits(
                frozen_gold_scores,
                counterfactual_scores,
                counterfactual_metadata,
                pairs,
                n_steps=self.n_trace_steps,
            )

        trajectory_advantages = build_transition_advantages(
            trajectory_rewards,
            n_steps=self.n_trace_steps,
            group_size=group_size,
            pairs=pairs,
            counterfactual_credits=(
                counterfactual_credits if pairs else None
            ),
            counterfactual_weight=float(
                self.trace_rl_config.get(
                    "counterfactual_credit_weight",
                    0.35,
                )
            ),
            local_weight=float(
                self.trace_rl_config.get("local_relation_weight", 0.10)
            ),
            credit_temperature=float(
                self.trace_rl_config.get(
                    "counterfactual_credit_temperature",
                    0.10,
                )
            ),
            local_temperature=float(
                self.trace_rl_config.get(
                    "local_relation_temperature",
                    0.08,
                )
            ),
        )
        answer_advantages = group_standardize(
            answer_rewards.unsqueeze(-1),
            group_size=group_size,
        )

        positive_counts = greedy_accuracy.view(
            -1,
            group_size,
        ).sum(dim=1)
        counterfactual_eligible = (positive_counts >= 1) & (
            positive_counts < group_size
        )
        positive_pair_eligible = (positive_counts >= 2) & (
            positive_counts < group_size
        )
        pair_hinges = (
            torch.tensor(
                [pair.hinge for pair in pairs],
                device=self.device,
            )
            if pairs
            else torch.zeros(1, device=self.device)
        )
        credit_abs = (
            counterfactual_credits.abs().mean()
            if pairs
            else torch.zeros((), device=self.device)
        )
        exact_pairs = sum(
            pair.source == "exact_outcome" for pair in pairs
        )
        continuous_pairs = len(pairs) - exact_pairs
        scored_steps = (
            len(selected_steps) if pairs else 0
        )
        self._last_trace_metrics = {
            "trace_policy/greedy_path_accuracy": greedy_accuracy.mean(),
            "trace_policy/sampled_answer_accuracy": sampled_accuracy.mean(),
            "trace_policy/positive_count": positive_counts.mean(),
            "trace_policy/mixed_group_fraction": (
                (positive_counts > 0) & (positive_counts < group_size)
            ).float().mean(),
            "trace_policy/counterfactual_eligible_fraction": (
                counterfactual_eligible.float().mean()
            ),
            "trace_policy/positive_pair_eligible_fraction": (
                positive_pair_eligible.float().mean()
            ),
            "trace_policy/hard_pair_count": torch.tensor(
                float(len(pairs)),
                device=self.device,
            ),
            "trace_policy/exact_pair_count": torch.tensor(
                float(exact_pairs),
                device=self.device,
            ),
            "trace_policy/continuous_pair_count": torch.tensor(
                float(continuous_pairs),
                device=self.device,
            ),
            "trace_policy/hard_pair_hinge": pair_hinges.mean(),
            "trace_policy/counterfactual_credit_abs": credit_abs,
            "trace_policy/counterfactual_question_coverage": torch.tensor(
                float(len(pairs))
                / float(max(1, len(questions))),
                device=self.device,
            ),
            "trace_policy/counterfactual_path_slot_coverage": torch.tensor(
                float(2 * len(pairs) * scored_steps)
                / float(max(1, len(group_questions) * self.n_trace_steps)),
                device=self.device,
            ),
            "trace_policy/frozen_gold_score": frozen_gold_scores.mean(),
            "trace_policy/frozen_gold_score_group_std": (
                frozen_score_stds.mean()
            ),
            "trace_policy/dense_active_group_fraction": (
                dense_active_groups.float().mean()
            ),
            "trace_policy/dense_outcome_abs": (
                dense_outcome_advantages.abs().mean()
            ),
            "trace_policy/trajectory_advantage_active_fraction": (
                trajectory_advantages.abs() > 1e-6
            ).float().mean(),
        }
        return {
            "group_questions": group_questions,
            "group_answers": group_answers,
            "rollout_group_size": group_size,
            "actions": actions,
            "innovations": innovations,
            "policy_states": policy_states,
            "rollout_paths": rollout_paths,
            "old_action_log_probs": old_action_log_probs,
            "trajectory_rewards": trajectory_rewards,
            "trajectory_advantages": trajectory_advantages,
            "greedy_accuracy": greedy_accuracy,
            "answer_input_ids": sampled_ids,
            "answer_attention_mask": sampled_mask,
            "old_answer_log_probs": old_answer_log_probs,
            "stage1_answer_log_probs": stage1_answer_log_probs,
            "answer_rewards": answer_rewards,
            "answer_advantages": answer_advantages,
            "sampled_accuracy": sampled_accuracy,
            "counterfactual_credits": counterfactual_credits,
        }

    def _trajectory_policy_update(
        self,
        rollout: Dict[str, torch.Tensor],
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        policy_states = rollout["policy_states"]
        actions = rollout["actions"]
        old_log_probs = rollout["old_action_log_probs"]
        advantages = rollout["trajectory_advantages"]
        micro_batch = self._optimization_micro_batch_size()
        total_items = actions.shape[0]
        policy_loss_sum = torch.zeros((), device=self.device)
        prior_kl_sum = torch.zeros((), device=self.device)
        ratio_deviation_sum = torch.zeros((), device=self.device)
        clip_fraction_sum = torch.zeros((), device=self.device)
        prior_weight = float(
            self.trace_rl_config.get("stage1_policy_kl_weight", 0.02)
        )
        prior_target = float(
            self.trace_rl_config.get("stage1_policy_kl_target", 0.02)
        )
        clip_epsilon = float(
            self.trace_rl_config.get("trajectory_clip_epsilon", 0.12)
        )
        for start in range(0, total_items, micro_batch):
            end = min(start + micro_batch, total_items)
            current_means = []
            current_log_stds = []
            reference_means = []
            reference_log_stds = []
            for step_index in range(self.n_trace_steps):
                mean, log_std = (
                    self.trajectory_policy.distribution_parameters(
                        policy_states[start:end, step_index].detach(),
                        step_index,
                    )
                )
                with torch.no_grad():
                    reference_mean, reference_log_std = (
                        self.stage1_policy_reference
                        .distribution_parameters(
                            policy_states[
                                start:end,
                                step_index,
                            ].detach(),
                            step_index,
                        )
                    )
                current_means.append(mean)
                current_log_stds.append(log_std)
                reference_means.append(reference_mean)
                reference_log_stds.append(reference_log_std)
            current_means = torch.stack(current_means, dim=1)
            current_log_stds = torch.stack(current_log_stds, dim=1)
            reference_means = torch.stack(reference_means, dim=1)
            reference_log_stds = torch.stack(reference_log_stds, dim=1)
            current_log_probs = gaussian_log_prob(
                actions[start:end].detach(),
                current_means,
                current_log_stds,
            )
            policy_loss = clipped_policy_loss(
                current_log_probs,
                old_log_probs[start:end].detach(),
                advantages[start:end].detach(),
                clip_epsilon=clip_epsilon,
            )
            ratio = torch.exp(
                current_log_probs
                - old_log_probs[start:end].detach()
            )
            ratio_deviation = (ratio - 1.0).abs().mean()
            clip_fraction = (
                (ratio < 1.0 - clip_epsilon)
                | (ratio > 1.0 + clip_epsilon)
            ).float().mean()
            prior_kl = diagonal_gaussian_kl(
                current_means,
                current_log_stds,
                reference_means,
                reference_log_stds,
            ).mean()
            prior_penalty = F.relu(prior_kl - prior_target)
            chunk_weight = float(end - start) / float(total_items)
            self.manual_backward(
                chunk_weight * (
                    policy_loss + prior_weight * prior_penalty
                )
            )
            policy_loss_sum += (
                policy_loss.detach() * float(end - start)
            )
            prior_kl_sum += prior_kl.detach() * float(end - start)
            ratio_deviation_sum += (
                ratio_deviation.detach() * float(end - start)
            )
            clip_fraction_sum += (
                clip_fraction.detach() * float(end - start)
            )
            del (
                current_means,
                current_log_stds,
                reference_means,
                reference_log_stds,
                current_log_probs,
                policy_loss,
                prior_kl,
                prior_penalty,
                ratio,
                ratio_deviation,
                clip_fraction,
            )
        return (
            policy_loss_sum / float(total_items),
            prior_kl_sum / float(total_items),
            ratio_deviation_sum / float(total_items),
            clip_fraction_sum / float(total_items),
        )

    def _answer_policy_update(
        self,
        rollout: Dict[str, torch.Tensor],
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        questions = rollout["group_questions"]
        actions = rollout["actions"]
        answer_ids = rollout["answer_input_ids"]
        answer_mask = rollout["answer_attention_mask"]
        old_log_probs = rollout["old_answer_log_probs"]
        stage1_log_probs = rollout["stage1_answer_log_probs"]
        advantages = rollout["answer_advantages"]
        micro_batch = self._optimization_micro_batch_size()
        total_items = len(questions)
        policy_loss_sum = torch.zeros((), device=self.device)
        reference_kl_sum = torch.zeros((), device=self.device)
        ratio_deviation_sum = torch.zeros((), device=self.device)
        clip_fraction_sum = torch.zeros((), device=self.device)
        objective_sum = torch.zeros((), device=self.device)
        clip_epsilon = float(
            self.trace_rl_config.get("answer_clip_epsilon", 0.12)
        )
        policy_weight = float(
            self.trace_rl_config.get("answer_policy_weight", 1.0)
        )
        reference_weight = float(
            self.trace_rl_config.get(
                "stage1_answer_kl_weight",
                0.10,
            )
        )
        reference_target = float(
            self.trace_rl_config.get("stage1_answer_kl_target", 0.02)
        )
        full_mask = torch.ones(
            actions.shape[:2],
            device=self.device,
            dtype=torch.bool,
        )
        for start in range(0, total_items, micro_batch):
            end = min(start + micro_batch, total_items)
            with torch.no_grad():
                trajectory = self._trajectory_latents(
                    questions[start:end],
                    forced_actions=actions[start:end],
                    forced_action_mask=full_mask[start:end],
                )
            current_log_probs = self._answer_token_log_probs(
                trajectory,
                answer_ids[start:end],
                answer_mask[start:end],
            )
            token_advantages = advantages[start:end].expand_as(
                current_log_probs
            )
            loss = clipped_policy_loss(
                current_log_probs,
                old_log_probs[start:end].detach(),
                token_advantages.detach(),
                clip_epsilon=clip_epsilon,
                mask=answer_mask[start:end],
            )
            ratio = torch.exp(
                current_log_probs
                - old_log_probs[start:end].detach()
            )
            active = answer_mask[start:end].to(ratio.dtype)
            active_count = active.sum().clamp_min(1.0)
            ratio_deviation = (
                (ratio - 1.0).abs() * active
            ).sum() / active_count
            clip_fraction = (
                (
                    (ratio < 1.0 - clip_epsilon)
                    | (ratio > 1.0 + clip_epsilon)
                ).to(active.dtype)
                * active
            ).sum() / active_count
            reference_kl = sampled_forward_kl(
                current_log_probs,
                stage1_log_probs[start:end].detach(),
                mask=answer_mask[start:end],
            )
            reference_penalty = F.relu(
                reference_kl - reference_target
            )
            objective = (
                policy_weight * loss
                + reference_weight * reference_penalty
            )
            chunk_weight = float(end - start) / float(total_items)
            self.manual_backward(chunk_weight * objective)
            policy_loss_sum += loss.detach() * float(end - start)
            reference_kl_sum += (
                reference_kl.detach() * float(end - start)
            )
            ratio_deviation_sum += (
                ratio_deviation.detach() * float(end - start)
            )
            clip_fraction_sum += (
                clip_fraction.detach() * float(end - start)
            )
            objective_sum += objective.detach() * float(end - start)
            del (
                trajectory,
                current_log_probs,
                loss,
                ratio,
                active,
                ratio_deviation,
                clip_fraction,
                reference_kl,
                reference_penalty,
                objective,
            )
        denominator = float(total_items)
        return (
            policy_loss_sum / denominator,
            reference_kl_sum / denominator,
            ratio_deviation_sum / denominator,
            clip_fraction_sum / denominator,
            objective_sum / denominator,
        )

    def trace_rl_training_step(
        self,
        batch,
        batch_idx=None,
        dataloader_idx=0,
    ):
        optimizer = self.optimizers()
        rollout = self.trace_policy_rollout(
            questions=list(batch["question"]),
            answers=list(batch["answer"]),
        )
        update_epochs = int(
            self.trace_rl_config.get("policy_update_epochs", 1)
        )
        if update_epochs < 1:
            raise RuntimeError("invalid Stage-2 policy update contract")
        trajectory_losses = []
        answer_losses = []
        answer_reference_kls = []
        answer_ratio_deviations = []
        answer_clip_fractions = []
        answer_objectives = []
        prior_kls = []
        ratio_deviations = []
        clip_fractions = []
        grad_norms = []
        trajectory_grad_norms = []
        answer_grad_norms = []
        optimizer_steps = 0
        for _ in range(update_epochs):
            optimizer.zero_grad(set_to_none=True)
            (
                trajectory_loss,
                prior_kl,
                ratio_deviation,
                clip_fraction,
            ) = self._trajectory_policy_update(rollout)
            (
                answer_loss,
                answer_reference_kl,
                answer_ratio_deviation,
                answer_clip_fraction,
                answer_objective,
            ) = self._answer_policy_update(rollout)

            trajectory_parameters = [
                parameter
                for parameter in self.trajectory_policy.parameters()
                if parameter.requires_grad
            ]
            answer_marker = f".{self.answer_adapter_name}."
            answer_parameters = [
                parameter
                for name, parameter in self.llm.named_parameters()
                if parameter.requires_grad and answer_marker in name
            ]
            if not trajectory_parameters or not answer_parameters:
                raise RuntimeError(
                    "Stage 2 requires disjoint trajectory and answer params"
                )
            trajectory_grad_norm = clip_grad_norm_(
                trajectory_parameters,
                max_norm=float(
                    self.trace_rl_config.get(
                        "trajectory_clip_grad_norm",
                        1.0,
                    )
                ),
            )
            answer_grad_norm = clip_grad_norm_(
                answer_parameters,
                max_norm=float(
                    self.trace_rl_config.get(
                        "answer_clip_grad_norm",
                        1.0,
                    )
                ),
            )
            grad_norm = torch.maximum(
                trajectory_grad_norm,
                answer_grad_norm,
            )
            optimizer_did_step = bool(
                torch.isfinite(trajectory_grad_norm)
                and torch.isfinite(answer_grad_norm)
            )
            if optimizer_did_step:
                optimizer.step()
                optimizer_steps += 1
                if bool(
                    self.all_config.model.training_kwargs.get(
                        "use_scheduler",
                        False,
                    )
                ):
                    scheduler = self.lr_schedulers()
                    if isinstance(scheduler, (list, tuple)):
                        for item in scheduler:
                            item.step()
                    elif scheduler is not None:
                        scheduler.step()
            else:
                optimizer.zero_grad(set_to_none=True)
            trajectory_losses.append(trajectory_loss)
            answer_losses.append(answer_loss)
            answer_reference_kls.append(answer_reference_kl)
            answer_ratio_deviations.append(answer_ratio_deviation)
            answer_clip_fractions.append(answer_clip_fraction)
            answer_objectives.append(answer_objective)
            prior_kls.append(prior_kl)
            ratio_deviations.append(ratio_deviation)
            clip_fractions.append(clip_fraction)
            grad_norms.append(grad_norm.detach())
            trajectory_grad_norms.append(
                trajectory_grad_norm.detach()
            )
            answer_grad_norms.append(answer_grad_norm.detach())

        trajectory_loss = torch.stack(trajectory_losses).mean()
        answer_loss = torch.stack(answer_losses).mean()
        answer_reference_kl = torch.stack(
            answer_reference_kls
        ).mean()
        answer_ratio_deviation = torch.stack(
            answer_ratio_deviations
        ).mean()
        answer_ratio_deviation_final = answer_ratio_deviations[-1]
        answer_clip_fraction = torch.stack(
            answer_clip_fractions
        ).mean()
        answer_clip_fraction_final = answer_clip_fractions[-1]
        answer_objective = torch.stack(answer_objectives).mean()
        prior_kl = torch.stack(prior_kls).mean()
        ratio_deviation = torch.stack(ratio_deviations).mean()
        final_ratio_deviation = ratio_deviations[-1]
        clip_fraction = torch.stack(clip_fractions).mean()
        final_clip_fraction = clip_fractions[-1]
        grad_norm = torch.stack(grad_norms).mean()
        trajectory_grad_norm = torch.stack(
            trajectory_grad_norms
        ).mean()
        answer_grad_norm = torch.stack(answer_grad_norms).mean()

        prior_weight = float(
            self.trace_rl_config.get("stage1_policy_kl_weight", 0.02)
        )
        prior_target = float(
            self.trace_rl_config.get("stage1_policy_kl_target", 0.02)
        )
        total_loss = (
            trajectory_loss
            + answer_objective
            + prior_weight * F.relu(prior_kl - prior_target)
        )
        raw_optimizer = getattr(optimizer, "optimizer", optimizer)
        learning_rate = float(raw_optimizer.param_groups[0]["lr"])
        logs = {
            "train/total_loss": total_loss.detach(),
            "train/trajectory_policy_loss": trajectory_loss.detach(),
            "train/answer_policy_loss": answer_loss.detach(),
            "train/answer_objective": answer_objective.detach(),
            "train/stage1_answer_kl": answer_reference_kl.detach(),
            "train/answer_ratio_deviation": (
                answer_ratio_deviation.detach()
            ),
            "train/answer_ratio_deviation_final_update": (
                answer_ratio_deviation_final.detach()
            ),
            "train/answer_clip_fraction": (
                answer_clip_fraction.detach()
            ),
            "train/answer_clip_fraction_final_update": (
                answer_clip_fraction_final.detach()
            ),
            "train/stage1_policy_kl": prior_kl.detach(),
            "train/action_ratio_deviation": ratio_deviation.detach(),
            "train/action_ratio_deviation_final_update": (
                final_ratio_deviation.detach()
            ),
            "train/action_clip_fraction": clip_fraction.detach(),
            "train/action_clip_fraction_final_update": (
                final_clip_fraction.detach()
            ),
            "train/trajectory_reward": rollout[
                "trajectory_rewards"
            ].mean().detach(),
            "train/answer_reward": rollout[
                "answer_rewards"
            ].mean().detach(),
            "train/output_length": rollout[
                "answer_attention_mask"
            ].float().sum(dim=-1).mean().detach(),
            "train/n_latent_forward": torch.tensor(
                float(self.n_trace_steps),
                device=self.device,
            ),
            "train/grad_norm": grad_norm.detach(),
            "train/trajectory_grad_norm": (
                trajectory_grad_norm.detach()
            ),
            "train/answer_grad_norm": answer_grad_norm.detach(),
            "train/effective_lr": torch.tensor(
                learning_rate,
                device=self.device,
            ),
            "train/optimizer_did_step": torch.tensor(
                float(optimizer_steps) / float(update_epochs),
                device=self.device,
            ),
            "train/policy_update_epochs": torch.tensor(
                float(update_epochs),
                device=self.device,
            ),
        }
        logs.update(
            {
                f"train/{name}": value.detach()
                for name, value in self._last_trace_metrics.items()
            }
        )
        self.log_dict(
            logs,
            sync_dist=True,
            prog_bar=True,
            batch_size=len(batch["idx"]),
        )
        return total_loss.detach()

    @torch.no_grad()
    def read_generate_with_trajectory(
        self,
        questions: Sequence[str],
    ):
        trajectory = self._trajectory_latents(
            questions,
            deterministic=True,
        )
        output_ids = self._generate_answers_from_trajectory(
            trajectory,
            do_sample=False,
        )
        n_latent = torch.full(
            (len(questions), 1),
            fill_value=self.n_trace_steps,
            device=self.device,
            dtype=torch.long,
        )
        return output_ids, n_latent, trajectory

    @torch.no_grad()
    def read_generate(self, questions: List[str]):
        output_ids, n_latent, _ = self.read_generate_with_trajectory(
            questions
        )
        return output_ids, n_latent

    @staticmethod
    def _visual_seed(question: str, index: int, base_seed: int) -> int:
        digest = hashlib.sha256(
            f"{base_seed}|{index}|{question}".encode("utf-8")
        ).digest()
        return int.from_bytes(digest[:8], "little") % (2**63 - 1)

    def _pairwise_path_distances(
        self,
        paths: torch.Tensor,
    ) -> torch.Tensor:
        return trajectory_distance(
            paths[:, None],
            paths[None, :],
            **self._distance_kwargs(),
        )

    @torch.no_grad()
    def _build_policy_visual_record(
        self,
        *,
        index: int,
        question: str,
        answer: str,
        gold_cot: dict,
        map_trajectory: Dict[str, torch.Tensor],
        map_local_index: int,
        map_prediction: str,
        map_accuracy: float,
        map_output_length: int,
    ) -> dict:
        group_size = int(
            self.trace_config.get("visual_group_size", 8)
        )
        seed = self._visual_seed(
            question,
            index,
            int(self.trace_config.get("visual_seed", 271828)),
        )
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        innovations = torch.randn(
            group_size,
            self.n_trace_steps,
            self.trajectory_policy.action_dim,
            generator=generator,
            dtype=torch.float32,
        ).to(self.device)
        questions = [question] * group_size
        answers = [answer] * group_size
        micro_batch = self._rollout_micro_batch_size()
        tensor_chunks: Dict[str, List[torch.Tensor]] = {
            "actions": [],
            "action_means": [],
            "action_log_stds": [],
            "latent_states": [],
            "implicit_residuals": [],
        }
        predictions = []
        correctness = []
        output_lengths = []
        for start in range(0, group_size, micro_batch):
            end = min(start + micro_batch, group_size)
            trajectory = self._trajectory_latents(
                questions[start:end],
                innovations=innovations[start:end],
            )
            output_ids = self._generate_answers_from_trajectory(
                trajectory,
                do_sample=False,
            )
            output_strings = self.tokenizer.batch_decode(
                output_ids,
                skip_special_tokens=True,
            )
            for output, target, token_ids in zip(
                output_strings,
                answers[start:end],
                output_ids,
            ):
                prediction = self.extract_answer_from_output(output)
                predictions.append(prediction)
                correctness.append(
                    int(self.verify_answer(target, prediction))
                )
                output_lengths.append(
                    int(
                        token_ids.ne(self.tokenizer.pad_token_id)
                        .sum()
                        .item()
                    )
                )
            for name in tensor_chunks:
                tensor_chunks[name].append(
                    trajectory[name].detach().float()
                )
            del trajectory, output_ids
        tensors = {
            name: torch.cat(chunks, dim=0)
            for name, chunks in tensor_chunks.items()
        }
        correctness_tensor = torch.tensor(
            correctness,
            device=self.device,
            dtype=torch.float32,
        )
        pairs = mine_question_local_hard_pairs(
            tensors["implicit_residuals"],
            correctness_tensor,
            group_size=group_size,
            margin=float(
                self.trace_rl_config.get("local_ranking_margin", 0.08)
            ),
            max_pairs_per_group=int(
                self.trace_rl_config.get(
                    "max_hard_pairs_per_group",
                    1,
                )
            ),
            distance_kwargs=self._distance_kwargs(),
        )

        fork_devices = (
            [torch.cuda.current_device()]
            if self.device.type == "cuda"
            else []
        )
        with torch.random.fork_rng(devices=fork_devices):
            torch.manual_seed(seed + 1)
            explicit_features, _ = self._collect_single_cot_features(
                [question],
                [gold_cot],
            )
            cot_corridors = self._build_single_cot_corridors(
                explicit_features,
                tensors["implicit_residuals"].unsqueeze(0),
                tensors["actions"].unsqueeze(0),
            )
        corridor_paths = cot_corridors["paths"][0]
        student_distances = self._pairwise_path_distances(
            tensors["implicit_residuals"]
        )
        corridor_distances = self._pairwise_path_distances(
            corridor_paths
        )
        return {
            "idx": int(index),
            "question": question,
            "answer": answer,
            "map_prediction": map_prediction,
            "map_correct": int(map_accuracy),
            "map_output_length": int(map_output_length),
            "map_path_type": "conditional_policy_mean",
            "map_actions": map_trajectory["actions"][map_local_index]
            .detach()
            .to(torch.float16)
            .cpu(),
            "map_action_means": map_trajectory["action_means"][
                map_local_index
            ]
            .detach()
            .to(torch.float16)
            .cpu(),
            "map_action_log_stds": map_trajectory["action_log_stds"][
                map_local_index
            ]
            .detach()
            .to(torch.float16)
            .cpu(),
            "map_latent_states": map_trajectory["latent_states"][
                map_local_index
            ]
            .detach()
            .to(torch.float16)
            .cpu(),
            "map_implicit_residuals": map_trajectory[
                "implicit_residuals"
            ][map_local_index]
            .detach()
            .to(torch.float16)
            .cpu(),
            "rollout_schema": "iid_conditional_gaussian",
            "rollout_seed": int(seed),
            "rollout_innovations": innovations.to(torch.float16).cpu(),
            "rollout_actions": tensors["actions"].to(torch.float16).cpu(),
            "rollout_action_means": tensors["action_means"]
            .to(torch.float16)
            .cpu(),
            "rollout_action_log_stds": tensors["action_log_stds"]
            .to(torch.float16)
            .cpu(),
            "rollout_latent_states": tensors["latent_states"]
            .to(torch.float16)
            .cpu(),
            "rollout_implicit_residuals": tensors[
                "implicit_residuals"
            ]
            .to(torch.float16)
            .cpu(),
            "rollout_predictions": predictions,
            "rollout_correctness": correctness,
            "rollout_output_lengths": output_lengths,
            "rollout_path_distance_matrix": student_distances.cpu(),
            "hard_pairs": [
                {
                    "correct_index": pair.correct_index,
                    "correct_peer_index": pair.correct_peer_index,
                    "wrong_index": pair.wrong_index,
                    "correct_radius": pair.correct_radius,
                    "wrong_distance": pair.wrong_distance,
                    "hinge": pair.hinge,
                    "has_correct_peer": pair.has_correct_peer,
                }
                for pair in pairs
            ],
            "corridor_schema": (
                "single_gold_cot_action_conditioned_monotone_corridor"
            ),
            "corridor_paths": corridor_paths.detach()
            .to(torch.float16)
            .cpu(),
            "corridor_assignments": [
                assignment.detach().to(torch.float16).cpu()
                for assignment in cot_corridors["assignments"][0]
            ],
            "corridor_progress_centers": cot_corridors[
                "progress_centers"
            ][0]
            .detach()
            .cpu(),
            "corridor_relation_probs": [
                relation.detach().to(torch.float16).cpu()
                for relation in cot_corridors["relation_probs"][0]
            ],
            "corridor_path_distance_matrix": corridor_distances.cpu(),
            "gold_cot_source": gold_cot["source"],
            "gold_cot_steps": list(gold_cot["steps"]),
            "answer_question_attention_access": int(
                self.answer_reads_question
            ),
            "answer_latent_attention_access": self.n_trace_steps,
            "visualization_contract": {
                "projection": "global_train_fit_pca_only",
                "manual_offsets": False,
                "per_path_rescaling": False,
                "outcome_used_for_projection": False,
            },
        }

    @torch.no_grad()
    def eval_generation(
        self,
        batch,
        split="val",
        batch_idx=None,
        dataloader_idx=0,
    ):
        indices = batch["idx"].tolist()
        questions = list(batch["question"])
        answers = list(batch["answer"])
        steps = batch["steps"]
        output_ids, n_latent, trajectory = (
            self.read_generate_with_trajectory(questions)
        )
        output_strings = self.tokenizer.batch_decode(
            output_ids,
            skip_special_tokens=True,
        )
        gold_cots = self._decode_single_gold_cots(batch)
        accuracies = []
        output_lengths = []
        for local_index, (
            index,
            question,
            reasoning,
            answer,
            token_ids,
            output,
            latent_count,
        ) in enumerate(
            zip(
                indices,
                questions,
                steps,
                answers,
                output_ids,
                output_strings,
                n_latent,
            )
        ):
            prediction = self.extract_answer_from_output(output)
            accuracy = self.verify_answer(answer, prediction)
            output_length = int(
                token_ids.ne(self.tokenizer.pad_token_id).sum().item()
            )
            if index not in self.sample_logs:
                self.sample_logs[index]["question"] = question
                self.sample_logs[index]["steps"] = reasoning
                self.sample_logs[index]["answer"] = answer
                self.sample_logs[index]["pred_answer"] = []
                self.sample_logs[index]["output_string"] = []
                self.sample_logs[index]["output_length"] = []
                self.sample_logs[index]["n_latent_forward"] = []
                self.sample_logs[index]["acc"] = []
            self.sample_logs[index]["pred_answer"].append(prediction)
            self.sample_logs[index]["output_string"].append(output)
            self.sample_logs[index]["output_length"].append(output_length)
            self.sample_logs[index]["n_latent_forward"].append(
                int(latent_count.item())
            )
            self.sample_logs[index]["acc"].append(accuracy)
            accuracies.append(accuracy)
            output_lengths.append(output_length)

            record_limit = int(
                self.trace_config.get("visual_record_limit", 0)
            )
            if (
                record_limit > 0
                and len(self._trace_visual_records) < record_limit
            ):
                self._trace_visual_records.append(
                    self._build_policy_visual_record(
                        index=int(index),
                        question=question,
                        answer=answer,
                        gold_cot=gold_cots[local_index],
                        map_trajectory=trajectory,
                        map_local_index=local_index,
                        map_prediction=prediction,
                        map_accuracy=accuracy,
                        map_output_length=output_length,
                    )
                )

        mean_accuracy = float(np.mean(accuracies))
        return {
            "monitor": mean_accuracy,
            f"{split}/acc": mean_accuracy,
            f"{split}/n_latent_forward": float(self.n_trace_steps),
            f"{split}/n_latent_forward_on_acc": float(
                self.n_trace_steps
            ),
            f"{split}/output_length": float(np.mean(output_lengths)),
        }

    def on_validation_epoch_start(self):
        self._validation_question_records = []
        return super().on_validation_epoch_start()

    def validation_step(
        self,
        batch,
        batch_idx,
        dataloader_idx=0,
    ):
        metrics = self.eval_generation(
            batch=batch,
            split="val",
            batch_idx=batch_idx,
            dataloader_idx=dataloader_idx,
        )
        for index in batch["idx"].tolist():
            self._validation_question_records.append(
                (
                    int(index),
                    float(self.sample_logs[index]["acc"][-1]),
                    int(self.sample_logs[index]["output_length"][-1]),
                )
            )
        return metrics

    def on_test_start(self):
        self._trace_visual_records = []
        return super().on_test_start()

    def _save_trace_visual_records(self, split: str):
        if not self._trace_visual_records:
            return
        trainer = getattr(self, "trainer", None)
        if trainer is not None and not getattr(
            trainer,
            "is_global_zero",
            True,
        ):
            return
        try:
            directory = Path(self.logger.log_dir)
        except Exception:
            directory = Path(".")
        torch.save(
            self._trace_visual_records,
            directory / f"trace_policy_visual_{split}.pt",
        )

    def on_validation_epoch_end(self):
        if self.trainer.sanity_checking:
            return super().on_validation_epoch_end()
        if dist.is_available() and dist.is_initialized():
            shards = [None] * dist.get_world_size()
            dist.all_gather_object(
                shards,
                self._validation_question_records,
            )
        else:
            shards = [self._validation_question_records]
        expected_count = len(self.trainer.datamodule.val_set)
        summary = summarize_unique_validation_records(
            shards,
            expected_count=expected_count,
        )
        self.log_dict(
            {
                "monitor": summary["accuracy"],
                "val/acc": summary["accuracy"],
                "val/output_length": summary["output_length"],
                "val/n_latent_forward": float(self.n_trace_steps),
                "val/unique_questions": summary["unique_questions"],
            },
            sync_dist=False,
            on_step=False,
            on_epoch=True,
            batch_size=expected_count,
        )
        self._save_trace_visual_records("val")
        return super().on_validation_epoch_end()

    def on_test_end(self):
        self._save_trace_visual_records("test")
        return super().on_test_end()
