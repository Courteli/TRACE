import argparse
import json
import sys
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.datasets.gsm8k_aug_nl import (  # noqa: E402
    FORMAL_FILES,
    FORMAL_GSM8K_DIR,
    split_cot_steps,
)
from src.models.trace_policy import (  # noqa: E402
    answer_causal_equation_slice,
    extract_unit_normalized_equations,
    is_numerically_valid_equation,
)


def build_target(cot: str, answer: str, separator: str) -> tuple[str, int]:
    equations = extract_unit_normalized_equations(split_cot_steps(cot))
    equations = answer_causal_equation_slice(equations, answer)
    if equations:
        operations = [equation.rsplit("=", 1)[0] for equation in equations]
        trace = ";".join(operations) + "\n"
    else:
        trace = ""
    return trace + separator + f"###Answer:{answer}", len(equations)


def summarize(values: list[int]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "p50": float(np.quantile(array, 0.50)),
        "p90": float(np.quantile(array, 0.90)),
        "p99": float(np.quantile(array, 0.99)),
        "max": int(array.max()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    config = OmegaConf.load(
        ROOT / "src/configs/models/trace_policy_qwen3_instruct.yaml"
    )
    model = config.model.model_kwargs
    trace = model.trace_policy_config
    budget = int(trace.compact_target_max_new_tokens)
    deployed_budget = int(model.hybrid_generation_config.max_new_tokens)
    if budget != deployed_budget:
        raise SystemExit("training and deployment target budgets differ")
    tokenizer = AutoTokenizer.from_pretrained(
        model.llm_path,
        trust_remote_code=bool(model.trust_remote_code),
    )
    separator = "<|latent_end|>"
    report = {
        "status": "PASS",
        "target_mode": str(trace.stage1_target_mode),
        "budget": budget,
        "generated_or_alternative_cots": False,
        "splits": {},
    }
    for file_name, (expected_count, _) in FORMAL_FILES.items():
        token_lengths = []
        equation_counts = []
        no_equation_count = 0
        invalid_equation_count = 0
        path = FORMAL_GSM8K_DIR / file_name
        with path.open(encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
        if len(rows) != expected_count:
            raise SystemExit(f"{path} row count changed")
        for row in rows:
            equations = extract_unit_normalized_equations(
                split_cot_steps(str(row["cot"]))
            )
            invalid = [
                equation
                for equation in equations
                if not is_numerically_valid_equation(equation)
            ]
            if invalid:
                raise SystemExit(
                    f"{file_name} id={row['id']} contains invalid compiled "
                    f"equations: {invalid}"
                )
            invalid_equation_count += len(invalid)
            target, equation_count = build_target(
                str(row["cot"]),
                str(row["answer"]),
                separator,
            )
            token_count = len(
                tokenizer.encode(
                    target + tokenizer.eos_token,
                    add_special_tokens=False,
                )
            )
            if token_count > budget:
                raise SystemExit(
                    f"{file_name} id={row['id']} target has "
                    f"{token_count} tokens, budget={budget}"
                )
            token_lengths.append(token_count)
            equation_counts.append(equation_count)
            no_equation_count += int(equation_count == 0)
        report["splits"][file_name] = {
            "questions": len(rows),
            "target_tokens": summarize(token_lengths),
            "equations": summarize(equation_counts),
            "no_parseable_equation_questions": no_equation_count,
            "numerically_invalid_equations": invalid_equation_count,
        }
    text = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
