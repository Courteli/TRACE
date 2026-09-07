import json
import tempfile
import unittest
from pathlib import Path

from tools.build_trace_vb_sufficiency_cache import (
    PRE_ACTION_ROLE_NAMES,
    PROMPT_SPEC,
    SCHEMA_VERSION,
    audit_stats,
    align_prefix_signal_to_pre_action_roles,
    build_answer_target,
    build_prefix_context,
    build_question_context,
    contains_explicit_answer,
    cumulative_prefixes,
    encode_scoring_context,
    first_answer_leakage_step,
    make_cache_row,
    make_fingerprints,
    normalise_prefix_logps,
    read_cache,
    sha256_json,
    split_cot_steps,
    summarise_rows,
    tokenizer_fingerprint,
    validate_cache_object,
    write_cache,
)


class TraceVBSufficiencyPureFunctionTests(unittest.TestCase):
    def test_prompt_protocol_matches_stage0_boundaries(self):
        question_context = build_question_context("What is 1+1?")
        prefix_context = build_prefix_context(
            "What is 1+1?", "One plus one equals two."
        )
        self.assertEqual(
            question_context,
            "Question: What is 1+1? Let's think step by step:"
            "(Thinking speed: 1)###",
        )
        self.assertEqual(
            prefix_context,
            question_context + "One plus one equals two.###",
        )
        self.assertEqual(build_answer_target("2", "<eos>"), "Answer:2<eos>")
        self.assertEqual(len(sha256_json(PROMPT_SPEC)), 64)

    def test_scoring_context_preserves_stage0_tokenisation_chunks(self):
        class BoundaryTokenizer:
            def encode(self, text, add_special_tokens=False):
                self.assert_no_special_tokens = not add_special_tokens
                return [len(text), sum(ord(character) for character in text)]

        tokenizer = BoundaryTokenizer()
        question = "What is 1+1?"
        encoded = encode_scoring_context(
            tokenizer, question, prefix="One plus one is two."
        )
        question_chunk = (
            "Question: What is 1+1? Let's think step by step:"
            "(Thinking speed: 1)"
        )
        reasoning_chunk = "###One plus one is two.###"
        self.assertEqual(
            encoded,
            [
                len(question_chunk),
                sum(map(ord, question_chunk)),
                len(reasoning_chunk),
                sum(map(ord, reasoning_chunk)),
            ],
        )

    def test_splitter_preserves_decimals_titles_and_units(self):
        self.assertEqual(
            split_cot_steps(
                "Mr. Lee earns $0.2 per min. He works 10 mins. "
                "So 0.2 * 10 = 2."
            ),
            [
                "Mr. Lee earns $0.2 per min.",
                "He works 10 mins.",
                "So 0.2 * 10 = 2.",
            ],
        )

    def test_prefixes_use_the_same_newline_join_as_registered_stage0_data(self):
        self.assertEqual(
            cumulative_prefixes(["First step.", "Second step."]),
            ["First step.", "First step.\nSecond step."],
        )

    def test_numeric_answer_detection_uses_numeric_boundaries(self):
        self.assertTrue(contains_explicit_answer("The result is $1,200.", "1200"))
        self.assertTrue(contains_explicit_answer("Thus x = 10.0.", "10"))
        self.assertFalse(contains_explicit_answer("There are 100 items.", "10"))
        self.assertFalse(contains_explicit_answer("The rate is 0.25.", "25"))

    def test_first_leakage_step_is_zero_based_and_masks_that_prefix(self):
        steps = ["First calculate 4 + 3.", "This equals 7.", "Done."]
        self.assertEqual(first_answer_leakage_step(steps, "7"), 1)
        signal = normalise_prefix_logps(
            -4.0,
            [-3.0, -2.0, -1.0],
            leakage_step=1,
        )
        self.assertEqual(signal.valid_mask, [True, False, False])

    def test_normalisation_uses_question_and_full_endpoints(self):
        signal = normalise_prefix_logps(
            -4.0,
            [-3.0, -2.0, -1.0],
            leakage_step=2,
        )
        self.assertEqual(signal.denominator, 3.0)
        self.assertAlmostEqual(signal.scores_raw[0], 1.0 / 3.0)
        self.assertAlmostEqual(signal.scores_raw[1], 2.0 / 3.0)
        self.assertEqual(signal.scores, [1.0 / 3.0, 2.0 / 3.0, 1.0])
        self.assertEqual(signal.valid_mask, [True, True, False])
        self.assertIsNone(signal.invalid_reason)

    def test_negative_and_large_raw_scores_are_clipped_but_auditable(self):
        signal = normalise_prefix_logps(
            -4.0,
            [-5.0, 1.0, -2.0],
            leakage_step=None,
        )
        self.assertEqual(signal.scores, [0.0, 1.0, 1.0])
        self.assertEqual(signal.scores_raw, [-0.5, 2.5, 1.0])
        self.assertEqual(signal.valid_mask, [True, True, True])

    def test_non_improving_full_prefix_invalidates_entire_row(self):
        signal = normalise_prefix_logps(
            -1.0,
            [-1.5, -1.1],
            leakage_step=None,
        )
        self.assertEqual(signal.valid_mask, [False, False])
        self.assertEqual(signal.scores, [0.0, 0.0])
        self.assertEqual(
            signal.invalid_reason, "full_not_better_than_question"
        )

    def test_anomalous_denominator_invalidates_entire_row(self):
        signal = normalise_prefix_logps(
            -100.0,
            [-80.0, -1.0],
            leakage_step=None,
            max_full_improvement=50.0,
        )
        self.assertEqual(signal.valid_mask, [False, False])
        self.assertEqual(signal.invalid_reason, "anomalous_denominator")

    def test_make_row_preserves_direct_dataset_idx_lookup_contract(self):
        source = {
            "id": 91,
            "question": "What is 1+1?",
            "cot": "Start with one. Add one to obtain 2.",
            "answer": "2",
        }
        row = make_cache_row(
            dataset_idx=4,
            source_row=source,
            logps=[-4.0, -3.0, -1.0],
            min_full_improvement=1e-4,
            max_full_improvement=50.0,
            max_abs_raw_score=8.0,
        )
        self.assertEqual(row["idx"], 4)
        self.assertEqual(row["source_id"], 91)
        self.assertEqual(row["n_steps"], 2)
        self.assertEqual(row["valid_mask"], [True, False])
        self.assertEqual(len(row["role_scores"]), 8)
        self.assertEqual(len(row["role_valid_mask"]), 8)
        cache = {
            "schema_version": SCHEMA_VERSION,
            "metadata": {"stats": summarise_rows([row])},
            "rows": [row],
            "by_idx": {4: row},
        }
        validate_cache_object(cache)
        self.assertIs(cache["by_idx"][4], row)

    def test_prefixes_align_to_eight_pre_action_roles_without_filling_empty_slots(self):
        aligned = align_prefix_signal_to_pre_action_roles(
            [0.25, 0.75], [True, False]
        )
        self.assertEqual(
            PRE_ACTION_ROLE_NAMES,
            (
                "PLAN",
                "SOLVE1",
                "SOLVE2",
                "SOLVE3",
                "SOLVE4",
                "SOLVE5",
                "REFINE",
                "COMMIT",
            ),
        )
        # Two observed steps occupy SOLVE1 and SOLVE5. Consequently the
        # SOLVE2 pre-action state sees prefix 0; middle empty endpoints remain
        # masked, and the leaked final endpoint cannot supervise REFINE/COMMIT.
        self.assertEqual(
            aligned["role_source_prefix_index"],
            [-1, -1, 0, None, None, None, 1, 1],
        )
        self.assertEqual(
            aligned["role_valid_mask"],
            [True, True, True, False, False, False, False, False],
        )

    def test_summary_and_audit_use_pre_leakage_signal_only(self):
        rows = [
            {
                "idx": 0,
                "n_steps": 3,
                "scores": [0.1, 0.5, 1.0],
                "scores_raw": [0.1, 0.5, 1.0],
                "prefix_logps": [-3.0, -2.0, -1.0],
                "valid_mask": [True, True, False],
                "leakage_step": 2,
                "denominator": 3.0,
                "invalid_reason": None,
            },
            {
                "idx": 1,
                "n_steps": 2,
                "scores": [0.0, 0.0],
                "scores_raw": [None, None],
                "prefix_logps": [-2.0, -3.0],
                "valid_mask": [False, False],
                "leakage_step": None,
                "denominator": -1.0,
                "invalid_reason": "full_not_better_than_question",
            },
        ]
        stats = summarise_rows(rows)
        self.assertEqual(stats["valid_rows"], 1)
        self.assertEqual(stats["valid_prefixes"], 2)
        self.assertEqual(stats["leakage_rate"], 0.5)
        self.assertAlmostEqual(stats["pre_leakage_score_span_mean"], 0.4)
        self.assertEqual(stats["pre_leakage_nonzero_gain_fraction"], 1.0)
        self.assertEqual(
            audit_stats(
                stats,
                min_valid_row_fraction=0.50,
                min_valid_prefix_fraction=0.20,
                min_nonzero_gain_fraction=0.30,
                min_score_span_mean=0.05,
            ),
            [],
        )
        failures = audit_stats(
            stats,
            min_valid_row_fraction=0.75,
            min_valid_prefix_fraction=0.20,
            min_nonzero_gain_fraction=0.30,
            min_score_span_mean=0.05,
        )
        self.assertEqual(len(failures), 1)
        self.assertIn("valid_row_fraction", failures[0])


class TraceVBSufficiencyFingerprintTests(unittest.TestCase):
    def test_tokenizer_fingerprint_only_tracks_tokenizer_assets(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "tokenizer.json").write_text(
                json.dumps({"version": 1}), encoding="utf-8"
            )
            (root / "tokenizer_config.json").write_text(
                json.dumps({"eos_token": "<eos>"}), encoding="utf-8"
            )
            (root / "model.safetensors").write_bytes(b"ignored model")
            first = tokenizer_fingerprint(root)
            (root / "model.safetensors").write_bytes(b"changed model")
            second = tokenizer_fingerprint(root)
            self.assertEqual(first["sha256"], second["sha256"])
            (root / "tokenizer.json").write_text(
                json.dumps({"version": 2}), encoding="utf-8"
            )
            third = tokenizer_fingerprint(root)
            self.assertNotEqual(first["sha256"], third["sha256"])

    def test_composite_fingerprint_changes_with_each_required_component(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "data.jsonl"
            checkpoint = root / "teacher.ckpt"
            model = root / "model"
            model.mkdir()
            data.write_text('{"x":1}\n', encoding="utf-8")
            checkpoint.write_bytes(b"checkpoint-v1")
            (model / "tokenizer.json").write_text("v1", encoding="utf-8")
            first = make_fingerprints(
                data_path=data,
                teacher_checkpoint=checkpoint,
                base_model=model,
            )
            self.assertEqual(len(first["data_sha256"]), 64)
            self.assertEqual(len(first["teacher_checkpoint_sha256"]), 64)
            self.assertEqual(len(first["tokenizer_sha256"]), 64)
            self.assertEqual(len(first["prompt_sha256"]), 64)
            self.assertEqual(len(first["composite_sha256"]), 64)

            checkpoint.write_bytes(b"checkpoint-v2")
            second = make_fingerprints(
                data_path=data,
                teacher_checkpoint=checkpoint,
                base_model=model,
            )
            self.assertNotEqual(
                first["composite_sha256"], second["composite_sha256"]
            )

    def test_pt_cache_round_trip_keeps_integer_index(self):
        row = {
            "idx": 3,
            "n_steps": 1,
            "prefix_logps": [-1.0],
            "scores_raw": [1.0],
            "scores": [1.0],
            "valid_mask": [False],
            "leakage_step": 0,
            "role_scores": [0.0] * 8,
            "role_valid_mask": [False] * 8,
            "role_source_prefix_index": [-1, -1, None, None, None, None, 0, 0],
        }
        cache = {
            "schema_version": SCHEMA_VERSION,
            "metadata": {"stats": summarise_rows([row])},
            "rows": [row],
            "by_idx": {3: row},
        }
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "cache.pt"
            write_cache(cache, output)
            loaded = read_cache(output)
        self.assertIn(3, loaded["by_idx"])
        self.assertEqual(loaded["by_idx"][3]["scores"], [1.0])


if __name__ == "__main__":
    unittest.main()
