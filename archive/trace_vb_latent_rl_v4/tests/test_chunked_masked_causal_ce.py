import unittest
from types import SimpleNamespace
from unittest import mock

import torch
import torch.nn.functional as F

from src.models.read import LitREADCoT


def _full_masked_causal_ce(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    loss_mask: torch.Tensor,
) -> torch.Tensor:
    """Pre-fix objective, kept here only as a numerical oracle."""
    if input_ids.shape[1] <= 1:
        return logits.new_zeros(())
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    shift_mask = loss_mask[:, 1:].contiguous().float()
    losses = F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.shape[-1]),
        shift_labels.reshape(-1),
        reduction="none",
    ).view_as(shift_labels)
    return (losses * shift_mask).sum() / shift_mask.sum().clamp_min(1.0)


def _ce_owner(chunk_size: int = 7):
    # Supply the common attribute/config spellings so this contract remains
    # independent of how LitREADCoT exposes the memory-tuning knob.
    return SimpleNamespace(
        masked_causal_ce_chunk_size=chunk_size,
        causal_ce_token_chunk_size=chunk_size,
        ce_token_chunk_size=chunk_size,
        token_chunk_size=chunk_size,
        readcot_config={
            "masked_causal_ce_chunk_size": chunk_size,
            "causal_ce_token_chunk_size": chunk_size,
            "ce_token_chunk_size": chunk_size,
        },
    )


def _actual_masked_causal_ce(logits, input_ids, loss_mask, *, chunk_size=7):
    return LitREADCoT._masked_causal_ce(
        _ce_owner(chunk_size),
        logits,
        input_ids,
        loss_mask,
    )


class ChunkedMaskedCausalCEContractTests(unittest.TestCase):
    def test_chunking_preserves_masked_objective_and_logits_gradient(self):
        generator = torch.Generator().manual_seed(20260808)
        shape = (3, 29, 47)
        actual_logits = torch.randn(
            shape,
            generator=generator,
            dtype=torch.float32,
            requires_grad=True,
        )
        reference_logits = actual_logits.detach().clone().requires_grad_(True)
        input_ids = torch.randint(
            0,
            shape[-1],
            shape[:2],
            generator=generator,
        )
        # Include internal holes and fractional weights. This rules out an
        # implementation that averages per-chunk means or only counts chunks.
        loss_mask = torch.tensor(
            [
                [1, 1, 0, 1, 0.25, 1, 1, 0, 0, 1, 1, 0.5, 1, 0, 1,
                 1, 0, 1, 1, 1, 0, 0.75, 1, 0, 1, 1, 0, 1, 1],
                [0, 0, 1, 1, 1, 0, 0.5, 0, 1, 1, 0, 1, 0, 1, 1,
                 0.25, 1, 0, 1, 0, 1, 1, 1, 0, 0.75, 1, 0, 1, 0],
                [1, 1, 1, 0, 1, 0, 1, 0.5, 0, 1, 0.25, 1, 1, 1, 0,
                 0, 1, 1, 0.75, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1],
            ],
            dtype=torch.float32,
        )

        actual = _actual_masked_causal_ce(
            actual_logits,
            input_ids,
            loss_mask,
            chunk_size=7,
        )
        reference = _full_masked_causal_ce(
            reference_logits,
            input_ids,
            loss_mask,
        )
        actual.backward()
        reference.backward()

        self.assertTrue(
            torch.allclose(actual, reference, atol=2e-6, rtol=2e-6),
            (float(actual), float(reference)),
        )
        self.assertTrue(
            torch.allclose(
                actual_logits.grad,
                reference_logits.grad,
                atol=2e-6,
                rtol=2e-6,
            ),
            float((actual_logits.grad - reference_logits.grad).abs().max()),
        )

    def test_actual_method_splits_a_long_token_axis(self):
        generator = torch.Generator().manual_seed(17)
        logits = torch.randn(2, 2050, 13, generator=generator)
        input_ids = torch.randint(0, 13, (2, 2050), generator=generator)
        loss_mask = torch.ones(2, 2050)
        original_cross_entropy = F.cross_entropy
        call_rows = []

        def recording_cross_entropy(values, labels, *args, **kwargs):
            call_rows.append(int(values.shape[0]))
            return original_cross_entropy(values, labels, *args, **kwargs)

        with mock.patch(
            "src.models.read.F.cross_entropy",
            side_effect=recording_cross_entropy,
        ):
            loss = _actual_masked_causal_ce(
                logits,
                input_ids,
                loss_mask,
                chunk_size=7,
            )

        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(
            len(call_rows),
            1,
            "masked causal CE still materializes one full token-by-vocab loss",
        )
        self.assertLess(
            max(call_rows),
            logits.shape[0] * (logits.shape[1] - 1),
        )

    def test_all_zero_mask_is_backward_safe_and_has_exact_zero_gradient(self):
        generator = torch.Generator().manual_seed(91)
        logits = torch.randn(
            2,
            23,
            31,
            generator=generator,
            requires_grad=True,
        )
        input_ids = torch.randint(0, 31, (2, 23), generator=generator)
        loss_mask = torch.zeros(2, 23)

        loss = _actual_masked_causal_ce(
            logits,
            input_ids,
            loss_mask,
            chunk_size=5,
        )
        self.assertTrue(loss.requires_grad)
        self.assertEqual(float(loss), 0.0)
        loss.backward()
        self.assertIsNotNone(logits.grad)
        self.assertEqual(int(torch.count_nonzero(logits.grad)), 0)

    def test_checkpoint_retains_no_new_vocab_dense_ce_intermediate(self):
        generator = torch.Generator().manual_seed(404)
        batch_size, sequence_length, vocab_size = 2, 33, 37
        logits = torch.randn(
            batch_size,
            sequence_length,
            vocab_size,
            generator=generator,
            requires_grad=True,
        )
        input_ids = torch.randint(
            0,
            vocab_size,
            (batch_size, sequence_length),
            generator=generator,
        )
        loss_mask = torch.ones(batch_size, sequence_length)
        saved_vocab_tensors = []

        def record_saved(tensor):
            if tensor.ndim == 2 and tensor.shape[-1] == vocab_size:
                saved_vocab_tensors.append(tensor)
            return tensor

        with torch.autograd.graph.saved_tensors_hooks(
            record_saved,
            lambda tensor: tensor,
        ):
            loss = _actual_masked_causal_ce(
                logits,
                input_ids,
                loss_mask,
                chunk_size=8,
            )

        self.assertGreater(len(saved_vocab_tensors), 0)
        input_storage = logits.untyped_storage().data_ptr()
        foreign_vocab_storage = [
            tensor
            for tensor in saved_vocab_tensors
            if tensor.untyped_storage().data_ptr() != input_storage
        ]
        self.assertEqual(
            foreign_vocab_storage,
            [],
            "forward retained a log-softmax-sized CE intermediate instead of "
            "checkpointing it for backward recomputation",
        )
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_bfloat16_forward_backward_is_finite_and_tracks_fp32(self):
        generator = torch.Generator().manual_seed(31337)
        fp32_logits = torch.randn(
            2,
            41,
            53,
            generator=generator,
            dtype=torch.float32,
            requires_grad=True,
        )
        bf16_logits = fp32_logits.detach().to(torch.bfloat16).requires_grad_(True)
        input_ids = torch.randint(0, 53, (2, 41), generator=generator)
        loss_mask = torch.randint(
            0,
            2,
            (2, 41),
            generator=generator,
            dtype=torch.long,
        )
        # Ensure both sides contain supervised tokens regardless of RNG details.
        loss_mask[:, -1] = 1

        try:
            bf16_loss = _actual_masked_causal_ce(
                bf16_logits,
                input_ids,
                loss_mask,
                chunk_size=6,
            )
            bf16_loss.backward()
        except RuntimeError as error:
            self.fail(f"bfloat16 chunked CE forward/backward failed: {error}")

        fp32_loss = _full_masked_causal_ce(
            fp32_logits,
            input_ids,
            loss_mask,
        )
        fp32_loss.backward()

        self.assertTrue(torch.isfinite(bf16_loss.float()))
        self.assertTrue(torch.isfinite(bf16_logits.grad.float()).all())
        self.assertTrue(
            torch.allclose(
                bf16_loss.float(),
                fp32_loss.detach(),
                atol=5e-2,
                rtol=2e-2,
            ),
            (float(bf16_loss), float(fp32_loss)),
        )
        self.assertTrue(
            torch.allclose(
                bf16_logits.grad.float(),
                fp32_logits.grad,
                atol=5e-3,
                rtol=5e-2,
            ),
            float(
                (bf16_logits.grad.float() - fp32_logits.grad)
                .abs()
                .max()
            ),
        )


if __name__ == "__main__":
    unittest.main()
