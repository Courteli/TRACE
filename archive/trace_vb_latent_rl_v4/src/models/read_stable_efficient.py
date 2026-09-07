import torch
import torch.nn.functional as F

from .read_efficient import LitREADCoTEfficient


class LitREADCoTStableEfficient(LitREADCoTEfficient):
    """Stable efficient READ-CoT training variant.

    This version keeps the CSA/HCA-inspired packed/block forward path from
    ``LitREADCoTEfficient`` and only changes the training objective schedule.
    TensorBoard showed a recurring pattern: validation accuracy peaks around
    6k-13k steps, while training losses keep shrinking and auxiliary losses keep
    updating the latent modules. To reduce this late-stage drift, the dependency,
    residual, and anchor losses can warm up early and then decay after the
    empirically useful window.
    """

    def _scheduled_loss_weight(self, name: str, base_weight: float):
        if base_weight == 0:
            return 0.0

        cfg = self.readcot_config
        step = float(self.global_step)
        warmup_steps = float(cfg.get(f"{name}_warmup_steps", cfg.get("aux_warmup_steps", 0)) or 0)
        decay_start = cfg.get(f"{name}_decay_start_step", cfg.get("aux_decay_start_step", None))
        decay_end = cfg.get(f"{name}_decay_end_step", cfg.get("aux_decay_end_step", None))
        min_scale = float(cfg.get(f"{name}_min_scale", cfg.get("aux_min_scale", 1.0)))

        scale = 1.0
        if warmup_steps > 0:
            scale *= min(1.0, max(0.0, step / warmup_steps))

        if decay_start is not None and step >= float(decay_start):
            if decay_end is not None and float(decay_end) > float(decay_start):
                progress = (step - float(decay_start)) / (float(decay_end) - float(decay_start))
                progress = min(1.0, max(0.0, progress))
                scale *= (1.0 - progress) + progress * min_scale
            else:
                scale *= min_scale

        return float(base_weight) * scale

    def forward(self, batch):
        questions = batch["question"]
        answers = batch["answer"]
        step_lists = self._decode_step_lists(batch)
        dependency_matrices = self._decode_cached_matrices(batch, "dependency_matrix")
        confidence_matrices = self._decode_cached_matrices(batch, "confidence_matrix")
        anchor_indices = self._decode_anchor_indices(batch)
        anchor_bonus_indices = self._decode_anchor_bonus_indices(batch)

        explicit_features = self._collect_explicit_batch_features(
            questions=questions,
            step_lists=step_lists,
            answers=answers,
            dependency_matrices=dependency_matrices,
            confidence_matrices=confidence_matrices,
        )
        compression_outputs = self._compress_explicit_reasoning(explicit_features)
        latent_outputs = self._question_only_latents(questions)
        residual_targets = compression_outputs["aggregated_explicit_residuals"]
        residual_corruption = self.readcot_config.get("residual_target_corruption", "none")
        if residual_corruption and residual_corruption != "none":
            if residual_corruption == "shuffle":
                perm = torch.randperm(residual_targets.shape[1], device=residual_targets.device)
                residual_targets = residual_targets[:, perm, :]
            elif residual_corruption == "reverse":
                residual_targets = torch.flip(residual_targets, dims=[1])
            elif residual_corruption == "random":
                residual_targets = torch.randn_like(residual_targets)
            elif residual_corruption == "cross_sample":
                residual_targets = torch.roll(residual_targets, shifts=1, dims=0)
            else:
                raise ValueError(f"Unsupported residual_target_corruption: {residual_corruption}")
        residual_outputs = self._compute_residual_alignment(
            implicit_residuals=latent_outputs["implicit_residuals"],
            aggregated_explicit_residuals=residual_targets,
        )

        answer_weight = self._scheduled_loss_weight(
            "answer",
            self.readcot_config.get("answer_loss_weight", 1.0),
        )
        if answer_weight > 0:
            answer_targets = [self.answer_template.format(answer) for answer in answers]
            answer_loss = self._teacher_force_target(
                past_key_values=latent_outputs["past_key_values"],
                context_attention_mask=latent_outputs["context_attention_mask"],
                target_texts=answer_targets,
            )
        else:
            answer_loss = latent_outputs["implicit_residuals"].new_zeros(())

        anchor_loss = answer_loss.new_zeros(())
        anchor_gate_loss = answer_loss.new_zeros(())
        use_anchor_supervision = self.readcot_config.get("use_hybrid", False) or self.readcot_config.get(
            "use_anchor_loss", False
        )
        if use_anchor_supervision:
            build_outputs = self._build_hybrid_targets(
                step_lists=step_lists,
                answers=answers,
                explicit_features=explicit_features,
                compression_outputs=compression_outputs,
                implicit_residuals=latent_outputs["implicit_residuals"],
                anchor_indices=anchor_indices,
                anchor_bonus_indices=anchor_bonus_indices,
                return_anchor_gate_targets=self.use_anchor_gate,
            )
            if self.use_anchor_gate:
                anchor_targets, _, anchor_gate_targets = build_outputs
            else:
                anchor_targets, _ = build_outputs
            prefix_texts = None
            if self.readcot_config.get("hybrid_seed_anchor_header", False):
                prefix = self.anchor_header + "\n"
                prefix_texts = [prefix] * len(anchor_targets)
                anchor_targets = [
                    target[len(prefix) :] if target.startswith(prefix) else target
                    for target in anchor_targets
                ]
            anchor_loss = self._teacher_force_target(
                past_key_values=latent_outputs["past_key_values"],
                context_attention_mask=latent_outputs["context_attention_mask"],
                target_texts=anchor_targets,
                prefix_texts=prefix_texts,
            )
            if self.use_anchor_gate and latent_outputs.get("anchor_gate") is not None:
                gate_target = torch.stack(anchor_gate_targets, dim=0).detach()
                anchor_gate_loss = F.smooth_l1_loss(latent_outputs["anchor_gate"], gate_target)

        dep_weight = self._scheduled_loss_weight("dep", self.readcot_config.lambda_dep)
        res_weight = self._scheduled_loss_weight("res", self.readcot_config.lambda_res)
        anchor_weight = self._scheduled_loss_weight("anchor", self.readcot_config.lambda_anchor)
        anchor_gate_weight = self._scheduled_loss_weight(
            "anchor_gate",
            self.readcot_config.get("lambda_anchor_gate", 0.0),
        )

        total_loss = answer_weight * answer_loss
        total_loss = total_loss + dep_weight * compression_outputs["dep_loss"]
        total_loss = total_loss + res_weight * residual_outputs["residual_loss"]
        if use_anchor_supervision:
            total_loss = total_loss + anchor_weight * anchor_loss
        if self.use_anchor_gate:
            total_loss = total_loss + anchor_gate_weight * anchor_gate_loss

        return {
            "total_loss": total_loss,
            "answer_loss": answer_loss,
            "dep_loss": compression_outputs["dep_loss"],
            "dep_f1": compression_outputs["dep_f1"],
            "residual_loss": residual_outputs["residual_loss"],
            "residual_mse_loss": residual_outputs["residual_mse_loss"],
            "residual_cosine_distance": residual_outputs["residual_cosine_distance"],
            "residual_similarity": residual_outputs["residual_similarity"],
            "anchor_loss": anchor_loss,
            "anchor_gate_loss": anchor_gate_loss,
            "lambda_answer_eff": answer_loss.new_tensor(answer_weight),
            "lambda_dep_eff": answer_loss.new_tensor(dep_weight),
            "lambda_res_eff": answer_loss.new_tensor(res_weight),
            "lambda_anchor_eff": answer_loss.new_tensor(anchor_weight),
            "lambda_anchor_gate_eff": answer_loss.new_tensor(anchor_gate_weight),
        }
