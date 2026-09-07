# TRACE Role-Native v9

This project is a fresh, end-to-end successor to the verified 72.3% TRACE
code line.  Its `src/` started from that run's exact source snapshot; no old
TRACE checkpoint is a component of the model or an initialization source.

The formal lineage is deliberately closed:

1. **Stage 0 — text-CoT SFT:** Qwen3-4B-Instruct + rank-64 LoRA is trained on
   the existing GSM8K textual chains of thought.
2. **Stage 1 — role formation:** a newly instantiated model learns the proven
   question + eight-latent + two-compact-anchor answer path.  The eight causal
   states are natively PLAN, SOLVE1-5, REFINE, and COMMIT.  Ordered contiguous
   spans of the observed gold CoT provide role supervision; no inferred step
   labels, hard negatives, counterfactual edits, or external memory are used.
3. **Stage 2 — joint latent/answer RL:** exact-answer group reward trains answer
   tokens, while per-role text-CoT alignment supplies discounted process credit
   to the latent Gaussian policy.  A frozen copy of this same run's Stage-1
   role policy provides KL stabilization.  SFT replay preserves the reliable
   information bridge.

Training-time validation covers the complete 747-example split at every epoch;
four-rank DDP reports 748 sampled entries because its sampler pads by one.  The
formal final metric therefore uses five independent single-device passes over
all 747 unique validation questions.  Each pass audits question IDs and checks
that the reported accuracy is exactly `correct_count / 747`.  The evaluator
also rejects any validation file whose SHA-256 differs from the immutable data
used by this run.  The formal run
uses four GPUs, ten Stage-1 epochs, and ten Stage-2 epochs.  `run.py` rejects
role-native initialization checkpoints outside the fresh run directory and
verifies that each stage consumes only its direct predecessor.

The formal supervisor scans all eight GPUs every ten seconds and acquires idle
cards one at a time.  Each acquired card continuously runs an isolated,
single-device Stage-2 small-batch stability replicate; these useful reservation
experiments never write into the formal run lineage.  As soon as four cards are
held, the supervisor stops the four reservation process groups and hands the
same physical cards directly to formal training.  Per-card guards preserve the
launch-time process baseline and reject newly arriving same-user GPU processes;
the four-card run has the same protection.  The formal run records a rolling
exact-recovery checkpoint every 100 optimizer steps and keeps Stage-0, Stage-1,
Stage-2, and final-evaluation completion markers separate.  A failed process
therefore returns to the same fresh-run stage and optimizer state; it does not
fall back to a historical checkpoint or silently skip evaluation.

Launch the complete pipeline with:

```bash
PHYSICAL_GPUS=0,1,2,3 scripts/run_full_native_v9.sh /disk1/dingxukai/TRACE/run_name
```
