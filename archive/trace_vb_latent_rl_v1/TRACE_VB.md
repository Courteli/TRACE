# TRACE-VB

**Role-Structured Latent Reasoning with Semantic-to-Outcome Value Bridging**

TRACE-VB uses one coherent training chain:

> CoT teaches latent semantics; terminal outcomes teach latent utility; a
> calibrated sufficiency-to-value bridge connects the two.

The formal model has a fixed eight-state physical program:

```text
Question -> PLAN -> SOLVE1 -> ... -> SOLVE5 -> REFINE -> COMMIT -> Answer
```

PLAN, the five SOLVE states, and REFINE are stochastic Gaussian actions.
COMMIT is deterministic and excluded from the policy mask.  During answer
generation `answer_context_mode=path_only`: the decoder can attend only to
COMMIT, not to the raw question or earlier latent states.

The low-level shared-policy code retains `check` only as a compatibility key
for the seventh head.  All formal manifests, visual records, figures, and
paper-facing evidence name that position REFINE.  No claim of explicit error
verification is made because GSM8K does not provide error-state labels.

## Stage 0: registered CoT warm start

Stage 0 is not retrained.  Every formal run reuses the registered full-data
checkpoint and verifies its SHA256 before loading it:

```text
/disk1/dingxukai/TRACE/logs/cot_qwen3_instruct/
  gsm8k_aug_nl-gsm8k_aug_nl/
  20260720-072412_674141_20260720-041426_trace_full_seed0_stage0/
  checkpoints/epoch1__step3364__monitor0.848.ckpt
```

Expected SHA256:

```text
1e58984dcae9dfd6885a2d5a58c8948d2832a7e19ec467273f8f74546bc7aeaa
```

## Offline sufficiency cache

For each original CoT prefix, a frozen Stage-0 teacher computes normalized
answer log-likelihood sufficiency.  Prefixes at and after the first direct
answer reveal are masked.  Scores are generated once and cached at:

```text
/disk1/dingxukai/TRACE/trace_vb_runs/cache/
  gsm8k_prefix_sufficiency_v1.pt
```

Before the full cache is built, the same frozen teacher scores a deterministic
256-row pilot.  The pilot and then the full 6,726-row cache must independently
pass the following signal gates before Stage 1 may allocate the model:

- valid-row fraction >= 0.35;
- valid-prefix fraction >= 0.20;
- pre-leakage nonzero-gain row fraction >= 0.30;
- mean pre-leakage score span >= 0.05.

Leakage rate itself is diagnostic, not an optimization target: a correct
GSM8K solution normally states its answer in the last sentence.  The critical
contract is that the leaked and later prefixes are masked while useful signal
still exists before leakage.  Teacher inference never runs inside Stage 1 or
Stage 2.

## Stage 1: latent semantic formation

Stage 1 starts directly from Stage 0 and runs:

- one deterministic mean trajectory;
- one stochastic question-only prior trajectory;
- one answer teacher-forcing loss from deterministic COMMIT.

There is no CoT-conditioned posterior, multi-posterior sampling, hybrid
anchor, compact-equation decoder branch, or extra answer path.  This removes
the train/deployment mismatch and the memory-heavy branch structure that
caused the preceding role-latent run to OOM.

The training losses are:

```text
L1 = Lanswer-from-COMMIT
   + 0.10 LPLAN-forecast
   + 0.20 LSOLVE-semantic
   + 0.10 LREFINE-semantic
   + 0.10 Lsufficiency
   + 0.005 Lvariance-floor
   + 0.02 Lefficacy
```

PLAN forecasts the five future SOLVE targets in a 256-dimensional semantic
space.  The five SOLVE targets are monotone contiguous chunks of the original
CoT; steps are never reordered and no synthetic role labels are created.
REFINE maps to the final answer-ready teacher state.  Sufficiency gain weights
semantic retention but is not used as an RL reward.  The efficacy hinge
requires stochastic actions to produce a proportional change in the causal
latent transition, using RMS-normalized action and residual distances with a
minimum ratio of 0.02; COMMIT is excluded by the stochastic-action mask.

Stage 1 runs ten complete 6,726-question epochs with a full 747-question
validation after every epoch.  Python/DDP processes are recycled only at full
epoch boundaries to bound allocator growth; optimizer, scheduler, callback,
and RNG states are restored from `last.ckpt`.  There is no early stopping.

## Stage 2: terminal-outcome latent actor-critic

At the Stage-1/Stage-2 boundary, the role-conditioned sufficiency head is
copied into a role-conditioned outcome critic.  The actor remains frozen for
the first 256 **rollout batches**, while the critic is calibrated using actual
terminal exact-correctness outcomes.

After warm-up, the only environment reward is:

```text
R = 1[generated final answer == gold answer]
```

The legacy dense gold-answer score, CoT semantic step reward, output-length
reward, answer-token PPO, counterfactual replacement, and hard-negative
mining all have zero weight or are disabled.

With `gamma=1.0` and `gae_lambda=0.95`, masked GAE propagates terminal outcome
credit over the seven stochastic actions.  COMMIT has action mask zero.
Rollouts save pre-action states, actions, old log probabilities, role IDs,
action masks, values, and terminal rewards.  Four head-only PPO epochs reuse
those detached states and must not rerun Qwen.  Only the stochastic role actor
heads and critic are trainable; the backbone, transition dynamics, COMMIT,
answer decoder, and Stage-1 policy reference remain frozen.

Formal Stage 2 uses:

- 2,048 unique training questions per epoch;
- eight IID question-only trajectories per question;
- rollout microbatch size 1;
- four PPO epochs per rollout batch;
- actor learning rate `8e-7`;
- critic learning rate `1e-4`;
- PPO clip `0.12`;
- Stage-1 reference KL weight `0.02`;
- 20,480 optimizer/scheduler steps over ten epochs;
- full 747-question deterministic validation after every epoch.

## Resource gates

Formal training requires exactly four unique physical GPUs.  Before either
training stage begins, every selected GPU must report at least 21,500 MiB free.
The dedicated Stage-1 single-GPU and four-rank DDP smokes execute the real
long-sample objective, backward pass, optimizer-state allocation, deterministic
COMMIT readout, and require at least 4,096 MiB effective device-wide headroom
after charging external processes and non-allocator CUDA memory.

The Stage-2 smoke checks all of the following on a real Stage-1 checkpoint:

- critic parameters were initialized from the sufficiency head;
- eight real Stage-1 paths retain mean action-to-transition efficacy >= 0.01
  before RL is allowed to start (the training target remains 0.02);
- rollout includes every pre-action state and a COMMIT-zero action mask;
- actions, old action log-probabilities, values, advantages, and returns are
  all stored as FP32;
- terminal-only GAE returns are finite;
- four repeated PPO head updates change actor and critic parameters;
- the update makes zero backbone/Qwen forward calls;
- trainable parameters are limited to the PLAN/SOLVE/REFINE mean and log-std
  heads plus the value critic; the shared policy trunk, step embeddings,
  dynamics, and COMMIT head remain frozen;
- the peak-memory/headroom gate passes.

After that single-GPU gate, a second smoke loads the same real Stage-1 best
checkpoint on exactly four ranks.  Each rank collects its own group of eight
terminal-exact rollouts, performs one critic-only warm-up backward/optimizer
step, then two actor-plus-critic DDP updates over the cached FP32 states.  It
fails unless gradients and post-step parameters are synchronized across all
four ranks, the later PPO ratio is nonunit, the trainable whitelist is exact,
and no cached-state update invokes Qwen.  Formal Stage 2 starts only after both
Stage-2 smokes pass.

Any missing measurement or violated contract is fatal.  Memory thresholds may
not be lowered in a formal run.

## Full evaluation contract

The best full-validation checkpoints from Stage 1 and Stage 2 are evaluated
as a paired set.  The evidence pipeline preserves the previously registered
scope without reduction:

- deterministic GSM8K test on all 1,319 questions;
- complete GSMHard, SVAMP, and MultiArith OOD tests;
- 200 held-out GSM8K questions with eight latent rollouts each;
- paired Stage-1 sufficiency versus Stage-2 terminal-outcome value calibration
  (Brier score, 10-bin ECE, and correct/wrong value separation);
- a shared unlabeled global PCA fit on 200 training questions;
- paired Stage-1 versus Stage-2 geometry statistics;
- terminal-value calibration and same-norm single-transition interventions on
  200 questions: each PLAN/SOLVE/REFINE replacement holds the earlier actions
  fixed and recomputes the complete downstream suffix and deterministic COMMIT;
- a 0--8 state-availability curve used only to verify the strict COMMIT
  bottleneck (positions 1--7 equal no readout by construction), never as a
  stepwise contribution or prefix-sufficiency claim;
- 10,000 bootstrap draws with familywise intervals;
- `COMPLETE.json` only after every required artifact passes verification.

`visual_record_limit=200` limits only cached visualization records.  It does
not truncate the 1,319-question task evaluation.

## Running the formal pipeline

The only full-pipeline entry point is:

```bash
bash scripts/run_full_pipeline_vb.sh 4,5,6,7 4
```

For a detached, GPU-gated launch with a status file and fatal-log monitor, use:

```bash
python tools/trace_vb_supervisor.py --gpu-csv 4,5,6,7
```

The optional second argument selects the single physical GPU used for cache
construction and sequential evidence.  All outputs are isolated under:

```text
/disk1/dingxukai/TRACE/trace_vb_runs/
```

The stages may also be invoked separately with `prepare_sufficiency_cache_vb.sh`,
`run_stage1_vb.sh`, `run_stage2_vb.sh`, and `run_evidence_vb.sh`.  The formal
pipeline records source hashes, configurations, checkpoints, audits, smokes,
and final evidence paths in its manifest.

## Scope of the claim

The CoT sentence/chunk alignment is weak semantic supervision, not annotated
ground-truth reasoning state.  The current evidence supports GSM8K and the
specified arithmetic OOD suites; it does not establish a general persistent
memory mechanism.  TRACE-VB uses causal latent/KV state only as within-question
working memory.  The deployed answer decoder cannot reread the raw question or
private PLAN/SOLVE/REFINE states; it reads only deterministic COMMIT.  The
registered equal-norm action replacement is a post-training sensitivity audit,
not hard-negative mining, training-time counterfactual supervision, or proof of
human-readable role semantics.  TRACE-VB deliberately excludes cross-question
memory, dynamic halting, fresh-cache COMMIT re-encoding, training-time
counterfactual state replacement, and unobserved step-correctness labels.
