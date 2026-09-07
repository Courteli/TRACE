# TRACE Role-Semantic Latent RL v1

This directory is an isolated implementation of the new experiment. It reads
the hash-registered datasets under `/disk1/dingxukai/TRACE`, but all new
checkpoints, logs, pipeline records, and evidence are written under
`/disk1/dingxukai/TRACE/role_semantic_runs`.

## Model program

The model executes one fixed eight-state latent program:

`PLAN -> SOLVE1 -> SOLVE2 -> SOLVE3 -> SOLVE4 -> SOLVE5 -> CHECK -> COMMIT -> Answer`

- PLAN, SOLVE, CHECK, and COMMIT have distinct mean heads; the five ordered
  SOLVE positions share one functional head and differ only through their step
  embeddings and recurrent state.
- PLAN, the five SOLVE positions, and CHECK are stochastic. COMMIT is the
  conditional mean and is excluded from entropy, PPO, and KL masks.
- The recurrence is the within-question working memory. No separate retriever,
  memory store, or unobservable role annotation is introduced.
- The answer decoder sees the question and COMMIT only. The seven private
  reasoning states cannot be read directly by answer tokens.

## Stage 1: role-semantic latent formation

Each registered training question contributes its one original textual CoT.
There are no generated rationales or synthetic role labels.

- PLAN target: a deterministic global summary of the frozen CoT teacher states.
- SOLVE targets: five balanced, contiguous residual chunks. If the CoT has
  fewer than five steps, empty slots are masked and its final observed step is
  anchored at SOLVE5.
- CHECK target: the residual correction from SOLVE5 to the teacher endpoint.
- COMMIT: trained only through the answer/compact-target likelihood, so it must
  compress the useful preceding computation for the COMMIT-only readout.

Three exchangeable posterior paths and one question-only conditional-mean path
share the same role losses. The Stage-1 objective contains answer likelihood,
PLAN/SOLVE/CHECK cosine losses, posterior-to-prior KL, and a role-dependent
minimum-entropy loss whose floor decreases from PLAN through CHECK.

## Stage 2: latent-only PPO

Stage 2 freezes the recurrent backbone, transition semantics, CoT teacher, and
answer decoder. Only the stochastic role-policy density is updated.

For each question, eight IID question-only paths receive:

- terminal exact-answer reward;
- a dense, group-standardized frozen gold-answer score;
- training-only PLAN/SOLVE/CHECK cosine rewards from the original CoT; and
- the registered output-length penalty.

Discounted role returns are standardized within each question and position.
PPO, Stage-1 policy KL, and the decreasing role-entropy schedule apply to the
seven stochastic roles only. COMMIT has zero policy advantage and remains
deterministic. Validation and test never expose the CoT to the policy.

## Formal protocol

- Stage 1: full 6,726-question train split, full 747-question validation every
  epoch, at most 10 epochs, patience 4, 16,820 scheduled optimizer steps.
- Stage 2: 2,048 unique questions per epoch, group size 8, 512 updates per
  epoch for 10 epochs, full validation every epoch, 5,120 scheduled updates.
- Evaluation: deterministic single-path (`test_times=1`, seed 0), full GSM8K
  test (1,319), GSMHard (1,319), SVAMP (1,000), and MultiArith (180).
- Mechanistic evidence: paired Stage-1/final 200-question x 8-path records,
  one shared train-fit PCA, role/COMMIT audits, 2,000 bootstrap and 1,024
  permutation geometry tests, plus 10,000-bootstrap causal/task summaries.
- The pipeline writes `COMPLETE.json` only after the evidence completeness gate
  verifies every required artifact and sample count.

## Entry point

```bash
bash scripts/run_full_pipeline.sh \
  1,4,6,7 \
  /disk1/dingxukai/TRACE/logs/cot_qwen3_instruct/gsm8k_aug_nl-gsm8k_aug_nl/20260720-072412_674141_20260720-041426_trace_full_seed0_stage0/checkpoints/epoch1__step3364__monitor0.848.ckpt \
  1
```

The entry point first runs unit/data/pipeline audits, a real one-GPU stress
test, and a real four-rank three-update DDP test. Any failure stops the formal
training before Stage 1 is launched.
