# 消融代码索引

以下是实际找到并归档的消融实现，不代表这些实验都跑完，也不代表可以合并到 native-v9 的结果表。

| 所属版本 / 入口 | 已有对照 |
| --- | --- |
| `ablations/legacy_trace_colar/run_trace_stage1_component_ablations_gpu0_20260716.sh` | 去 path consistency、去 progress anchor、去 multiview |
| `ablations/legacy_trace_colar/run_trace_stage2_mechanism_ablations_waiter_20260716.sh` | 历史 Stage2 replay / Stage1 路径相关机制对照 |
| `ablations/legacy_trace_colar/run_trace_multipath_ablation_20260704.sh` | answer-only、no-mode、no-hard、no-step、L8/L16/L40 |
| `ablations/legacy_trace_colar/run_trace_trajectory_ablation_20260704.sh` | 历史 trajectory 分支对照 |
| `ablations/requested_suite_20260705/run_requested_suite.sh` | no-hard、no-mode、no-filter 小预算队列及 OOD/geometry 任务 |
| `archive/trace_vb_latent_rl_v*/tools/trace_policy_causal_summary.py` | 历史 action replacement / COMMIT readout 因果诊断 |

`legacy_trace_colar` 同时包含这些入口对应的 `src/`、`scripts/`、`tools/` 和预处理代码，避免只放一个无法找到模型定义的脚本。

## native-v9 的现状

未找到独立、完整的 native-v9 专用消融队列。本次整理不会假造消融实现或宣称旧版本结果属于 v9。v9 配置和代码中已有 role-loss、latent/answer policy、Stage1-policy KL、SFT replay 等参数入口；后续正式消融需单独制定同预算、同数据、独立 run 的协议，并保留谱系检查。

## 执行警告

历史脚本可能包含固定 GPU、绝对路径、旧 checkpoint、等待循环、`tmux kill-session` 或进程管理逻辑。本次仅复制，没有执行。不要在当前共享训练机器上直接运行这些历史队列。先调整独立输出目录与资源分配，再核对该版本的真实参数语义和环境。
