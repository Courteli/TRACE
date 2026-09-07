# TRACE 改进版本：位置、区别与实际状态

更新：2026-09-08。2026-09-07 的 `6b5e803` 只归档了两条设计路线；本次实现其中进一步收敛的无依赖图新方法。原 v9 正式运行未切换。

| 版本 | 位置 | 状态 |
| --- | --- | --- |
| 原正式 native-v9 | [main/native_v9](../main/native_v9/README.md) | 原源码及结果归属保留，未改动 |
| v9 纠错验证版 | [versions/v9_correctness_validation](../versions/v9_correctness_validation/README.md) | DESIGN_ONLY，仍未独立实施 |
| 无依赖图结构统一版 | [versions/next_structured_method](../versions/next_structured_method/README.md) | IMPLEMENTED_CPU_TESTED；正式 GPU 训练未运行 |

## 1. 新实现在哪里

新方法目录已有真实的 code/run.py、src/trace_structured、scripts/test_cpu.sh 和 tests；不是复制旧目录后改名字。详见其 [运行说明](../versions/next_structured_method/code/README.md)、[实现设计](../versions/next_structured_method/DESIGN.md)、[审查记录](../versions/next_structured_method/REVIEW.md) 和 [状态](../versions/next_structured_method/STATUS.json)。

纠错验证版仍只有设计和预留 code/README.md；本次没有把新模型的测试结果记在纠错版名下。

## 2. 两条路线的区别

| 维度 | v9 纠错验证版（仍为计划） | 无依赖图新方法（已实现） |
| --- | --- | --- |
| 目的 | 检查原 v9 输入、动作重放、概率契约 | 统一 SFT 学习结构与 RL 辅助评分 |
| 角色 | 原 PLAN → SOLVE1–5 → REFINE → COMMIT | PLAN → SOLVE1–6 → READOUT |
| 数据 | 原 QSA / dependency 数据格式 | 已有普通 CoT JSONL，仅四字段 |
| 图及旧路径目标 | 暂不重写原结构目标 | 无图输入/模块/损失；仅 SOLVE 子链统一几何 |
| 显式锚点 | 原机制 | 普通 CoT 固定文本进度取标签，推理自己生成 |
| 教师与评分 | 保留并审计原有在线目标 | 新 Stage0 冻结目标缓存；Stage1 固定评分副本 |
| SFT 扰动 | 修复原 query-noise 分支 | 同一高斯动作机制，后期固定小方差扰动 |
| RL | 对齐采样、重放及 COMMIT 梯度 | READOUT 可微重算、有界退火过程信用、固定尺度联合 surrogate |
| 成绩 | 无本版结果 | 有 CPU 正确性测试，无正式训练结果 |

新方法继承了输入/概率/梯度契约，但这不代表另一个纠错目录也已实施，更不证明新方法优于原 v9。过程对齐不是正确性 oracle，GRPO-based 混合目标不是无偏联合策略梯度的证明。

## 3. 数据和实验隔离

原 v9 保留 data/GSM8k-Aug-NL/readcot_qsa_qwen_dc；新方法直接读取同级 gsm8k_*_processed.jsonl。两者题目划分对应 6726 / 747 / 1319；不删除历史图数据，不转换或重新切分普通文件。详见 [数据说明](DATASETS.md)。

新流水线从新的 Stage0 开始，随后重训 Stage1 / Stage2；不复用旧 Stage2 断点或优化器。完整权重、缓存和产物保留在新的独立输出目录，不提交 Git，也不覆盖正在运行的源目录和正式 run。

## 4. 完成标准

新版本当前状态为 IMPLEMENTED_CPU_TESTED，而非 EXPERIMENT_COMPLETED。已运行的 tiny-Qwen / LoRA 测试包括三阶段、合成验证五次重测、恢复和 CPU 双进程规约；不应把合成验证的两条样本写成正式 747 条评测。

后续须另行确认资源，完成实际底座 GPU 集成和完整训练，再执行五次单设备 747 unique 验证及谱系审计。性能改善、锚点必要性、角色分工和随机/均值部署差距仍需要真实证据。
