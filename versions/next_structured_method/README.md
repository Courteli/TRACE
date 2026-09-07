# TRACE：无依赖图的结构统一版本

状态：**IMPLEMENTED_CPU_TESTED**（2026-09-08）。已实现并完成代码审查及 CPU 回归；未进行本版本的正式 GPU 训练，没有准确率提升或顶会水平的实证结论。不替代正在运行的 native-v9。

2026-09-07 提交 `6b5e803` 是最初的设计归档。本次实现的是进一步收敛、完全不读取依赖图的独立版本，标识为 `trace-structured-graph-free-v1`。

默认链为 `PLAN → SOLVE1–6 → READOUT`。普通 CoT 的连续分段目标由冻结 Stage0 教师提取，统一用于 SFT、SOLVE 几何和 RL 过程评分；保留最多两个固定文本进度锚点，不使用图选锚点。

数据直接读取仓库已有的 `data/GSM8k-Aug-NL/gsm8k_{train,val,test}_processed.jsonl`，固定 6726 / 747 / 1319 条。新入口拒绝 `readcot_qsa_qwen_dc` 目录及 dependency/confidence 等额外字段。历史数据仍保留给原 v9，不删除、不重新序列化、不重新切分。

- [DESIGN.md](DESIGN.md)：实现的目标坐标、损失、采样、过程信用与局限。
- [code/README.md](code/README.md)：检查、独立三阶段流水线与恢复命令。
- [ACCEPTANCE.md](ACCEPTANCE.md)：已通过和仍待 GPU / 正式实验验证的项目。
- [REVIEW.md](REVIEW.md)：审查发现、修复和验证范围。
- [STATUS.json](STATUS.json)：机器可读状态及提交信息。

必须建立新的 Stage0 → Stage1 → Stage2 谱系；旧 Stage2、旧优化器以及没有兼容契约的旧 Stage0 均不能直接接入。
