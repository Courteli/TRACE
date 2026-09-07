# TRACE v9 纠错验证版

> 状态：DESIGN_ONLY。这里已经整理好方案，但尚未实现、测试或训练。它不是原 v9 的严格验证脚本，也不是已验证成功的模型。

版本标识：`v9_correctness_validation`。基线为初始提交 `f982ee102ca3d29dc0e5c0e8af424c9d896d4050` 中的 [native-v9](../../main/native_v9/README.md)。

本版回答：保留原八角色架构时，消除输入条件、重放梯度和采样概率的不一致后，原方法是否可靠？

- [DESIGN.md](DESIGN.md)：四项实现修复、一个独立信用分配调整及修改位置。
- [ACCEPTANCE.md](ACCEPTANCE.md)：全部待执行的验收项目。
- [STATUS.json](STATUS.json)：机器可读状态。
- [code/README.md](code/README.md)：后续代码的放置位置；当前没有可运行实现。
- [两版总览](../../docs/VERSION_ROADMAP.md)：与原 v9、新方法版的区别。

保留 `PLAN → SOLVE1–5 → REFINE → COMMIT`，不在本版同时改成六个 SOLVE 或冻结教师方案。原数据划分、原 v9 目录和正在运行的实验均保持不动。
