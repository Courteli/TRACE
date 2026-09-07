# TRACE 结构统一的新方法版

> 状态：DESIGN_ONLY。这里是前次讨论的新方法设计，尚未实现、测试或训练。“新”指后续方案，不表示当前正式运行已切换，也不表示其效果优于 v9。

版本标识：`next_structured_method`。概念基线为提交 `f982ee102ca3d29dc0e5c0e8af424c9d896d4050` 的 [native-v9](../../main/native_v9/README.md)，并要求继承纠错验证版的输入、梯度和概率一致性契约。

核心改进是：让 SFT 的结构监督、路径约束和 RL 的过程评分使用同一份推理分解，不再给同一 latent 叠加不同的轨迹解释。

- [DESIGN.md](DESIGN.md)：六项方法改进、代码放置和待确定参数。
- [ACCEPTANCE.md](ACCEPTANCE.md)：全部待执行的正确性和集成检查。
- [STATUS.json](STATUS.json)：机器可读状态。
- [code/README.md](code/README.md)：未来实现的位置，当前没有可运行代码。
- [两版总览](../../docs/VERSION_ROADMAP.md)：与纠错验证版和原 v9 的区别。

拟采用 `PLAN → SOLVE1–6 → READOUT`；仍保留八 latent 预算、低维动作、共享 SOLVE 策略头和必要的显式锚点。角色定义和监督改变后，需要重新训练 Stage1/Stage2，不能接上旧 Stage2 断点宣称是同一次实验。
