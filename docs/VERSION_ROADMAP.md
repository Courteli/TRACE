# TRACE 两个改进版本：位置、区别与实施路线

整理日期：2026-09-07。依据：2026-09-05 的方法评审与改进讨论，以及仓库初始提交 `f982ee102ca3d29dc0e5c0e8af424c9d896d4050` 中的 native-v9 源码。

> 状态：本次新增的是两版的设计文档、目录规划和验收清单，不是两套已经实现或验证通过的训练代码。“验证版”表示它的用途，不表示已经通过验证；“新方法版”表示后续设计，不替代当前正式 v9，也暂不赋予未经确认的 v10 版本号。

## 1. 去哪里找

| 版本 | 仓库位置 | 当前状态 |
| --- | --- | --- |
| 当前正式 native-v9 | [main/native_v9](../main/native_v9/README.md) | 已归档的现有源码；保留原方法 |
| v9 纠错验证版 | [versions/v9_correctness_validation](../versions/v9_correctness_validation/README.md) | DESIGN_ONLY：方案已归档，独立实现尚未开始 |
| 结构统一的新方法版 | [versions/next_structured_method](../versions/next_structured_method/README.md) | DESIGN_ONLY：方案已归档，独立实现尚未开始 |

两版使用相同的目录结构：

~~~text
versions/
├── v9_correctness_validation/
│   ├── README.md        # 定位、状态、导航
│   ├── DESIGN.md        # 改进细节与待修改的原代码位置
│   ├── ACCEPTANCE.md    # 尚未执行的验收清单
│   ├── STATUS.json     # 机器可读状态与基线提交
│   └── code/
│       └── README.md   # 未来代码放置约定；现在没有可运行实现
└── next_structured_method/
    ├── README.md
    ├── DESIGN.md
    ├── ACCEPTANCE.md
    ├── STATUS.json
    └── code/
        └── README.md
~~~

真正实现时，各版自己的 `run.py/src/scripts/tests` 放进自己的 `code/`。这次没有把原 v9 复制两份并改名充当新版本，没有新训练入口，也没有新增模型结果。

实际数据和基础环境继续统一引用仓库的 [data](DATASETS.md)、[environment](ENVIRONMENT.md)；不复制、重新切分或修改正式数据。各版后续环境增量应写入自己的说明。

## 2. 两版到底改进什么

| 维度 | 纠错验证版：先把现有实现做对 | 新方法版：统一监督和优化 |
| --- | --- | --- |
| 核心问题 | 原 v9 的输入、动作重放及概率计算是否自洽？ | SFT 学到的结构能否直接成为 RL 优化的结构？ |
| 角色 | 保留 PLAN → SOLVE1–5 → REFINE → COMMIT | 拟改为 PLAN → SOLVE1–6 → READOUT，仍是八个 latent |
| 输入条件 | 统一 SFT、rollout、重放、评测的默认零号 view | 使用统一入口；扰动训练与 RL 共用高斯动作机制 |
| 确定性末角色 | 重算 COMMIT，接通答案路径梯度，不参加随机策略损失 | READOUT 同样保持确定性、可微，不虚构动作概率 |
| 过程信用 | 末角色评分向前传播是单独标注的设计调整，不冒充偶发 bug | 统一目标下的过程评分作为辅助结构正则，并控制尺度与退火 |
| 监督目标 | 暂不重写原角色和旧路径目标 | 一份不重叠分段目标供角色损失、SOLVE 路径损失和过程评分共用 |
| 教师与评分头 | 保留现状并准确说明在线目标、局部角色 KL 的范围 | 冻结本谱系 Stage0 特征提取流程；Stage2 使用固定评分标尺 |
| SFT 噪声 | 恢复原本接收却未应用的 query-noise 参数，限制使用分支 | 用同一高斯动作机制做后期扰动 SFT，限制探索方差 |
| RL 概率 | 采样分布与记录的 old/current log-prob 一致 | 继承概率契约，另明确联合 surrogate 的归一化 |
| 不应宣称 | 已修好、已提升、已经是新方法 | 已实现、已有结果、过程相似度就是正确性、已达到顶会水平 |

## 3. 验证版的具体工作

详见 [设计说明](../versions/v9_correctness_validation/DESIGN.md) 和 [验收清单](../versions/v9_correctness_validation/ACCEPTANCE.md)。

1. 将遗漏的 `trace_view_ids` 统一解析为零号 view，明确区分“零号 view”与“完全不使用 view”。
2. 重放时固定前七个随机动作，但由当前网络重新计算 COMMIT 均值；检查专属头确实收到答案路径梯度。
3. 恢复 `trace_noise_std` 的预期 SFT query-noise 行为；评测及受控 RL 重放不得无意加入新噪声。
4. 对齐答案采样和 log-prob 的分布；首选温度 1、无 top-p/top-k 截断的明确配置，并审计其他 logits 处理。
5. 将“COMMIT 过程分数向前传播”作为独立的信用分配设计变更，记录开关和测试，不与纯修复混为一谈。
6. 暂不扩大为多轮 PPO；记录概率比、梯度、参考状态和恢复语义。单次更新、在线教师和现有 loss 加权不一概归类为程序错误。

## 4. 新方法版的具体工作

详见 [设计说明](../versions/next_structured_method/DESIGN.md) 和 [验收清单](../versions/next_structured_method/ACCEPTANCE.md)。

1. 删除没有真实纠错监督支持的 REFINE 语义，将预算用于第六个 SOLVE；COMMIT 改为与实现相符的 READOUT。
2. 使用统一的教师步骤边界状态，将文本转移划分成六个连续、不重叠区间；每个转移只覆盖一次。
3. SFT 的角色和几何监督、RL 的过程评分共用这份目标；几何只约束 SOLVE 子链。
4. 缓存冻结 Stage0 教师的训练特征；记录教师、tokenizer、数据、提取规则及相关固定投影的指纹。
5. SFT 前期训练均值路径，后期引入与 RL 相同机制的受控高斯扰动；不再依赖额外视图间的强制几何分离作为核心。
6. 以答案反馈为任务目标，过程对齐仅作有界、可退火的辅助约束；评分头和教师目标保持固定。
7. 显式规定动作/token 聚合与固定尺度归一化，保留 PAD/EOS、分布式 reduction 和断点恢复的一致性测试。

这些是待检验的设计，不是提升准确率的保证。损失权重、噪声比例、方差上限和退火日程尚需确定，不在本文中伪造“已验证最优”数值。

## 5. 实施顺序与实验隔离

1. 固定原 v9 基线提交、正式数据指纹与原实验记录，继续保留 `main/native_v9/`。
2. 在验证版 `code/` 中实施最小修复，逐项建立回归测试；信用分配调整用独立提交/配置记录。
3. 完成独立小规模验收后，再决定该版的正式训练，不占用或重启当前正式任务。
4. 在新方法版 `code/` 中实现统一目标与固定教师，再接入 SFT、RL 和部署路径。
5. 新方法角色定义/监督变化后，从合规的 Stage0 权重重新训练 Stage1 和 Stage2，不能把旧 Stage2 的下一步直接写成新方法续跑。

验证版若仅做兼容的 Stage2 诊断，可评估从原 Stage1 权重开始的独立派生实验；这需要显式来源记录与经过测试的派生加载流程。若修复改变了 Stage1 噪声训练行为，完整比较需重新训练 Stage1。不得通过关掉谱系检查、复用旧优化器状态或覆盖原 run 来规避区别。

未来输出建议放在被 Git 忽略的 `runs/<version_id>/<new_run_id>/`，两版不得共用输出目录。大权重、特征缓存和训练 checkpoint 仍不进入 Git。

## 6. 如何判断“完成”

`DESIGN_ONLY → IMPLEMENTED_UNVERIFIED → TESTED → EXPERIMENT_COMPLETED` 是建议的状态流转，不是当前已有成果。

每次升级状态必须填入对应实现提交、测试命令和结果路径。当前两份 `STATUS.json` 明确记录：实现不存在、训练未运行、没有结果。原 v9 的 15 项 CPU 测试不能作为这两版已经通过测试的证据。
