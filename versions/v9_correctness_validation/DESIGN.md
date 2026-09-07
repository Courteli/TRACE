# v9 纠错验证版：具体改进与实现落点

状态：设计待实施。以下“现状”指基线提交 `f982ee1`，结合 2026-09-05 的诊断；“修法”和代码片段是未来实现要求，不是当前仓库已经完成的补丁。

## 1. 保持什么不变

保持八个角色、16 维连续动作、共享 SOLVE 策略头、小幅残差注入、问题与全部 latent 的答案读取方式。暂时保留原角色目标、BRIDGE 路径监督、锚点和 SFT replay，不同时引入新方法版的目标重设计。

这只能控制变更范围，不代表原角色语义和所有辅助损失已经充分合理。REFINE 不应被描述成已有证据支持的自主纠错，COMMIT 也不是唯一答案瓶颈。

## 2. 四项实现修复

### F1：统一默认 view 条件

原位置：[trace_role_native.py](../../main/native_v9/src/models/trace_role_native.py)，`_question_only_latents()`；相关调用见 [trace_bridge.py](../../main/native_v9/src/models/trace_bridge.py) 的 `forward()`、`eval_generation()`，以及 native rollout/replay。

现状：SFT 和验证明确传入零号 `trace_view_ids`；native RL 未传时，不添加 view embedding。它们不是同一个条件。

拟改动：

- 在统一入口把省略值解析成 batch 大小的零号 view 张量，处理正确的 device/dtype。
- rollout、动作重放、SFT replay 和评测共享这个约定。
- 若确需无 view 模式，使用另一个显式选项；不要让 `None` 同时承担两个含义。
- 不删除原 Stage1 使用过的 view embedding 参数，避免把不兼容迁移当作最小修复。

验收时锁定动作、噪声、dropout/执行模式；省略 view 与显式零号 view 的输入、latent 和答案 log-prob 应一致。

### F2：重算确定性 COMMIT，接通答案路径梯度

原位置：[role_native.py](../../main/native_v9/src/modules/role_native.py)，`RoleLatentPolicy.realize()`；重放入口为 `trace_rl_training_step()`。

现状：`forced_action` 分支优先，连 rollout 中无梯度保存的 COMMIT 也被重放；该专属均值头没有经由这一答案 RL 路径更新。

预期分支语义如下，尚未实现：

~~~python
if role_index == COMMIT_INDEX:
    action = mean
elif forced_action is not None:
    action = forced_action.detach()
elif deterministic:
    action = mean
else:
    action = mean + log_std.exp() * noise
~~~

前七个随机动作固定重放；COMMIT 由当前网络重新生成。不把 COMMIT 放入随机动作策略损失，也不对其虚构 Gaussian log-prob。仅给保存动作设置 requires_grad 并不能重新连接策略头。

验收须关闭 SFT replay 和其他可向 COMMIT 头提供梯度的辅助项，构造非零答案优势，确认答案路径独自给 COMMIT 头带来有限非零梯度。还要确认前七个动作数值和重放概率条件没有改变。

### F3：恢复 query-noise 参数的实际作用

原位置：native `_question_only_latents()`；缩放位置参照 [BRIDGE 同名函数](../../main/native_v9/src/models/trace_bridge.py)。

现状：native 接收 `trace_noise_std`，却没有使用该参数。view embedding 仍有效，不能据此说整个 multiview 都没有工作。

拟改动：

- 按原 query-noise 的注入顺序、query scale 与 gate 顺序实现，不随意把噪声改加到最终 latent。
- 明确区分 SFT 主视图、多视图、RL rollout、RL replay 和评测的噪声约定。
- 评测和受控 RL 概率比较不引入未记录的 query 噪声；如以后启用，必须保存并重放同一扰动。
- 零噪声路径不应额外消耗随机数；固定 seed 时非零噪声可重现。

恢复此参数会改变原 SFT 训练行为。仅做 Stage2 诊断不能冒充“Stage1 也已纠正”的完整实验。

### F4：统一答案采样与概率计算

原位置：[trace_role_native.py](../../main/native_v9/src/models/trace_role_native.py)，`_generate_from_role_outputs()`、`_answer_logprobs_from_role_outputs()` 和 `native_role_rollout()`。

现状：答案采样使用温度/top-p，而记录概率使用原始 softmax，二者不严格对应。

首选的简化契约：

- 温度为 1，top-p 为 1，top-k 显式禁用。
- 审计继承的 generation_config 和 logits processors，包括重复惩罚、最短长度、强制 token 等，避免隐式改变采样分布。
- old/current log-prob 对同一前缀、同一 token 和同一采样条件计算。
- 若保留任何处理器，要么匹配其真实分布，要么明确排除非采样强制 token；不能只改温度就宣称完全对齐。
- 有效 EOS 计入一次，PAD/EOS 同 ID 时也不能用简单 token 值判断有效长度。

先通过小词表分布和 prompt/mask 测试，再做模型级集成检查。这里给出的是待实现契约，不是已经安装的可运行配置。

## 3. D1：末角色过程分数向前传播——单独记录的设计调整

原位置：[role_native.py](../../main/native_v9/src/modules/role_native.py)，`discounted_role_returns()`；旧测试见 [test_role_native.py](../../main/native_v9/tests/test_role_native.py)。

原测试明确要求排除 COMMIT 分数，因此这是现有信用分配约定的改变，不能称为偶发漏写。推荐实现为独立开关/提交，分别记录 pure-fixes 和 fixes-plus-terminal-credit 的配置身份；开关名称尚未定，不是当前可用参数。

若保留末状态评分，令回报从它开始向前递推：

~~~python
running = step_scores[:, COMMIT_INDEX]
for index in range(COMMIT_INDEX - 1, -1, -1):
    running = step_scores[:, index] + gamma * running
    returns[:, index] = running
~~~

COMMIT 的随机策略 mask 仍为 False，输出形状契约需明确；终局答案奖励走原独立通道，不能再在这里重复加入。

只给 COMMIT 评分 1 时，紧前角色回报应为 gamma；gamma=1 且八个分数全为 1 时，首角色回报应为 8。旧约定若保留为对照，必须使用不同配置身份，不静默改掉同名历史实验。

## 4. 不混入最小修复的选择

- 保留每批 rollout 一次参数更新，不为使 clipping 产生效果而自动改成多轮 PPO。
- 现有 latent/answer 分别平均再加权的 surrogate，按实际约定说明；统一归一化属于新方法版的显式设计。
- 在线 stop-gradient 教师不是固定教师；Stage1 局部角色 KL 也不约束整个答案模型。准确记录，不把它们统一称为 bug。
- 加上概率比、参考 KL、有效样本数、各项梯度和 finite 检查；需要固定评估模式的概率一致性测试应关掉 dropout 等额外随机源。

## 5. 后续代码放置与运行来源

待实现的代码根目录为本版 [code/](code/README.md)，主要改动应落在其未来的 `src/models/trace_role_native.py`、`src/modules/role_native.py`、模型配置和测试中；本页链接到原 v9 仅用于定位，不能就地覆盖它。

实现与测试以 [ACCEPTANCE.md](ACCEPTANCE.md) 为准。新建输出目录和版本标识。若从原 Stage1 权重建立兼容诊断，需要显式派生加载与来源校验，不得关闭原谱系检查；完整纠正版若改变 Stage1 噪声训练，则从同一合规 Stage0 重新训练 Stage1。
