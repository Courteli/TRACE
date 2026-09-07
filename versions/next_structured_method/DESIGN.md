# 已实现的无依赖图结构统一方法

本文件描述 `code/` 的实际实现及候选默认值。先前设计保留在 Git 提交 `6b5e803`；2026-09-08 收敛后的实现不是原 v9 的热更新。CPU 正确性测试不证明方法性能。

## 1. 数据和推理的信息边界

仅使用普通 JSONL 的 `id/question/cot/answer` 四字段。加载器固定文件哈希、数量、唯一 ID / 题目；全数据审计检查 split 间题目不重叠。JSONL 按物理文件行读取，CoT 仅按换行符分步骤，不把 JSON 字符串中的 Unicode 行分隔符误当成新记录。

没有依赖矩阵、置信矩阵、图编码器、图压缩分配、图损失或旧 view embedding，也不导入 native-v9 模块。显式锚点保留为轻量输出标签：从普通 CoT 的约 1/3、2/3 进度位置取最多两个步骤，优先提取已有算式，否则取短文本，每条最多 72 字符。短 CoT 去重。它们不是新增图标注。

训练时 CoT / 答案只用于标签、固定教师目标和奖励。学生 latent 的唯一入口是 `roles(question)`；生成只读取问题、全部 latent 和固定格式头，由模型自己生成锚点及最终 `Answer:` 字段，绝不把金标准锚点当推理输入。答案奖励只解析明确的最终答案字段，不奖励锚点中的偶然正确数字。

## 2. 角色与连续动作

默认八个 latent：PLAN、六个共享策略头的 SOLVE、确定性 READOUT。SOLVE 数可配置，目标、head、mask、奖励和检查点 schema 随之更新；六不是推理步数的自然常数。

每个随机动作是 16 维对角高斯。各角色输入为：

~~~text
latent_input_k = bridge(question_state)
               + 0.1 * learned_query_k
               + 0.05 * tanh(action_projection(a_k))
~~~

PLAN / SOLVE 依据前一因果状态和角色位置产生动作。READOUT 使用当前均值，答案解码仍可读取全部 latent，因此它不是唯一信息瓶颈，也不宣称具备没有纠错监督支持的 REFINE 功能。

采样采用原始 softmax、temperature=1，无 top-k / top-p 或隐藏 logits processor。底座及 LoRA dropout 关闭，但训练梯度保持开启。每题先收集完整 G=8 组，再重放并积累梯度；组内及全局 question batch 内不更新参数。

## 3. 同一目标及坐标

本谱系新 Stage0 的 LoRA 教师执行 eval + no_grad。取问题格式头最后 token 状态及每个非空 CoT 步骤最后内容 token 状态（在该步骤换行符之前），统一做无参数 layer_norm，乘固定随机投影 P，得到 h_0,…,h_m。P 默认 256 维、种子 1701，并随检查点保存；学生使用完全相同的归一化和 P。

~~~text
delta_j = h_j - h_(j-1)
d_k = sum(delta_j in contiguous, disjoint span I_k)
c_k = sum_(i<=k) d_i
sum_k d_k = h_m - h_0
~~~

区间长度相差最多一，余数分配到前面的 SOLVE；短 CoT 的尾部空槽无监督 mask，非空零向量保留 Huber、关闭其 cosine 项，空 CoT 拒绝。

令学生投影状态为 z_PLAN、z_SOLVE1…K、z_READOUT：

- PLAN 预测所有 d_k，是前瞻表示监督，不是对“人类式规划”的证明。
- SOLVE 第 k 段预测为 z_k - z_(k-1)，其中 z_0=z_PLAN。
- 累计几何为 z_k - z_PLAN 对齐 c_k，只约束 SOLVE 子链。
- READOUT 对齐绝对末边界 h_m。

一份 Targets(solve, span_mask, cumulative, end) 同时服务上述损失和过程评分。不存在另一套解释全部八位置的旧 BRIDGE 路径目标。

距离 D 为逐维平均 smooth-L1 加 0.1 倍有效方向的 cosine distance。默认 SFT：

~~~text
L_SFT = L_answer + 1.0 L_anchor
      + 0.08 D_PLAN + 0.12 mean_valid D_SOLVE
      + 0.14 mean_valid D_cumulative + 0.10 D_READOUT
~~~

answer / anchor 均按有效 token 求和后除固定 96，不除每条回答自身长度。Stage0 是全文 CoT + Answer 的 token NLL，固定除以 768。上述数值是可测试的候选设置，未声称调参最优。

## 4. 固定教师、评分头和扰动 SFT

训练 CoT 目标缓存固定；禁止为验证或测试构建训练缓存。清单记录实际 base 权重内容、Stage0 完整检查点、tokenizer、数据、代码、配置、运行库版本、前向 dtype、投影及目标文件哈希；逐样本校验 ID / 题目 / CoT / 答案。任何不匹配拒绝继续。

Stage1 前半程走均值路径，后半程 25% 样本采用同一动作机制、固定标准差 0.12 的高斯扰动。SFT 不训练探索方差；默认 RL 初始标准差也是 0.12，避免阶段切换时无声明地增大噪声。Stage2 再学习 log_std，限制在 [-2.5, 0.5]。

Stage2 起点分别复制并冻结 Stage1 的角色策略和 PLAN 评分头；学生 PLAN 预测头仍可训练。二者与教师缓存是三个不同对象，均保存身份，恢复时不重新初始化。

## 5. GRPO-based 混合目标和有界过程信用

每题各轨迹的终局奖励 R 是答案字段的数值 exact match。答案优势为同题组内去均值 / 标准差（分母下界 1e-4）。没有价值网络。

过程分数在 [0,1] 内：

~~~text
r_PLAN    = exp(-D_PLAN)            # 固定 Stage1 PLAN 评分头
r_SOLVE_k = exp(-0.5*(D_SOLVE_k + D_cumulative_k))
r_READOUT = exp(-D_READOUT)
U_k = sum_(j>=k) gamma^(j-k) mask_j r_j
    / sum_(j>=k) gamma^(j-k) mask_j
~~~

gamma 默认 0.9；READOUT 分数只出现一次，但进入前序动作的归一化折扣回报。终局答案奖励不放入 U_k，避免重复计数。空监督槽仍可执行动作，其局部过程标签屏蔽；有效后续状态可提供信用。

过程优势按同题同角色的 U 去均值，再除 max(std,0.1)，截断到 [-2,2]，因此极小差异不会放大到单位尺度。系数 beta 从 0.15 线性降至最后一次更新的 0（单更新 smoke 例外）。只把 beta*A_process 加到随机 latent 优势；答案 token 优势只用 A_answer。

~~~text
L_RL = mean_questions mean_group [
         sum_valid_token clipped_surrogate(A_answer) / 96
       + sum_random_role clipped_surrogate(A_answer + beta*A_process)/(K+1)
       + 0.02 KL_role - 0.001 entropy_role
       ] + 0.05 mean_questions L_SFT_mean_path
~~~

clipped_surrogate 使用 PPO min-ratio 形式，clip epsilon=0.12。Gaussian log-prob 对动作维求和；KL 对动作维及角色取平均，entropy 对动作维求和后对角色取平均。角色 KL 只是在当前前状态上约束固定参考策略，不约束整个答案分布。

前 K+1 动作 detach 固定重放，READOUT 用当前参数可微重算；不把 READOUT 算作随机动作。答案 token 记录显式长度和首个 EOS，兼容 PAD=EOS。各项梯度在同一参数快照积累，全局按实际问题数平均，不使用填充样本。

这是 GRPO-based 的联合离散/连续混合 surrogate，包含确定性参数路径、不同固定尺度、KL、SFT replay 和辅助过程优势；不能宣称是精确无偏联合轨迹策略梯度。在线单次更新时 ratio 初值为 1，clip 通常不实际激活，多轮 PPO 不在本实现内。

## 6. 完整谱系、恢复与模型选择

默认新 Stage0 3 epoch → 提取固定训练缓存 → Stage1 10 epoch → Stage2 10 epoch（每轮最多 2048 个不放回训练问题）。LoRA rank=64、alpha=32；SFT LR=1e-5、RL LR=5e-7、全局问题 batch=4。完整配置见 code/src/trace_structured/config.py，可通过严格 JSON 覆盖；未知旧参数报错。

每次更新按实际全局问题数归一化；分布式训练和验证都不补齐重复题目，空本地末批也参与梯度规约。优化器、日程、固定参考、逐 rank RNG、epoch / offset / step、父检查点和缓存哈希一同保存，精确恢复要求相同 world size / 代码 / 配置 / 基座 / tokenizer / 数据 / 运行库 / 精度。CUDA RNG 仅记录当前 rank 设备，不访问其他卡。

Stage1/2 每个完整 epoch 在唯一验证集上评测，记录全部逐题结果和指纹；全程最佳（平分取最早）才是下一阶段父检查点。完成最后 checkpoint 后的 best 标记和汇总可幂等补写。最后执行同一 Stage2 best 的五次单设备、每次 747 unique 验证，并核验整个谱系。五次确定性重复是可重复性检查，不是五个独立统计样本。

新目录不覆盖原 v9；旧 checkpoint 显式拒绝。本次不复用旧 Stage0，也不对现有 Stage2 进行热切换。

## 7. 尚未解决的研究与工程验证

固定特征对齐仍可能偏爱单一参考解法；退火不是策略不变性证明。固定随机投影的有效性、均值部署与随机探索差距、锚点必要性、角色是否真的学到分工均需实证。

当前日志有分量 loss、结构项、概率差异、全模型总梯度范数及小型专属 head 梯度探针；探针不是完整模型各损失的全量梯度归因。CPU tiny-Qwen 测试不替代真实底座的 BF16 / CUDA / NCCL / 显存 / 吞吐测试。本版没有正式实验结果，不预断优于原 v9。
