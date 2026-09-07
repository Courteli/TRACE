# 实现后代码审查与验证记录

日期：2026-09-08。范围仅为 versions/next_structured_method/code 及对应说明；未对原 v9 进行热修补，未启动其恢复或巡检任务。

## 结论与审查方式

实现后逐模块自审，随后针对发现的边界问题补回归测试并重新执行全部测试；没有第二位独立审查者。当前没有已知尚未修复的阻断性发现，但 GPU 集成和正式效果仍未验证，不能把自审写成“已证明不存在 bug”。

已通过 29 项 unittest。用随机初始化 tiny-Qwen3 + LoRA 实际走通三阶段和恢复，不是只做 mock 张量形状检查。测试中的五次评测是每次 2 条合成验证样本，不是正式 747 条模型结果。

## 发现并修复

| 问题 | 修正与证据 |
| --- | --- |
| JSONL 用 str.splitlines 会把 JSON 字符串中的 Unicode 分隔符拆成新记录 | 改为迭代物理文件行；真实数据与 Unicode 回归通过 |
| 只按题目校验缓存不足以发现同 ID 的 CoT 变动 | 缓存保存完整样本内容指纹，并核验文件、教师、投影、库版本与精度；错误 CoT、教师、投影和文件损坏均拒绝 |
| 完成 last.ckpt 后、best/summary 写完前中断，恢复可能直接返回 | 提取幂等完成过程；缺失汇总可补写，再次恢复保持 best 内容指纹稳定 |
| 参数/父模型预检查失败可能先创建新 run 标记 | 先验证必要条件和直接父谱系，再写新运行标记 |
| 只检查汇总数量不足以证明全轮次选优和唯一题覆盖 | 核验逐 epoch 顺序、逐题结果、答案重算、指纹、last 完成状态和最早最高分 best；严格 pass 另绑定评测身份 |
| 默认 SFT 与 Stage2 初始方差存在不必要跳变 | 默认两者 std 都为 0.12；SFT 固定方差，RL 再学习 |
| 过程权重最后一次更新未真正到零 | 退火使用 total_steps-1；最后更新为零，单步 smoke 显式例外 |
| 合并 latent loss 无法直接观察过程辅助贡献 | 增加 latent-answer 与 process-increment 的 loss/head 梯度探针；增量是合并 clipped 目标减去答案目标，不假定 clipping 可加 |
| 保存所有 CUDA 设备 RNG 可能创建额外设备上下文 | 仅保存当前 rank 已初始化设备 RNG；不为了 CPU 测试触碰 CUDA |
| 新配置拒绝旧参数的测试误期望 ValueError | 改为实际严格 dataclass 的 TypeError；未知旧字段没有被静默接受 |
| 保护路径的测试曾硬编码本机 checkout | 改为按测试文件定位仓库，保证换目录克隆后仍检查真实保护路径 |

同时复核并测试：首 EOS / PAD=EOS 掩码、固定动作重放、READOUT 当前均值的答案梯度、固定评分/参考不被更新、训练缓存 train-only、teacher/student 坐标闭合、组优势隔离和下界、固定长度尺度、全局实际问题数规约、空本地末批和未使用参数保持 grad=None。

## 执行命令

从仓库根目录，在已有 ROT 环境执行：

~~~bash
TRACE_PYTHON=/home/dingxukai/miniconda3/envs/ROT/bin/python \
  bash versions/next_structured_method/code/scripts/test_cpu.sh

PYTHONDONTWRITEBYTECODE=1 /home/dingxukai/miniconda3/envs/ROT/bin/python -B tools/audit_repository.py

git diff --check
bash -n versions/next_structured_method/code/scripts/test_cpu.sh
~~~

新方法全部 15 个 Python 文件通过 AST 语法检查。仓库完整性审计 issues=[]：15 个数据文件、9 个模型元数据文件、85 个主代码来源文件、11 个历史树 / 1241 个历史源文件均保持原指纹。未上传任何模型权重、运行 checkpoint、优化器状态或大缓存。

测试脚本强制 CPU，双进程是 CPU/Gloo 子进程，不是训练卡任务。运行产物仅在临时目录，测试结束清理。

普通数据文件：6726 / 747 / 1319；归档 tokenizer 下最大 question / teacher / SFT-output token 数分别为：

| split | question | teacher | SFT output |
| --- | --- | --- | --- |
| train | 216 | 439 | 71 |
| val | 150 | 439 | 59 |
| test | 193 | 393 | 54 |

均在候选配置预算内，全部答案可按数字解析。此检查不将 test 标签用于训练或选优。

## 仍需后续验证的边界

- 本次没有加载真实完整基座做 GPU 训练。BF16 下 rollout/replay 数值误差、显存、吞吐、NCCL 多卡和真实进程崩溃恢复仍需单独资源授权。
- 三阶段正式训练、五次 747 unique 的真实验证没有运行；不能据 CPU 测试预报成绩。
- 各项 loss 和少数 head 梯度探针已经记录，不等于完整主干分量梯度归因；原始 metrics.jsonl 为追加事件日志，异常回滚后可能有重复 step，应按 checkpoint 游标及逐 epoch 验证证据判断完成，不按日志行数计训练步数。
- 缓存构建不是增量恢复协议：中断的非空缓存目录不会被覆盖，应排查后换独立新目录重建。
- 固定教师与单一 CoT 的对齐偏差、随机探索到均值部署的差距、显式锚点必要性、随机投影及角色分工的有效性依然属于待实证研究问题。
- 原 v9 的源代码、图数据与正在运行的实验未改；本版本必须重新建立 Stage0→Stage1→Stage2 谱系。
