# 代码地图

## 当前主版本：main/native_v9

| 文件 | 职责 |
| --- | --- |
| `run.py` | 配置合并、训练/测试、同 run 与相邻阶段 checkpoint 限制 |
| `src/models/trace_role_native.py` | native-v9 角色轨迹、Stage1 角色损失、Stage2 answer/latent RL 与 SFT replay |
| `src/modules/role_native.py` | PLAN / SOLVE1–5 / REFINE / COMMIT、角色目标、Gaussian policy 与信用分配 |
| `src/models/trace_bridge.py` | 问题与全轨迹读取、BRIDGE 训练路径与共用功能 |
| `src/models/read.py`、`read_efficient.py` | 依赖关系、压缩、latent 与 anchor 等底层实现 |
| `src/models/model_base.py` | 基础模型、LoRA、优化器与 trainable-only 权重保存 |
| `src/configs/models/trace_role_native_qwen3_instruct.yaml` | v9 主配置；仅将本机模型路径改为环境变量 |
| `src/configs/datasets/`、`src/datasets/` | 数据配置与 QSA/JSONL datamodule |
| `scripts/run_full_native_v9.sh` | 三阶段 pipeline、100-step recovery、严格验证与最终谱系记录 |
| `scripts/select_last_checkpoint.py` | 跨当前阶段目录选择最近的有效续训 checkpoint |
| `scripts/select_best_checkpoint.py` | 跨当前阶段目录选择最佳 monitored checkpoint |
| `tools/run_native_v9_strict_validation.py` | 五次单卡验证、747 unique ID 与数据 SHA-256 审计 |
| `scripts/verify_lineage.py` | 三阶段谱系检查 |
| `tests/test_role_native.py`、`test_lineage_guard.py`、`test_strict_validation.py` | CPU 单元检查 |

主目录还保留 CoT、CoLaR、iCoT、Coconut、Distill、TRACE-Bridge/MultiPath 等已有基础实现。存在代码不等于该 baseline 已在 v9 协议下完成实验。

## 后续改进版本（设计文档，不是现成代码）

- [v9 纠错验证版](../versions/v9_correctness_validation/README.md)：保留原角色架构，安排输入、COMMIT 梯度、噪声和采样概率修复。
- [结构统一的新方法版](../versions/next_structured_method/README.md)：规划六段 SOLVE、READOUT、统一目标和固定教师。

各版的 `code/README.md` 标明未来代码位置，`DESIGN.md` 链接本页中的原始文件作为修改依据，`ACCEPTANCE.md` 的项目均尚未执行。详细区别与状态见 [VERSION_ROADMAP.md](VERSION_ROADMAP.md)。不得把这些目录计入已实现的主代码或已完成的消融。

## 原样归档

`archive/` 和 `ablations/` 中的代码保留所属历史版本，不进行跨版本拼接。早期代码可能依赖历史 checkpoint、旧环境和绝对路径；不能作为 v9 的初始化来源。

## 本次整理修改范围

只在这个独立副本中添加目录说明、数据/来源清单、核心环境记录、CPU 检查和便携入口；主训练脚本及配置只做路径替换。模型实现、损失、正式数据字节、训练 batch/epoch/seed 配置不改动。

`docs/source_manifest.json` 为主版本的 85 个复制文件逐一记录来源和原始/整理后 SHA-256；`packaging_modified` 标记副本中的差异。11 个历史源码目录另外记录文件数量，以及按相对路径和文件 SHA-256 生成的确定性树哈希（覆盖实际复制的文件集合，不代表整个原工作区）。运行中的原工作区未被修改。
