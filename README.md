# TRACE — 私密研究仓库

整理日期：2026-09-07。主版本为 **TRACE Role-Native v9**；历史消融与旧版本分别归档，不与 v9 主实验混用。本仓库从独立副本整理，未改动正在训练的源目录、checkpoint 或进程。

两条后续路线分别位于 **[v9 纠错验证版](versions/v9_correctness_validation/README.md)** 和 **[无依赖图结构统一版](versions/next_structured_method/README.md)**。截至 2026-09-08，前者仍为 `DESIGN_ONLY`；后者已有独立实现、代码审查和 CPU 回归，为 `IMPLEMENTED_CPU_TESTED`，但尚未正式 GPU 训练、没有效果结论，也不替代当前 v9。先看 **[两版位置与状态总览](docs/VERSION_ROADMAP.md)**。

| 目录 | 内容 |
| --- | --- |
| `main/native_v9/` | 当前主代码、模型配置、三阶段训练、严格验证、谱系检查和单元测试 |
| `versions/v9_correctness_validation/` | v9 纠错验证版方案：输入、梯度、噪声、概率一致性；尚未实现 |
| `versions/next_structured_method/` | 已实现的无依赖图新方法：普通 CoT、统一目标、固定教师及 SFT/RL；CPU 测试通过，正式训练未运行 |
| `ablations/legacy_trace_colar/` | 历史 Stage1/Stage2、MultiPath、Trajectory 消融及其配套源码 |
| `ablations/requested_suite_20260705/` | 历史 no-hard / no-mode / no-filter 队列和几何分析工具 |
| `archive/` | pre-v9、role-v1、VB-v1/v2/v4/v5/v6/v7/v8 的代码快照；未找到独立 v3 源目录，不补造 |
| `data/` | 实际 GSM8K/QSA 训练、验证、测试文件，以及 GSM8K-Hard、SVAMP、MultiArith 数据 |
| `environment/` | 实际训练环境版本、完整 pip/Conda 快照及原始旧依赖清单 |
| `models/base_reference/` | 原基础模型配置、tokenizer 和权重索引；不包含权重分片 |
| `docs/` | 代码地图、消融说明、环境与复现边界、来源清单 |
| `scripts/`、`tools/` | 独立便携入口和不占 GPU 的仓库/数据检查 |

## 快速检查（不启动训练）

```bash
python tools/audit_repository.py
```

检查数据字节级 SHA-256、核心源文件、主代码语法、文件大小及常见密钥模式。此检查不证明模型训练结果或所有历史消融已复现。

## 环境

当前正式环境是 Python **3.13.5**、PyTorch **2.6.0**、Transformers **5.5.0**、Lightning **2.6.1**、PEFT **0.18.1**。推荐先使用已有兼容 CUDA 的 PyTorch 环境，再安装核心依赖：

```bash
conda create -n trace-v9 python=3.13.5 pip
conda activate trace-v9
python -m pip install -r requirements.txt
bash scripts/test_cpu.sh
```

完整环境重建记录见 [环境说明](docs/ENVIRONMENT.md)。原项目的旧 `requirements.txt` 含失效的本地 wheel 路径，不能代表当前环境。

## 原 native-v9 训练入口（需要自行分配四张空闲 GPU）

```bash
export TRACE_MODEL_PATH=/absolute/path/to/the/original/base-model
export PHYSICAL_GPUS=0,1,2,3
bash scripts/run_native_v9.sh --dry-run
# 确认路径、基础权重与 GPU 分配后才执行：
bash scripts/run_native_v9.sh /absolute/path/to/a/new/run
```

上述 native-v9 便携入口只替换机器路径，不调整模型、损失、数据、batch 或训练阶段。它不会自动占卡或终止其他进程。**不要把当前正在训练的 run 目录作为新实验输出目录。**

无依赖图新方法的入口另见 [独立运行说明](versions/next_structured_method/code/README.md)，不要使用上面的原 v9 命令启动新方法。新方法只读取普通 `gsm8k_*_processed.jsonl`，不读取 `readcot_qsa_qwen_dc`。

## 重要边界

- v9：Stage0 text-CoT SFT → Stage1 role formation → Stage2 joint RL；最终严格验证为同一验证集的五次单卡、747 个唯一问题审计。
- 四卡训练内验证因 sampler padding 包含 748 条采样记录，不等同于最终严格验证。
- 现存专门消融队列主要属于旧版本；[消融索引](docs/ABLATIONS.md) 标明归属，不能把它们当作 v9 已完成消融。
- 所有历史启动器保留原有路径及调度行为，可能涉及 tmux、占卡或进程信号；仅作源码归档，未经审查不要运行。尤其不要使用 `protect_gpu_lease.sh`。
- 未上传基础模型权重、训练 checkpoint、优化器状态、缓存、运行日志、账号凭据或当前巡检记录。因此本仓库不是可单独恢复当前运行的完整 checkpoint 备份。
- 原有 Apache-2.0 文件及模型许可保留。第三方数据不因仓库私密而改变其原有许可；公开发布或再分发前需分别复核其来源与许可。

更多内容见 [代码地图](docs/CODE_MAP.md)、[数据说明](docs/DATASETS.md)、[复现说明](docs/REPRODUCIBILITY.md)。
