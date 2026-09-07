# 复现范围与边界

## 基础模型

原始模型本地目录标为 `Qwen3-4B-Instruct/qwen3-instruct`。`models/base_reference/` 保留其实际 config、tokenizer、generation_config、许可和 safetensors index；该目录不含约 8 GB 权重分片，不能独立加载为模型。

不要仅凭本地目录名字把它替换成某个最新 Hugging Face 模型。当前没有足够来源信息确认其唯一的远端仓库 revision。使用原权重副本，或在取得可靠来源后逐项核对 `models/base_reference/manifest.json`；设置 `TRACE_MODEL_PATH` 指向完整基础模型目录。

## 当前方法与原运行

根入口保留 Stage0 3 epochs、Stage1 10 epochs、Stage2 10 epochs、seed 1701、四卡 global batch 4、Stage2 group size 8 及原损失配置。正在运行的历史实例中 Stage1 曾被提前停止并选择 epoch2 checkpoint；从头执行默认完整 pipeline 不会自动重现这次人工停止。这一差异必须在复现实验记录中明确。

续训依赖同 run 的原始 checkpoint、优化器/调度器/loop 状态和 hparams。它们没有上传；仅克隆源码不能恢复当前进程，也不能保证 mid-epoch 非 stateful dataloader 的精确逐样本重放。

## 路径可移植性

副本中的主 pipeline 使用动态 SOURCE_ROOT、`TRACE_PYTHON`、`TRACE_MODEL_PATH`、`TRACE_DATA_ROOT` 和新 RUN_ROOT。主配置中当前 Qwen3 模型与数据路径改为环境变量；其余算法数值保持原样。

运行主代码或 CPU 测试应从 `main/native_v9/` 执行；根目录脚本会代为设置工作目录。`archive/`、历史消融、旧运维脚本仍保留原始机器路径，不承诺一键迁移。

## 验证层级

仓库审计、语法检查、CPU 单元测试只验证打包完整性及其覆盖的逻辑。它们不等于在新环境完成三阶段训练、最终严格验证、OOD 评测或 v9 消融。本次上传没有启动新训练，也没有暂停或修复主任务。

所有模型/数据结果声明应来自对应版本和对应协议的实际输出，而不是历史论文占位图、运行日志片段或文件名。
