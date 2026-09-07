# 独立代码入口（无依赖图）

代码不导入 main/native_v9，不管理 GPU、不发送进程信号、不启动巡检。使用已有兼容的 torch / transformers / peft 环境，不需要 pytest。当前核验环境为 Python 3.13.5、torch 2.6.0、transformers 5.5.0、peft 0.18.1；正式 GPU 集成尚未执行。

以下命令从仓库根目录执行：

~~~bash
export PYTHONDONTWRITEBYTECODE=1
python -B versions/next_structured_method/code/run.py audit-data
bash versions/next_structured_method/code/scripts/test_cpu.sh

# 仅打印流水线，不加载模型、不启动训练：
python -B versions/next_structured_method/code/run.py pipeline \
  --model-path /absolute/path/to/local/base-model \
  --output /absolute/path/to/new-graph-free-run --workers 4 --dry-run
~~~

测试脚本强制 CUDA_VISIBLE_DEVICES 为空；可用 TRACE_PYTHON 指定解释器。测试含随机 tiny-Qwen + LoRA 三阶段、受控中断恢复及双 CPU/Gloo 进程，所有产物位于自动清理的系统临时目录，不使用正式数据训练或占用显卡。

## 实际启动（需要另行确认资源）

~~~bash
# GPU 编号仅为示例，必须由操作者先确认可用；入口不会自行占卡。
CUDA_VISIBLE_DEVICES=0,1,2,3 python -B versions/next_structured_method/code/run.py pipeline \
  --model-path /absolute/path/to/local/base-model \
  --output /absolute/path/to/new-graph-free-run --workers 4
~~~

使用本地完整权重，禁止远程模型代码，仓库 models/base_reference 只有元数据不能用于训练。pipeline 顺序运行新 Stage0 → 单设备教师缓存 → 新 Stage1 → 新 Stage2 → 单设备五次严格验证 → 全谱系审计。Stage1 / Stage2 的父模型是上一阶段完成后的选优检查点；不加载原 v9 断点。

默认数据根为仓库 data/GSM8k-Aug-NL，仅读取其 gsm8k_*_processed.jsonl；不是该路径下的 readcot_qsa_qwen_dc 子目录。train/val/test 固定数量 6726/747/1319，SHA-256 内置校验，不允许图字段混入。不会改写数据。

所有配置默认值见 [config.py](src/trace_structured/config.py)。可传 --config /absolute/path/to/candidate.json，文件为部分配置字段的 JSON 对象，未指定项使用默认值，未知字段报错；变更任何配置都视为新实验身份，不能忽略身份继续恢复旧运行。

## 分阶段和恢复

~~~bash
python -B versions/next_structured_method/code/run.py train stage0 \
  --model-path /absolute/path/to/local/base-model --output /absolute/path/to/new-run/stage0

python -B versions/next_structured_method/code/run.py cache \
  --model-path /absolute/path/to/local/base-model --parent /absolute/path/to/new-run/stage0/last.ckpt \
  --output /absolute/path/to/new-run/teacher_cache

python -B versions/next_structured_method/code/run.py train stage1 \
  --model-path /absolute/path/to/local/base-model --parent /absolute/path/to/new-run/stage0/last.ckpt \
  --cache /absolute/path/to/new-run/teacher_cache --output /absolute/path/to/new-run/stage1

python -B versions/next_structured_method/code/run.py train stage2 \
  --model-path /absolute/path/to/local/base-model --parent /absolute/path/to/new-run/stage1/best.ckpt \
  --cache /absolute/path/to/new-run/teacher_cache --output /absolute/path/to/new-run/stage2

# 同一目录、配置、父检查点、缓存和 world size：
python -B versions/next_structured_method/code/run.py train stage2 \
  --model-path /absolute/path/to/local/base-model --parent /absolute/path/to/new-run/stage1/best.ckpt \
  --cache /absolute/path/to/new-run/teacher_cache --output /absolute/path/to/new-run/stage2 \
  --resume /absolute/path/to/new-run/stage2/last.ckpt
~~~

以上分阶段示例是单进程。多 GPU 请在 run.py 前使用 python -m torch.distributed.run --standalone --nproc_per_node=N，并设置已分配设备；不能以单进程命令精确恢复 N 进程 checkpoint。pipeline 会为已有 last.ckpt 加上显式恢复参数，已完成阶段只核对/补写汇总，不继续训练。

--max-updates N 只用于显式 smoke 暂停，保存未完成 checkpoint，不会把 N 步标为整阶段完成。--gradient-audit-every N 控制专属 head 梯度探针频率，0 关闭；常规 loss / 总梯度日志仍保留。

缓存构建使用新的空目录且只在完成时写 manifest；若缓存生成中断并留下部分文件，应保留该目录排查，选择另一个新输出目录重新构建。不会默默覆盖或跳过未完成缓存。

## 评测和产物

~~~bash
python -B versions/next_structured_method/code/run.py evaluate \
  --model-path /absolute/path/to/local/base-model --checkpoint /absolute/path/to/new-run/stage2/best.ckpt \
  --output /absolute/path/to/new-run/strict_validation

python -B versions/next_structured_method/code/run.py verify-lineage --output /absolute/path/to/new-run
~~~

每个训练目录含 run_identity.json、metrics.jsonl、last.ckpt、training_summary.json；Stage1/2 另含 best.ckpt 与逐 epoch 验证记录。输出应放仓库之外，不提交权重、缓存、日志和逐题产物。独立代码的所有 Python 文件内容哈希、实际配置、依赖版本及精度进入检查点身份；代码升级后不是无条件兼容续训。

审查与实验边界见 [REVIEW.md](../REVIEW.md)；五次验证为确定性重测，不提供独立样本置信区间。
