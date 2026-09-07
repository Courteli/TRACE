# 环境记录

来源：现有 `ROT` Conda 环境；本次未安装、升级或卸载该环境中的任何包。

- `requirements-core.txt`：native-v9 text-only 路径的主要依赖，采用实测版本。
- `requirements-lock.txt`：现有环境完整 `pip freeze`，包含与 TRACE 无关的包；用于审计，不等于干净的最小环境。
- `conda-linux-64-explicit.txt`：Python 和 Conda 包的 Linux x86_64 精确记录；不含 pip 包。
- `runtime.json`：Python、PyTorch/CUDA、库版本与 GPU/驱动信息快照。
- `legacy-requirements.txt`：旧项目清单，含不可移植的 `file:///rapids`、`/opt` wheel 路径；仅归档，勿直接安装。

## 安装

```bash
conda create -n trace-v9 python=3.13.5 pip
conda activate trace-v9
python -m pip install -r requirements.txt
bash scripts/test_cpu.sh
```

如需按现有 Linux 环境记录重建，可先用 `conda create -n trace-v9-snapshot --file environment/conda-linux-64-explicit.txt`，再在该环境安装 `environment/requirements-lock.txt`。完整快照可能需要额外构建工具；本次没有创建新环境进行整套 GPU 训练复验，不能保证其他驱动/平台无需调整即可安装。

当前环境没有安装 `flash-attn`、`datasets`、`pytest`、`sentencepiece`；主路径直接读本地 JSON，测试使用 unittest，不把这些包伪装成已验证必需依赖。

现有 `pip check` 报告：`qwen-vl-utils 0.0.14` 缺少可选视频依赖 `av`。这是现有全环境问题，未修改；text-only native-v9 不使用该视频路径。CPU 测试结果与完整环境健康状态应分别理解。
