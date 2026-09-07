# 数据集与字节级校验

仓库中包含实际数据，不是仅提供下载链接。逐文件大小、样本数、源路径与 SHA-256 见 `data/manifest.json`。

| 路径 | 用途 |
| --- | --- |
| `data/GSM8k-Aug-NL/readcot_qsa_qwen_dc/{train,val,test}.json` | native-v9 正式 QSA：6726 / 747 / 1319 条 |
| `data/GSM8k-Aug-NL/gsm8k_*_processed.jsonl` | 旧 JSONL 数据接口的固定划分 |
| `data/GSM8k-Hard/` | Hard 测试 JSONL 与 QSA 格式 |
| `data/SVAMP/` | SVAMP 测试 JSONL 与 QSA 格式 |
| `data/Multiarith/` | MultiArith 处理后划分与 QSA 测试格式 |

正式 v9 验证文件必须保持下列 SHA-256：

```text
5f2ddd39f09f95d834a1b2840ce12d9fd2461bc909a755f1b8c429497bdfa49b
```

`python tools/audit_repository.py` 会验证数据文件与清单一致。不要重排、重新切分或重新序列化正式 JSON 来代替原文件。

四卡训练内验证有 sampler padding（748 条采样）；最终协议要求五次单卡、每次 747 个唯一验证问题。这里的测试文件和 OOD 数据被保存，不意味着它们被用来选择 v9 的最佳验证模型。

历史预处理脚本位于 `ablations/legacy_trace_colar/data_preprocessing/`，部分脚本使用旧路径或不同划分，不能据此声称可以逐字节重建当前 dependency/confidence 注释。当前已处理文件本身及其哈希是复现依据。

只归档 TRACE 配置和历史消融实际涉及的数据；未复制 RoT 工作区中无关的 GPQA、MATH 等数据目录。未上传模型 hidden-state/cache 文件。第三方数据的原有权利与许可不因私密归档而改变，公开前另行审查。
