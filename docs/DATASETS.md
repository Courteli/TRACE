# 数据集与字节级校验

仓库中包含实际数据，不是仅提供下载链接。逐文件大小、样本数、源路径与 SHA-256 见 `data/manifest.json`。

| 路径 | 用途 |
| --- | --- |
| `data/GSM8k-Aug-NL/readcot_qsa_qwen_dc/{train,val,test}.json` | native-v9 正式 QSA：6726 / 747 / 1319 条 |
| `data/GSM8k-Aug-NL/gsm8k_*_processed.jsonl` | 无依赖图新方法使用的普通 CoT 固定划分；也是已有旧 JSONL 接口的数据 |
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

## 无依赖图新方法的数据契约

2026-09-08 的 `versions/next_structured_method` 直接使用已有普通文件，只接受 `id/question/cot/answer` 四字段，显式拒绝图目录和 dependency/confidence 等额外字段。没有重写任何历史数据；原 v9 仍可读取其 DC 格式。

| 普通文件 | 条数 | SHA-256 |
| --- | --- | --- |
| `gsm8k_train_processed.jsonl` | 6726 | `31e256348cb35ef34bb63c66339a2b8483be44896547c2644d31c0d92e56540c` |
| `gsm8k_val_processed.jsonl` | 747 | `c9ef2ef23b44ea661577e5eb02456738b02a133d70a8b29e57adf86342da0b4f` |
| `gsm8k_test_processed.jsonl` | 1319 | `5395be51d54d7af531883af873e130d3f280a7e5d9aaf849e7c20ed356e3847b` |

已核对普通文件与 DC 三个 split 的题目集合一致，split 内无重复且 split 间不交叉。序列化不同，因此新版本必须校验上表普通 JSONL 哈希，不能套用原 v9 的 DC 验证哈希。新版本训练内验证使用无 padding 分片，正式严格协议仍为五次单设备 747 unique，测试集不参与模型选择。

只归档 TRACE 配置和历史消融实际涉及的数据；未复制 RoT 工作区中无关的 GPQA、MATH 等数据目录。未上传模型 hidden-state/cache 文件。第三方数据的原有权利与许可不因私密归档而改变，公开前另行审查。
