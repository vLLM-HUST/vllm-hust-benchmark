# 环境与复现入口

## 新功能运行（2026-11-15）

- DiffSpec：固定 `48b182803f2b1bc397d61fd8f5a50c8716f2c4ed`；公开源码清单见 `functional/pr11-source-manifest.json`。
- Core：`762f85b311fbab0bcf8921dd216f5093cd58b9b8` 加已有 KV 分组补丁；Ascend：`4e57439e58ed3d78e675f9fd7b4614fb183c5394` 加已有 BF16/GDN/平面对齐补丁。实际 HEAD、改动清单和 20 个文件摘要见 `functional/backend-identity.json`，全部与冻结实测一致。
- Python 3.12.13，Torch 2.10.0+cpu 与 torch_npu 2.10.0.post4；四张实际 Ascend910B2，runtime indices 0–3。设备名与 UUID 来自真实运行 `preflight.log`，不是文件名推断。
- target：`Qwen/Qwen3.5-35B-A3B@59d61f3ce65a6d9863b86d2e96597125219dc754`；draft：`jiapingW/Qwen3.5-35B-A3B-Eagle3-Specforge@d07a277ececba408f017a0416002d7abb23caf36`。沿用已固定的 BF16 模型视图及历史 inference-file 校验，草稿保留自身输出头和 d2t，辅助层 `[4,24,36]`。
- TP4/draftTP4、PP1、K4、PIECEWISE；APC/async 关闭；budget 2048、chunk 64、interval 4、adaptive=false。
- 功能运行 max_model_len=8192、max_num_seqs=2、capture sizes `[4,8,12]`、memory utilization 0.75；两请求分别 4096/4352 输入、256 输出，temperature=0、seed=17、ignore_eos=true。与历史 B4 性能配置分开。

`validation_launcher.py` 记录实际命令、环境和退出；`model_smoke.py` 读取既有固定输入，`trace_worker.py` 只承担观测和明确标记的 dummy 测试构造。源代码及命令随每个 attempt 保存，历史输入与模型视图需按固定版本恢复。对应用的 Core/Ascend 本地补丁见历史冻结包的 `runtime-dependencies/`。

## 历史性能

旧报告、旧源码及原始附件保持原字节；旧报告中的绝对路径是当时身份。迁移后的材料集中于实验归档，不用当前路径或包版本替代当时源码摘要。旧硬件批次与新功能运行分开记录，不做跨批次性能比较。

本次只新增指定功能回归；完整 HTTP serving、长输出无损和标准准确率验收没有由该功能材料替代。
