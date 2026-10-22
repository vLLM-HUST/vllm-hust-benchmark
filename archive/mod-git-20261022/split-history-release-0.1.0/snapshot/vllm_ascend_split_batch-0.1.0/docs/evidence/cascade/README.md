# Cascade 证据（随插件仓发布）

本目录是 **cascade 线的证据副本**，让外部读者不必访问开发工作区即可核对数字。
权威原件在开发工作区的 `knowledge/evidence/cascade/`，本目录是**裁剪 + 遮蔽**后的
快照（入仓日期 2026-09-26；同一批文件已随 `main` 的合并提交公开）。

## 1. 读法（先读这一节再看数字）

| 项 | 值 |
|---|---|
| 新基线 | vllm-hust v1 `0.28.1.post1.dev143+gf18cf803c`；vllm-ascend-hust `0.25.1rc2.dev125+hust.20260903.4.g74f0c0a27`；torch 2.13.0+cpu / torch_npu 2.13.0rc1；CANN 9.1.0；kernel wheel `ascend-kernel==2026.3.9` |
| 硬件 | 910B2（单卡；2×910B2 机器，卡 6/7） |
| 旧基线留档 | `EVIDENCE.md` §8 是 0.23.0rc1 / CANN 9.0.1 的记录，**不继承** |
| 模型口径 | ⚠️ **`Qwen2.5-Coder-14B-Instruct` 是性能替身**（几何同 14B，但短文 prompt 3–5 token 即 EOS）；行为类/逐 token 比对类证据一律用 `Qwen2.5-14B-Instruct`。**两者不可跨模型比较** |
| 当前状态 | 两个 cascade carrier 为 `active`（`section4-active-enablement.md` 为准）；`planner:plan_dual_pad` 仍 `import_only`（无验收证据） |
| 已知边界 | 每请求 1 个 query row；投机解码（含 MTP）与多 query 批次 fail-closed 回原生路径 ⇒ 那些形态**没有** cascade 收益（见根 README「Speculative decoding boundary」） |

## 2. 文件与角色

| 文件 | 内容 |
|---|---|
| `EVIDENCE.md` | 总入口：三项证据结论速览、缺口处置、旧基线留档 §8 |
| `section2-correctness.md` | 证据②（正确性对齐，含真模型 like-for-like 与发散臂统计） |
| `section3-performance.md` | 证据③（TPOT 对比、自适应 gate 的 9 格判定与 G6 修复） |
| `section4-active-enablement.md` | 翻 `active` 的 §3 启用验证（`inspect`/`check`/`dry-run`/真模型 serve） |
| `fix-sigdrift-report.md` | D1–D4（漂移口径）宿主签名漂移的定位与修复 |
| `ev2_*.sh` / `ev2_*.py` / `run_evidence1_default_off.sh` | 复现脚本（表格里的数字由它们产出） |
| `logs/**/results/*.json` | 每腿原始结果（生成的 token id 与逐步耗时），数字的唯一来源 |
| `logs/**/*.txt` | 控制台小结（版本、环境、对比原始输出） |
| `logs/serve-*.log`、`logs/v1-active-enablement/`、`logs/v1-real-model/`、`logs/fix-sigdrift/`、`logs/sig-audit/` | 服务启动/门控/回归的原始日志 |
| `MANIFEST.sha256` | 上面每个文件的 sha256（**针对入仓后的字节**）+ 未入仓原始件的 sha256 与字节数 |

## 3. 入仓边界（可核对，不隐瞒）

- **入仓 147 个文件 / 5.77 MB**：`*.md`、脚本（`*.sh`/`*.py`）、`*.txt`、`results/*.json`、
  顶层 `serve-*.log` 与四组小结日志（`fix-sigdrift` / `v1-active-enablement` /
  `v1-real-model` / `sig-audit`）。
- **未入仓 209 个文件**：体量最大的逐步 sweep 日志（`logs/v1-ev3*/`，合计约 18 MB），
  按体积裁剪；它们可由本目录脚本重放，且已排除项清单在 `MANIFEST.sha256`
  末尾逐条列出（路径 + 字节数 + sha256，可对照工作区原件校验）。
- **遮蔽 1 类 3 处**：私网地址（torch distributed init 打印的
  `distributed_init_method=tcp://<address>:<port>`）已改为 `192.168.x.x`，见上表 3 个 `serve-*.log`。
  其余端口/回环地址（`127.0.0.1`、`0.0.0.0`）为服务绑定与健康检查，原样保留。
  （同一类遮蔽也应用于 `docs/evidence/w2b-fi-sampling/logs/` 下的 10 个日志，2026-09-26 首发前复核。）
- **未纳入的内容**：模型权重与语料（只引用路径）、kernel wheel 二进制（在 kernel 仓构建）、
  宿主源码（本仓不含，按契约只走公开面）。

## 4. 引用注意

- 报告正文里的路径按**当时的开发环境**书写（`/vllm-workspace/…`、`/data/shared_models/…`）；
  在仓内请以本目录的相对路径为准。
- 被裁剪的原始件若被正文引用，其承接点在 `MANIFEST.sha256` 的排除清单，不影响任何已发布的
  数字（数字来自 `results/*.json`，已全部入仓）。
