# BigCodeBench 原始任务生成阶段验证

## 已完成范围

2026-10-23 在现有双卡 Ascend 910B2、Qwen3.5-27B、TP=2、BF16 服务上， 预选原始 v0.1.4 文件前三条 BigCodeBench/0、/1、/2，使用
Instruct 视图生成。 与先前读取检查是同三条任务，不累计为六个独立任务。

| 原始任务       | prompt tokens | completion tokens | 终止 | 全输入预检匹配 |
| -------------- | ------------: | ----------------: | ---- | -------------- |
| BigCodeBench/0 |           150 |               287 | stop | 是             |
| BigCodeBench/1 |           150 |               226 | stop | 是             |
| BigCodeBench/2 |           130 |               348 | stop | 是             |

原数据固定于
[b74c0d0b 版本](https://huggingface.co/datasets/bigcode/bigcodebench/tree/b74c0d0bf70d2c0bc459be537895cca163007f1a)，
文件为 data/v0.1.4-00000-of-00001.parquet，1140条；按来源目录核验摘要。 未缩短或改写原任务，参考实现与测试代码未送入模型。

## 原版协议与差异

固定发布方
[09dd993f 代码](https://github.com/bigcode-project/bigcodebench/tree/09dd993f46c3fbf3a799465bb96d524edcb0b199)，
直接加载未修改的 provider/utility.py 中 make_raw_chat_prompt， 以及 gen/util/openai_request.py 中
make_request；执行前核验原文件摘要。 采用发布方 run_codegen 默认 instruction/response prefix 与1280输出预算，
greedy为temperature=0、n=1，请求函数保留其top_p=0.95。 首尾strip是发布方函数行为，磁盘原始prompt不变。

本次不是完整官方CLI/参考环境复现：模型替换为已有本机NPU服务，服务默认关闭thinking。
未使用发布方无限重试循环，而调用单次请求函数，客户端timeout=300、max_retries=0。 未调用sanitize、代码执行或评分函数，也没有观测/采集钩子。
请求和输出仅保留于服务器私有目录，公开记录不含生成代码或答案。

## 独立生成入口

`scripts/run_bigcodebench_original_generation.py` 只执行生成阶段，原始数据与发布方代码 由使用者在私有目录取得。`--official-code`
应包含上述固定版本的 bigcodebench/provider/utility.py 和 bigcodebench/gen/util/openai_request.py；
入口校验其字节摘要，不修改或分发第三方代码。

本次独立客户端环境为 Python3.12、openai3.1.0、transformers5.5.4、pyarrow19.0.1。
这些是本次探测版本，不是发布方完整官方环境；没有加载CPU模型或CUDA推理。

```bash
python -m pip install openai==3.1.0 transformers==5.5.4 pyarrow==19.0.1
python scripts/run_bigcodebench_original_generation.py \
  --data-root /private/bigcodebench \
  --official-code /private/bigcodebench-code \
  --output /private/bigcodebench-generation-new \
  --model workload-qwen35-27b-128k-nonthinking \
  --origin http://127.0.0.1:19466 --context-capacity 131072
```

原始数据根目录下须有 data/v0.1.4-00000-of-00001.parquet。输出目录必须不存在， 且不能与原始数据或代码目录互相包含。只允许无凭据的本机HTTP服务，不自动下载、
换模型、缩短题目或执行生成代码。完整输入先经服务tokenize预检，随后核对实际usage。 length单独标为截断；空输出、请求失败或预检不符保留失败类型。
只在三条生成均正常stop时返回0，但返回0仍不代表功能正确、评分或全套完成。 原始请求和响应留在权限受限的私有目录，不在终端打印代码或答案。

2026-10-25入口扩展：默认仍选原始0、1、2；可添加 `--indices 5 8 11`、 `--start 3 --count 3` 或
`--full`（1140题）选择模式。显式ID须有序、唯一、至少三条， 不得与范围/全量模式混用；私有结果文件名、回执和准备后的答案均保留原始位置及任务ID。
准备入口自动复用生成回执中的选择，并保留完整1140条原始数据供官方 `BIGCODEBENCH_OVERRIDE_PATH` 使用。所有模式仍采用原版Instruct提示、1280预算、
greedy设置，不因失败换题或重写输出。退出0要求全部选定题正常stop，而非仅默认三题。

这些通用选择模式现有CPU夹具测试，尚无新服务器运行证据；上文2026-10-23的回执仍指向 当时固定前三题及旧脚本摘要，未改写为新版本已验证。后续功能评分仍须执行原版sanitize、
原始测试和官方评分，生成及准备成功均不足以勾选完成。

## 检查清单

### 2026-10-25 后续评分准备（尚未完成）

已在服务器获取上述固定 `09dd993f` 版本的完整官方源码归档，并解包至独立私有目录； 未重新生成已有三题，也未执行其候选代码。官方 `Docker/Evaluate.Dockerfile`
使用 Python 3.10，并安装独立的 `Requirements/requirements-eval.txt`。其构建步骤还引用动态 main
分支的依赖文件，复现时应使用本次固定源码中的文件，记录最终实际依赖，不能声称 仅下载源码即已复现官方环境。

当前待办：核验官方 sanitize、原始测试和本地评分入口，准备隔离环境及其依赖，复用 已保存三题输出执行原生功能评分，再核验公共入口及研究侧消费。此前 HumanEval、 DS1000
的隔离验证不能替代 BigCodeBench 的依赖和运行验证；本项仍不打勾。

可先运行静态预检（只读取原始数据和源码，绝不导入官方模块或执行候选代码）：

```bash
python scripts/audit_bigcodebench_scoring_assets.py \
  --official-code /private/bigcodebench-code \
  --data-root /private/bigcodebench
```

该入口复用生成阶段的 PyArrow 数据读取依赖，核对固定源码摘要、原始 Parquet 摘要和 完整1140条原始ID顺序，列出固定前三题参考实现及测试的静态导入。动态导入、运行时资源、
sanitize、参考实现检查和候选代码评分均不在其证明范围内。当前只通过软件夹具测试； 上传执行时 SSH 32048 返回连接拒绝，尚无服务器执行本入口的回执，不能计为原生验证。

源码校验覆盖固定归档中全部29个Python模块及评分依赖清单，而非只检查主评分文件； 缺失、内容变化、根目录外符号链接或额外Python模块均拒绝。应指向原始解包源码目录，
不将安装过程中新增版本模块的目录冒充原始源码。摘要是本次固定官方公开归档的观察值， 不等同于发布方签名，也不证明第三方依赖或执行环境已经安全、兼容。

已准备原始答案后，可生成官方隔离内命令计划：

```bash
python scripts/prepare_bigcodebench_scoring_plan.py \
  --prepared-dir /private/bigcodebench-score-input \
  --official-code /private/bigcodebench-code
```

此入口不执行命令。它先核验准备回执、完整原始ID顺序及答案文件摘要，再列出官方 sanitize和本地evaluate命令。保留calibrated、参考实现检查、pass@1、原始选题与作者默认
worker选择；使用独立输出缓存，避免沿用其他选题的groundtruth缓存。选定参考实现必须 全部通过，才能接受候选评分；缺依赖、运行环境失败不能当成模型答错。官方默认地址空间/
数据限制为30GiB、栈限制10MiB，不能照搬DS1000隔离器的2GiB硬限制。仍须另行完成
隔离launcher的实际验证、依赖及并发预检、实际sanitize/参考实现/候选评分和完整结果覆盖检查。

### 隔离容器执行入口（软件实现，尚未实跑）

固定源码中`evaluate`还会顶层导入`gradio_client`、`e2b`和生成模块，`sanitize`依赖`tree_sitter_python`等CLI库；不能只安装`requirements-eval.txt`就视为完整环境。容器worker现先在已核验隔离内执行这两个真实官方模块的导入探测，再执行原sanitize及评分；失败按环境缺口保留，不记作模型答错。该探测不带题目参数，不执行候选代码；软件夹具通过不等于真实镜像已通过。

2026-10-26已实际执行下述源码`prepare`及独立`verify`，32份公开源码文件校验通过；仅约110KB源码保存于忽略目录，无数据集、模型、企业载荷或源码执行。服务器恢复后仍须验证原数据准备、镜像依赖和隔离执行，不能把源码获取验证升级为原生评分。

固定评分源码现在可复用已有发布方文件获取入口，不依赖Git clone，也不下载数据或模型：

```bash
python scripts/prepare_jsonschemabench_author_source.py --source bigcodebench \
  --author-root /private/bigcodebench-fixed-source --action plan
python scripts/prepare_jsonschemabench_author_source.py --source bigcodebench \
  --author-root /private/bigcodebench-fixed-source --action prepare --max-bytes 109939
python scripts/prepare_jsonschemabench_author_source.py --source bigcodebench \
  --author-root /private/bigcodebench-fixed-source --action verify
```

脚本名称为历史兼容入口；不传`--source`仍准备JSONSchemaBench。BigCodeBench选项固定32份文件共109939字节（29个Python模块、两个依赖文件、官方Dockerfile），逐文件核验发布方Git
blob、字节数和SHA256。已有正确文件直接复用，错误文件或遗留partial拒绝覆盖；这里只取得评分源码子集，不声称取得整个Git仓库、可构建镜像或完成原生执行。

固定作者版本的`Docker/Evaluate.Dockerfile`仍包含可变的`python:3.10-slim`、最新Git
clone、main分支评分依赖URL、未固定版本的pip安装及构建时full/hard数据下载；不能直接把该配方构建结果当成本次固定版本环境。静态预检现同时校验官方Dockerfile和CLI依赖文件，并输出`environment_preparation`中的未冻结输入及所需构建记录。
准备评分环境时，应保留作者Python3.10、系统依赖和评分库要求，但将代码/评分依赖指向本次已核验版本，数据使用本次私有原始工件；记录基础镜像digest/架构、OS与Python包实际版本和最终不可变镜像ID。此处尚无已构建、依赖检查通过的镜像，不提供虚构镜像ID，也不将固定配方校验算成运行成功。

`scripts/run_bigcodebench_container_scoring.py`可在Linux上调用本机Docker/Podman，
复用准备后的原始输出，不再请求模型。操作者须先取得并核验本机评分镜像的不可变 `sha256:`镜像ID，设置`SCORING_IMAGE_ID`；本入口不拉取或构建镜像，不用可变tag。
镜像须提供官方Docker约定的Python3.10及`/usr/local/bin/python3`、完整官方评分依赖
和CLI导入依赖；容器内逐项核验固定requirements版本及`pip check`，不满足就失败。

```bash
python scripts/run_bigcodebench_container_scoring.py \
  --engine docker --image "$SCORING_IMAGE_ID" \
  --prepared-dir /private/bigcodebench-score-input \
  --official-code /private/bigcodebench-code \
  --work-root /private/fresh-bigcodebench-container-run \
  --sandbox-uid 65534
```

启动器要求已授权Linux root进行私有目录分配；候选代码只在非root容器中执行。
禁网、只读根文件系统/输入/官方源码、清空capabilities、no-new-privileges，不映射NPU、 宿主PID或Docker
socket；只允许新建输出目录可写。总内存32GiB、禁止swap、进程数1024， 并保留作者30GiB地址空间/数据和10MiB栈限制，不覆盖官方worker选择；资源不足不能算完成。
执行前比较宿主与容器mount/net/PID namespace身份，并再次核对暂存原始文件和源码摘要。
官方sanitize、reference检查及candidate评分按原流程运行，reference必须全部通过，
sanitize结束后、调用evaluate之前先核对原选题ID和顺序；丢题、重排、重复或非对象记录立即失败，不继续执行功能测试。合法空答案不替换，仍交官方评分。
且结果覆盖全部选定ID、每题一个原始输出、pass@1与逐题状态一致，才产生原生评分回执。

原代码、评分细节、日志及容器句柄均保留于私有工作目录；失败不自动重启、删除容器或 回退宿主执行。`launch-private.json`记录容器名称和终态；未结束时应按该名称inspect，
不能因观察超时重开。该隔离是受限本地评测环境，不是对任意恶意代码的安全认证。 当前仅有命令、拒绝条件和结果验收的软件夹具证据；服务器不可达，尚无真实容器运行、
依赖兼容、并发容量或研究侧消费证明。原生评分回执本身仍不自动勾选最终交付。

2026-10-24历史服务器的[隔离预检](validation/code-execution-isolation-preflight-20261024.json)：网络namespace可创建，但mount/PID隔离和私有bubblewrap无害探测失败，且无Docker/Podman/socket。该历史结论不代表2026-10-25已验证的其他来源隔离环境；BigCodeBench自身功能评分仍未运行。保存的生成结果继续复用，待其隔离环境与依赖验证后接原sanitize/功能测试及评分，不裸执行、不重生成刷分。

- [x] 三个原始任务读取、prepare、resolve输入/参考独立往返。
- [x] 原版提示/请求函数驱动NPU生成；三条完整输入预检匹配、正常stop。
- [x] 可复用公开生成入口、CPU协议测试及服务器同三题交接复跑；结果一致，不累计独立任务数。
- [ ] 原版功能测试与适用pass@1评分。
- [ ] 保留完整1140条原始数据，完成固定抽样的完整原生评分、本批最终工件及研究端交接；不要求凭抽样宣称1140条全部实跑。

交付范围按统一规则区分：完整原数据的字段、ID和索引必须保留；运行采用事前固定的原始样本（当前前三题），每题均需通过官方sanitize、参考实现环境检查及候选功能评分。样本内不得删减流程、换失败题或把缺依赖算作模型答错。小协议预计执行及评分不超过30分钟则全跑，否则固定至少三条；完整1140条质量测评另记，不与本批抽样可用性验证混淆。评分计划的`validation_scope`明确记录数据总量、选题数和是否全选。

当前未找到Docker/Podman或Docker socket，已有HumanEval隔离探测也未通过。 这不证明每种沙箱都不可行，但不能绕过隔离裸执行生成代码。
必须记为生成阶段完成、功能评分未运行，不能据HTTP成功或stop宣称代码正确、 官方pass@1、完整任务验证或性能提升。
