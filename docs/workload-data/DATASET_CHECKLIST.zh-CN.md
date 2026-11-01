# 数据集逐项验证清单

### SWE-smith 当前补接入记录（2026-11-02）

- [x] 复用已有24条完整作者历史轨迹，保留原始消息、patch与历史结果；共3,083,863字节，服务器回查未新增下载或模型调用。
- [x] 补充原始任务发布版本、11份分片身份及复用获取入口；夹具检查通过，不代表服务器已下载或执行原题。
- [x] 在服务器核对官方源码：原patch引入缺陷，gold模式反向应用，模型修复正向应用；不能直接复用SWE-bench参考答案映射。
- [x] 11份原始分片全部取得并逐份核验官方LFS身份，共277,767,206字节；公共`--verify-only`通过。服务器直连HF失败，使用4MiB分块内存中转，本地未保存原数据。直接下载分支尚未在服务器验证。
- [x] 固定原第一分片前三条。第三条原issue为空，保留、不换题。见[原题准备回执](validation/swesmith-original-first3-preparation-20261102.json)。
- [x] 公共源码`iter_workload`、准备索引和未修改的研究消费端在服务器对原前三条回查一致，七字段完整恢复，缺陷patch和测试划分不进入模型输入，索引未被读取修改；仍不代表wheel或原生评分通过。
- [x] 公共读取器遍历11份原分片全部59,136条任务，逐条七字段合同检查通过；这是读取验证，不是全量Agent执行。
- [x] 绿色CPU
  CI的实际wheel已在专用服务器安装，82项包内工件逐项匹配提交源码；原前三条公共与未修改研究消费端回查一致。构建提交、artifact与哈希见准备回执，不代表原生Agent或评分完成。
- [ ] 原生Agent与官方评分执行，以及原生结果的研究侧接通验证。

当前专用CPU为ARM64；官方容器创建与评分路径显式请求x86_64。尚未验证兼容执行，不能仅凭profile支持ARM镜像就宣称可跑。见[源码语义审计](validation/swesmith-original-semantics-audit-20261102.json)。

原始任务获取与读取（仅准备，不运行Agent）：

```bash
python scripts/prepare_swesmith_original_assets.py --output /external/swesmith
python scripts/prepare_swesmith_original_assets.py --output /external/swesmith --verify-only
```

读取时以`/external/swesmith/data`为`data_root`（通用获取器在外层增加`data/`目录），显式选择原分片，不需重新构造prompt：

```python
from pathlib import Path
from vllm_hust_workload_data import iter_workload
from vllm_hust_workload_data.swesmith import restore_original_task

for record in iter_workload("SWESMITH", data_root=Path("/external/swesmith/data"),
                            files=["data/train-00000-of-00011.parquet"], limit=3):
    original_task = restore_original_task(record)
    # original_task交给原生执行器；不得把patch或测试划分拼进模型输入。
```

需要安装公共数据包的`data`可选依赖。原生Agent与评分的可运行命令仍待环境验证，不把以上读取命令写成benchmark完成。

历史轨迹不是本次原生实跑，SWESMITH仍未完成、不得打来源完成勾。证据见[部分复用回执](validation/swesmith-author-history-reuse-20261102.json)。原始轨迹及研究分析不提交到公开仓库。

## 完成口径：来源与原生协议分别记录

验证单位是“来源＋明确的官方原生协议”。来源打勾表示其列明的交付协议已可使用并有固定原题实跑证据，不表示该来源全部可选任务、模态或全量模型评测完成。

- 必须提供原始来源及获取方式、明确环境与模型绑定、可执行完整所选流程的公开入口、适用的原评价流程，以及公共输入和未修改研究消费者的接通证据。
- 大数据集按固定规则抽样验证，小数据集在可行时全跑；入口不能仅支持所选几条。完整流程不等于必须对全语料生成，抽样不能改题、截断原生关系或缩减任务步骤。
- 未完成模式单列缺口，保留40类场景覆盖要求。文本DST不计为语音交互，离线动作预测不计为在线浏览器Agent，数据或评分器准备不计为原生任务执行。
- 按已有证据逐项复核，不因口径调整批量补勾；失败和低分如实保留，不用替换样本凑通过。113项来源与另5项仅分类的范围不变。

当前交付
**85/113项**，待完成28项，另5项仅分类。首批#432的60项和第二批#538的ScienceWorld已合入main；早期五个草稿已关闭为被汇总承接，历史和分支保留。第三批来源及精确协议范围以机器清单和各交付回执为准，尚未合入。DailyTalk按原Guo前三段完整对话29条NPU批量合成协议交付，公共准备入口支持其他完整原对话；不代表全量音频、全split生成、MOS或流式时序。ScienceAgentBench按原verified固定前三题单轮生成、NPU辅助评分和隔离科学执行协议交付，三题程序失败如实保留，不代表全102题执行或其他Agent分支。TemporalWiki按原GPT2-large基座探针全部35,948行PPL交付，不代表持续训练checkpoint或其他语料模式。StreamingBench按real固定前三组原视频NPU推理与原评分交付，其他任务／模态缺口单列。LJSpeech按原VITS固定前三条NPU文本到波形协议交付，不代表全13,100条合成或MOS评价。StructEval按19个完整非渲染原生协议收口，渲染/VQA模式缺口继续单列，不称整套44协议或2035题全部完成。OpenMath/OpenCode是训练语料原问题生成，不是独立质量benchmark。打勾不代表全量模型评测、高分或性能保证。

StructEval新增配置进展：经确认，`qwen35-9b-nonthinking-32768`与原`publisher-null`分开记录。仅原Text→TikZ位置37的已知失败题运行新配置，传输timeout3600秒、外层/SDK重试均0；保存响应为length／32768输出tokens。CPU原工作区autostop导致渲染转发失败，900秒等待后原生子进程以RenderBlocked失败退出；原响应、代码和待渲染请求保留，实际VQA调用0，不计完整任务。原80GiB持久卷和模板已原样恢复，Docker仍在恢复旧容器；不重启daemon、不重新生成，不重放位置38的结果未知请求。共享任务根`structeval-capped-failed-tikz-window-32068-001`保留终态，计数不增加。

StructEval恢复增量：原LaTeX镜像12层、972434432字节经只读原层SHA核验后流式导入独立Docker，镜像ID与所有diff_id不变；未重建、未扩容、未重启原Docker。五项原渲染器CPU控制通过。此前保存的32768-token原题输出已实际渲染并回传，原render_score为0、无图像，请求SHA核验一致；未新增模型调用。原控制器失败记录保留，原最终评价续接与逐题审计尚未完成，不增加总计数。

StructEval评分续接：同一保存输出已调用未经修改的原`render_file_async`及`evaluate_file`完成，原render_score为0、final_eval_score为0.08。无图像时按原规则跳过实际VQA请求；通过禁止新增模型调用的控制核验新增生成/评委请求均0，不把“VQA流程返回”误称为实际评委推理。原推理文件逐字保留，原null结果未动；独立逐题审计待完成，总计数暂不增加。回执`structeval-capped-saved-final-evaluation-20261101-001/qualification-public.json`。

StructEval逐题审计续报：任务001304（全局原行726、协议位置37）的10个原字段、9组VQA、原提示词、响应、原代码解析与渲染请求SHA均核对通过；原评分器独立重算与0.08保存结果完全相同，新增模型调用0。该独立32768配置单列，不与null配置合计质量分。此前未尝试的位置39–49共11题已在研究空闲、现有9B无在途请求后启动；控制器和子进程均核验存活，不能据此计完成。第38题未知结果不重放，已有有效输出不重生成，卡6服务不变。

StructEval新增11题终态：位置39–49的11次原生成与11次原渲染全部返回，原评分流程退出0；独立审计已核对原字段、提示词、响应、代码解析、渲染请求及完整条件VQA。11次评委请求中10次正常返回、1次BadRequestError，原评分器对后者的全None分支与0.3最终分照实保留，未重试或伪装成正常评分。该组所有保存结果的审计不发起新模型请求；[逐题回执](validation/structeval-capped-tikz11-audit-20261101.json)。独立32768配置与原null配置继续分开，来源总勾不变。

上述BadRequest已定位：任务001334原渲染PNG为1433×1、86字节；同一9B服务日志明确报长宽比1433超过处理器上限200。原图片不缩放、不替换，不为凑通过改图或重试；原评分器的失败分支和分数保留。

StructEval非渲染交付复核：按已确认的“来源＋明确原生协议”口径复核，而非要求所有可选模式同时完成。19个非渲染协议各50题、共950题的既有结果已找到；37个保存运行目录的111份receipt/inference/evaluation文件重新核对原审计哈希，缺失0、变化0、新模型调用0。这里的37是运行目录数，不是37个来源。新增公开原数据/官方源码获取入口及独立客户端安装步骤；服务器公开只读验证退出0、27个原源码模块一致，13个直接依赖版本一致，清除客户端继承CANN路径后pip
check退出0。复用了原文件硬链接，没有重复下载或宣称新装环境/冷下载通过。回执`structeval-public-original-assets-verify-20261101-002/qualification-public.json`。原132次固定provider/研究读取证据保留；待本次公开工件与当前未修改研究消费者复核后，再决定协议级交付打勾，渲染模式缺口继续单列。

StructEval非渲染交付收口：本次CI249dff34的tests/lint及CPU
smoke均通过，实际工件13054253582的78份软件载荷与Git、wheel、安装字节一致；当前未修改研究提交9933c1e与公开入口完成全部2035行字段核对、2035条索引及44组协议的132次完整记录回查，输入/参考/注释无丢失，数据库未变，0模型调用。结合已核验的19协议950题原生执行与原评分、111份原保存文件及公开获取/环境说明，按已确认协议级口径打勾。[交付回执](validation/structeval-nonrender-native-delivery-20261101.json)。这不是重跑950题，也不改变其他渲染模式的未完成状态。

### StreamingBench：real原视频协议已交付，其他模式缺口保留

- [x] real原固定前三组完整原题已由官方支持的Qwen2-VL-7B在NPU完成3次生成，保留原0起点及31／206／403秒终点、原fps表达式、151200像素设置与128-token预算；原评分器2/3，保存结果独立重算一致，审计新增模型调用0。首次file://解码失败保留；仅对解码工具做同一文件路径规范化，未改聊天模板消息。当前6f5549b4两项CI通过，实际工件13070701192独立安装78载荷核验，8原文件／4948行索引及24条公共／当前研究完整回读一致，3组与实际原生输入匹配。9B原命令恢复ready，卡6的14B不变，互斥租约已释放。[交付回执](validation/streamingbench-real-native-delivery-20261101.json)

- [ ] 全量real2252组／2500问／499视频尚未模型运行；SQA、proactive、omni/audio及streaming原生模式仍未验证，45份选定音频仍待取得。下列媒体准备与失败记录为历史检查点，不表示当前real协议仍未完成。

StreamingBench媒体准备增量（2026-11-01）：清理后专用CPU实测空闲14141509632字节，原固定10份视频总量2823253471字节，原容量阻塞已解除，不扩卷。原已核验视频不重下载；余9份尝试公开下载入口时首项返回Network
is
unreachable，失败记录保留，现复用原分块获取器经至多8MiB原字节转发到服务器。45份音频及原生模型/预处理/任务/评分仍待验证，不补来源总勾。另核查作者CLI仅分派MiniCPM-V/MiniCPMo，虽提供Qwen2-VL类却未接入该CLI；不能把未知模型回退到基类当成有效推理，后续必须明确接原模型类并披露NPU设备/attention兼容绑定，原题、时间窗、自适应fps和128-token预算不改。

- [x] 原视频新增1份：sample_24完整274336183字节，原ZIP成员CRC/EOF与观测SHA256核验、全23869帧解码通过，保留原音轨；与此前sample_2合计2/10份、289598876字节、32252帧。公开`--verify`及`--download`复用分支均退出0，不联网或重复解码。回执为`/root/streamingbench-public-second-media-reuse-20261101-001/qualification-public.json`。第二份新视频TLS中断后保留部分文件，按原字节比对前缀、只追加缺失后缀继续获取；已完成视频复用，不重新获取，不计原生任务。

- [x] 第三份原视频sample_16已取得150322640字节，原CRC／SHA及全16934帧解码通过，公开验证／既有文件复用CLI各退出0。该检查点累计3/10视频、439921516字节、49186帧，不计任务。回执为`/root/streamingbench-public-third-media-reuse-20261101-001/qualification-public.json`。

- [x] 第四、五份原视频sample_17_proactive及sample_221_real已取得，原ZIP成员CRC/EOF、观测SHA及全部38802帧解码通过。两份公开`--verify`和`--download`复用CLI各退出0，无联网、重复解码或模型调用。累计5/10视频、908128954字节、87988帧；其余5份视频及45份音频未完成，原生执行未完成，不计来源交付。回执分别为`/root/streamingbench-public-fourth-media-reuse-20261101-001/qualification-public.json`和`/root/streamingbench-public-fifth-media-reuse-20261101-001/qualification-public.json`。

- [x] 第六份原视频sample_348_real完整667768200字节、原CRC/EOF与观测SHA及全部58247帧解码通过，保留原音轨；默认real三组原题所需媒体已齐。累计6/10视频、1575897154字节、146235帧；其余4份视频及45份音频未完成，不计原生任务。回执`/root/streamingbench-remaining-ram-member-04-20261101-001/qualification-public.json`。

- [x] 第七份原视频sample_416_real完整335839436字节、原CRC/EOF与观测SHA及全部20783帧解码通过，保留原音轨。累计7/10视频、1911736590字节、167018帧，余3份视频和45份音频未完成，不计原生执行。回执`/root/streamingbench-remaining-ram-member-05-20261101-001/qualification-public.json`。

- [x] 第八、九份原视频sample_17_sqa、sample_32_sqa的原CRC/EOF、观测SHA和全部65201帧解码通过，保留音轨；累计9/10视频、2421684080字节、232219帧。回执为`/root/streamingbench-remaining-ram-member-06-20261101-001/qualification-public.json`及`/root/streamingbench-remaining-ram-member-07-20261101-001/qualification-public.json`。最后一份SSH中断保留192879471字节partial，核查无残留写进程后按原字节前缀核对续传；已完成9份未重新下载或解码。续传仅对幂等初始化/加锁原字节追加的SSH255传输错误至多重试3次，不重试原生模型任务或未知终态解码任务。45份音频和原生模型执行/评分仍未完成，来源不打勾。

- [x] 第十份sample_42_sqa完整原文件已保留并独立复核原ZIP成员CRC266770352、401569391字节、观测SHA及全部34450帧；累计固定视频10/10、2823253471字节、266669帧。传输完成后的校验RPC因SSH中断未启动，核查无存活进程/回执后，复用完整文件做校验，不重新下载；恢复阶段没有重放压缩流，不把原始字节复核冒称为再次验证deflate
  EOF。CPU双线程仅用于完整性解码，不修改模型预处理。回执`/root/streamingbench-remaining-ram-member-08-20261101-001/qualification-public.json`。45份音频和原生推理/评分仍待完成，来源不打勾。

- [x] 明确原Qwen2-VL类的候选NPU入口及分类索引已补，不走作者CLI未知模型回退；服务器固定与全量real计划通过，0模型调用。默认3组共用sample_348原视频，保持3个原问题，不因共享视频去重任务或改用已取得的视频。原31份软件、128-token预算、作者时间窗及评分流程保留；`streamingbench-public-native-plan-20261101-004/qualification-public.json`是软件计划证据，不是实际NPU推理或全量媒体验证。

- [x] 固定real前三组所需的同一份667,768,200字节原视频从既有CPU缓存复用至NPU存储，完整SHA256一致；不重新下载、Windows不落原文件。保留24MiB/120MiB阶段的SSH中断原字节前缀，改用双持久连接完成原字节转发，未重跑CPU全帧校验或模型。

- [x] 原前三组0至31／206／403秒的切片与CPU处理器全部通过，3060／18736／36536输入token；原切片函数、fps表达式、151200像素上限及128生成预算保留。独立核验三份切片的大小/SHA、H264/AAC、1920×1080及原时长，缓存共581,528,836字节，未重编码或重跑模型。独立FFmpeg/x264/AAC及Python绑定已补齐，前两次模块导入和首次缺x264运行库配置失败保留。[准备证据](validation/streamingbench-prefix-preflight-20261101.json)

- [x] 公共候选入口增加严格核验的原前缀缓存复用，走作者原有“文件存在”分支，保留全部原问题和评分，不复用答案；SDPA小规模NPU数值控制通过，但不代表原Qwen整模型或完整长上下文成功。候选默认SDPA，eager仍须显式选择；不减帧、改时间窗或缩预算来适配显存。

- [ ] 原Qwen2-VL完整固定三组NPU推理、原评分及最终工件/研究接入；完成前来源不打总勾。

- [x] 作者FPS解析问题用3个合成文件名的原函数前缀计算复核，`.mp4`数字被当成时间，原分支均1fps；未读视频或调用模型，原公式不改。不以分支常量列表宣称实际自适应fps，修正需另行披露协议变体。真实运行及原评分结果仍待完成，整来源不打勾。

### ToolBench：原软件准备完成，原生任务未完成

- [x] 固定官方版本`d56fdd89faf8c91fa135090b212bb9057ee5cfc2`，服务器取得79份原Python/shell/说明/依赖/许可文件，共397381字节，逐份Git对象与SHA256核验；64份Python仅AST解析通过，未导入执行，未取得原题或权重。
- [x] 复用公共作者获取器`--source toolbench`，只读`--action verify`在服务器通过；回执为共享任务根`toolbench-original-software-32068-001/qualification-public.json`、`toolbench-public-source-entry-32068-001/qualification-public.json`。软件准备不计来源总勾。
- [ ] 原数据包、完整工具交互环境、合适NPU模型/检索器、原ToolEval评分与实际工件/研究消费仍待接通。官方主线要求ToolBench后端key或自己的RapidAPI环境，当前均未核验；不自行编造响应，也不静默改用StableToolBench模拟协议。

### 已完成批次的存储清理

- [x] 专用CPU卷不扩容。OVO的103份CPU重复视频在与NPU副本逐份字节/SHA256核对且无进程打开后删除，保留NPU原文件及所有评分/审计回执。清理记录位于原CPU目录`cleanup-verified-duplicate-media-20261031.json`。
- [x] 删除已退出且安装失败的旧BigCodeBench准备容器，保留安装日志、容器配置、合格镜像和正式结果；仅非强制清理2个无标签且无容器引用的镜像，76个引用冲突保留。CPU可用空间由1090072576增至14216171520字节，累计回收约13.1GB。回执位于`/root/workload-native-cleanup-20261031-001/`。后续逐批清理闲置重复数据/临时环境，不对整个Docker目录或数据根目录递归删除。

### LongMemEval：原生问答与研究输入接通（已交付）

- [x] 三份原Oracle/S/M资产取得并固定身份；原文件不改，公开作者获取、读者及评分入口与环境说明齐备。详见[交付回执](validation/longmemeval-original-native-delivery-20261101.json)。
- [x] Oracle/S各原前三题，6份已有完整历史生成复用；原提示、会话关系、CoN800预算和保存回答审计通过，不重新生成。S输入113452–116259 tokens，未截断或换题。
- [x] M使用作者原BM25-session/no-expansion、10工作进程及top10/JSON/双侧消息/CoN800读者。固定原前三题的482/472/473个会话、5057/4783/4847话轮、重复会话ID及完整字段保留；3次NPU生成正常stop，输入34366/31034/30916
  tokens。全历史1201716-token超限记录保留为另一基线，不冒充M整体不可用。
- [x] Oracle时间推理、S用户信息及M助手信息三种题型，各正例／错误负例／空答案负例通过后，对9份保存回答运行原QA提示、10-token预算、yes子串解析及聚合，独立审计通过。实际judge明确为Qwen3.5-9B，不是官方70B/GPT-4复现；原14B空答案误判记录保留。自检9次不计原题，9次正式评分没有重复读者生成，审计0模型调用；abstention和不同题型必须分别匹配控制。
- [x] 实际成功CI41215520402工件13047975722的78份载荷与Git／安装字节一致；未修改研究提交dd410314的消费者及公共入口完成Oracle/S/M各3条完整原记录回读，答案／证据与输入隔离、原会话关系和数据库身份核验通过。共享任务根`longmemeval-fixed-ci-research-handoff-20261101-001/qualification-public.json`保留终态证据，0模型调用，未启动研究观测分析。

已声明协议完成，不表示每份500题全量生成或本地judge广泛准确率。M所选助手证据题被作者用户侧检索聚合排除，空正例recall_all不计检索成功。Dense检索、索引扩展、其他模型及未匹配题型／abstention绑定仍未验证；这些可选模式与已交付协议分开记录。所有原题、回答、检索日志和权重留在服务器私有目录。

### Mind2Web（固定原题原生交付完成）

- [x] 固定官方源码版本 `33bd95caeee7bba22dd08ecc935845e15c5e5dc7`，31个文件逐一核对Git对象；服务器源码归档SHA256与中转端一致。
- [x] 核对官方LLM入口、原始提示、候选排名、每轮五选项比较及评分；原生50-token生成预算不变。
- [x] 固定原始数据版本及`test.zip`、`scores_all_data.pkl`的官方LFS大小和SHA256。
- [x] 两份数据资产完整取得及服务器大小、官方LFS SHA256校验通过；加密测试归档15个JSON成员目录安全检查通过，展开约6.1GB，不在本地落盘。
- [x] 官方旧版依赖安装到独立Python3.11环境，原版psutil5.9.5通过匹配头文件构建；`python -I -m pip check`及官方入口导入通过。必须使用`-I`排除继承的CANN路径，不改推理服务依赖。
- [x] 三个split各固定前3原题，共9任务89动作；原始字段、完整动作、官方候选分数和排名覆盖40994个候选元素，受限pickle读取校验通过。
- [x] 官方加载器及原始tokenizer端到端CPU预检通过，24/35/30个原始动作次序一致；各split的top50候选召回缺失为7/4/6，保留原生记零行为，不补候选、不换题。零模型调用，非原生执行完成。
- [x] 固定9原题89动作完成846次NPU请求及官方评分；原提示、候选比较和50-token预算保留。显式非思考模型适配，首次默认模式失败保留，不是作者GPT-3.5复现。

首次实机尝试已保留：现有Qwen3.5-9B服务首个模型请求HTTP200，但`message.content`为`None`，官方`postprocess_action_llm`调用`.strip()`报错，未形成有效评分。固定原题、50-token预算和评分器均未更改；本次不计完成，后续先核对模型响应/思考模式适配，不补造答案、不将接口200当作原生流程跑通。验证窗口已释放共享租约，原模型服务未停止。

- [x] 本地构建包在服务器独立安装后，9原题89动作经公共索引、公共消费与现有研究消费接口逐字段还原一致；研究消费文件SHA256与当前工作区一致，未改研究代码、未运行研究分析。此证据不是远端CI产物验证。
- [x] 实际CI产物12929054859的68个运行时文件与Git内容一致；安装后9题89动作双端索引消费通过。公共入口用846份保存响应复核，提示、次序、全部预测及官方指标一致，新增模型调用为零。原生结果审计及租约释放检查通过。[正式回执](validation/mind2web-native-delivery-20261030.json)。

这里是离线网页动作预测，不是实时浏览器任务。原格式使用最近五条参考历史动作；正确候选不在top-k时原评分器直接记零，不得补入正确候选来提高通过率。保留官方`marco_*`字段及micro指标，分别解释其含义。解密测试文件仅存服务器，不公开再分发。

原题准备与执行（先在独立客户端安装作者固定依赖；`tokenizer`使用本地Flan-T5-base tokenizer，模型推理由已部署服务承担）：

```bash
python -I scripts/prepare_mind2web_original_tasks.py --archive /external/mind2web-original-assets/test.zip --output /external/mind2web-fixed-tasks --execute
python -I scripts/run_mind2web_original.py --author-root /external/mind2web-author --tasks-root /external/mind2web-fixed-tasks --tokenizer /external/flan-t5-base-tokenizer --scores /external/mind2web-original-assets/scores_all_data.pkl --output /external/mind2web-results --endpoint http://127.0.0.1:19467/v1 --model Qwen/Qwen3.5-9B --disable-thinking --execute
```

执行者必须事先取得自己的服务访问与资源窗口；此处回环地址是本次独占保护代理示例，不会自动创建服务或租约。去掉`--execute`仅生成计划，不调用模型。输出目录必须新建且位于仓库外。原生运行约846次请求，不是9次短问答；原ZIP读取采用标准库，解密需要CPU时间，不能据此宣称引擎性能。

协议与资产身份见[固定官方流程配置](../../configs/workload-native/mind2web-original-protocol.json)。配置检查仅证明约束已登记，不代表原生运行完成。

公共资产入口（服务器已使用`verify`核对两份完整原文件；不解密、不执行pickle）：

```bash
python scripts/prepare_reused_research_assets.py MIND2WEB --mode plan
python scripts/prepare_reused_research_assets.py MIND2WEB --mode verify --data-root /external/mind2web-original-assets
```

`data-root`中必须是原始`test.zip`和`scores_all_data.pkl`。缺文件时使用同一入口的`download`模式，显式指定外部目录和足以覆盖812936103字节的`--max-bytes`；不得将加密归档替换为自建题目。原生运行入口仍在实机验证，尚不能将此资产校验命令当作benchmark运行命令。

固定源码准备（服务器已实测31个原文件身份校验；不会执行作者代码或安装依赖）：

```bash
python scripts/prepare_mind2web_author_source.py --author-root /external/mind2web-author
python scripts/prepare_mind2web_author_source.py --author-root /external/mind2web-author --archive /external/Mind2Web-33bd95c.tar.gz --execute
```

归档必须来自`https://api.github.com/repos/OSU-NLP-Group/Mind2Web/tarball/33bd95caeee7bba22dd08ecc935845e15c5e5dc7`；每个文件分别检查Git对象、字节数和SHA256，而非仅依赖归档名称。保留MIT许可证。客户端采用作者`requirements.txt`指定版本，已在Python3.11.6/aarch64独立环境安装和导入验证；源码构建psutil需要匹配Python头文件，不能通过升级版本绕过。NPU推理仍使用既有Qwen3.5-9B服务，CPU客户端的torch2.0.1不承担模型推理。非思考模式是显式的模型适配，不是作者GPT-3.5复现，也不能将抽样分数当作全量榜单或性能结论。

### LiveCodeBench（原生固定抽样交付完成；自修复显式兼容）

核查依据为[官方源码](https://github.com/LiveCodeBench/LiveCodeBench/tree/28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24)。固定源码压缩包已下载到专用
CPU 工作区，SHA256 为
`56c7c42aee979def86b69547ecd6d5a877a0e4940af8923bf4325ca0defecbaf`。以下两个原始数据文件已取得并与发布仓库 LFS
SHA256、字节数核对一致；这不是原生运行或评分通过。

| 原始数据仓库                  | 固定 revision                            | 原路径                           | 字节数 | SHA256                                                           |
| ----------------------------- | ---------------------------------------- | -------------------------------- | -----: | ---------------------------------------------------------------- |
| livecodebench/execution-v2    | ff6ea0e2a638001006ddcc41259eff23a4283fb2 | data/test-00000-of-00001.parquet |  71483 | c54e4377642f689758c2ee8664457af1b95ba95813a18208003c7a5a1efa410f |
| livecodebench/test_generation | 6f3ac40bbecf81eba15899139d279b077f2816fd | test.parquet                     |  73331 | 4826aa00c059d6d47a099606ceed2d0e51d3aeeb1868f1bcf349a038bb64b4b1 |

原始文件仅在服务器 `/root/livecodebench-original-data-20261028-001`，未进入 Git 或
Windows。官方直连返回连接拒绝后，经显式镜像传输获取同一固定版本。完整代码生成仓库 `livecodebench/code_generation` 的 revision 为
`bb83f1c3c643a772b7a2a8f07bd62c2aab76be8f`，`test.jsonl` 为9,375,644,586字节，现已由公共获取入口完整下载至服务器
`/root/livecodebench-full-original-data-20261028-001` 并通过原始LFS SHA256校验，退出码0。固定原顺序前3题
`abc301_f`、`abc301_a`、`abc301_c`
已由公共准备入口保留全部11字段及不解码的完整私有测试，退出码0；现已完成这3题的30候选原始生成、原评分器评分与独立保存结果核验。证据见[分阶段回执](validation/livecodebench-original-readiness-20261028.json)。数据卡仅标注
`license: cc`，不能据此推断具体 CC 条款或随意再分发；代码 MIT 许可不替代数据许可。

- [x] 核查
  `benchmarks/code_generation.py`、`runner/parser.py`、`runner/scenario_router.py`、`runner/main.py`、`runner/runner_utils.py`
  和 `runner/oai_runner.py`。
- [x] 三个固定版本的原文件已取得并校验，固定前3题已按原字段完整准备；未将准备完成当作原生交付。
- [x] 固定3题已在无网络、非root、只读根文件系统的隔离容器中通过官方题目构造器和完整测试绑定检查，原公开加私有测试分别为27、15、34条，未减少测试。这里只验证格式兼容，不代表原生环境整体合格或已生成/评分。
- [x] 公开CPU环境配方实际构建、53个依赖版本及原模块导入核验通过；公开提示准备入口输出与实际NPU请求逐字节一致。
- [x] 代码生成固定3题×10候选已在现有Qwen3.5-9B上完成推理和官方隔离评分，7/30候选正确；21个length输出全部保留，不能当作模型质量或论文复现证据。首次90秒通信超时保留，仅将后续通信等待显式改为1200秒，生成2000-token预算与评分6秒时限未改。评分输出和指标独立核验通过，不重跑推理。
- [x] 自修复原模板缺字段异常已实测保留；经明确允许，增加显式`error-message-only`兼容模式，仅引用原有错误消息，原始反馈不改写、缺失测试输入/期望输出不补造。公开准备入口已生成真实23个失败候选的修复请求，保留7个已通过候选。
- [x] 自修复显式兼容配置的23个新请求、原评分及保存审计已闭环，原7个通过候选未重新推理。独立核验原反馈文件未改、20个缺字段候选仅使用原错误消息；12个修复输出length原样保留。组合后30候选由原评分器评分，9/30通过；不宣称未经修改的官方自修复模板通过，也不据此作模型质量或引擎提升结论。
- [ ] 固定原始数据仓库 revision、原文件哈希及许可。默认加载器使用 `code_generation_lite`；`--not_fast` 对应 `code_generation`
  完整测试。当前完整加载函数虽然接收 `release_version`，却未将它传给数据加载器，不能仅凭命令参数声称固定发布版本。
- [x] 代码执行预测、测试输出预测各固定原始前三行完成3请求×10候选推理及官方隔离评分，均退出0；60个候选均为stop，保存结果独立核验通过。代码执行保留重复题号2777及不同原生id；测试输出保留题号和test_id组合，不按题号去重。未做全量或引擎性能结论。
- [x] 在隔离容器中加载原生题目和评分。私有测试字段可能经 base64、zlib 和 pickle 解码，禁止在宿主机反序列化；测试输出预测的原评分器也使用
  `eval`，不能因为任务名字不含代码执行就放到宿主机运行。对外读取接口保留原字段，不自行删测试或改题。
- [x] 保留原生默认生成预算：每题10个样本、temperature 0.2、top_p 0.95、max_tokens 2000；代码生成原评分默认 timeout
  6秒。固定抽样只减少题数，不减少每题候选和测试。
- [x] 核验现有 NPU 模型与原生提示风格的适配。官方 OpenAI runner 可使用 SDK 的服务地址配置，但官方主入口仍要求模型在 `LanguageModelStore`
  中登记；不得冒用 GPT 名称掩盖实际模型，也不因依赖声明含 vLLM 而在 CPU 上运行模型。
- [x] 分别明确代码生成、自修复、代码执行、测试输出预测的抽样覆盖状态。自修复承接真实代码生成及评测反馈，使用已声明兼容模板；不能宣称未经修改的官方模板或全量复现。四场景抽样已有证据，但统一接入和研究消费尚未闭环。
- [x] 公共入口、可复现环境、四场景原始输出及评分回执、维护中研究侧各3条无损回读、实际CI工件12831578310安装载荷及索引回读已核验；47e9f804三项远端CI通过。按固定原生抽样及显式兼容边界更新机器清单。

### MVBench视频理解（固定原生协议交付完成）

- [x] 官方video分支完整3895文件共15,796,297,352字节已在服务器逐文件校验，复用单份外部缓存，不下载到开发机或提交原始视频。

- [x] 固定版本VLMEvalKit的原始生成函数已生成全部4000题MVBench_MP4表，MD5与作者定义一致；正常完整构造器通过，不伪造部分数据完整性。

- [x] 固定20类每类原顺序前3题，共60题，已完成原始16帧解码与提示构造检查；0模型调用，不代表原生执行完成。

- [x] 60题原始流程推理及官方支持的exact_matching评分已完成，34/60；全部60题得到有效响应和0/1评分，实际子进程退出码0。只代表固定样本可执行，不是4000题全评、论文分数复现或性能结论；不替换失败。

- [x] 候选纯数据读取/投影在服务器核验全部4000题原字段、选项/答案和媒体关系；额外时间边界、fps、show_name与subtitle原样保留，不静默丢弃、不塞进原协议没有使用的模型输入。这不是统一提供侧或研究消费闭环。

- [x] 公共获取、统一数据适配、研究消费者与实际CI工件消费闭环：实际CPU CI工件独立安装62个包载荷字节一致，完整4000条索引、60条提供侧与当前研究消费者逐值回读通过；无新增模型调用。

  候选安装包证据已由实际CI包独立复核：工件12792027468、来源提交ce823541；62载荷字节一致，source-plan的4个入口可发现，完整4000题/索引与60条提供侧/当前研究回读通过（此前75b36c27的有效证据保留）。公共获取复用完整缓存、校验3895文件并建立不复制视频的普通文件视图；原生60题保存请求/响应/评分也已按原qa_template交叉核验。来源勾表示固定协议可用，不是全量评测、高准确率或推理引擎性能结论。

复用已核验的外部完整快照、VLMEvalKit和FFmpeg/Decord环境。模型为现有NPU视觉模型Qwen3.5-9B，是适配的替代模型，不宣称复现VideoChat2论文分数。保持原始16帧、2048输出token和温度0；关闭thinking为显式模型profile，不改题。原始API角色预处理及编码保持不变，评分使用官方支持的exact_matching，不调用付费judge。抽样评分表使用连续评价索引，原始全局索引另存source_index；题目、选项、答案和视频关系不变。

以下是准备入口（默认仅计划，无下载、无模型调用）：

完整资产获取和校验：下载依赖单独准备环境中的`huggingface_hub==0.36.0`；计划、校验与建立普通文件视图只依赖Python标准库。下载需显式完整字节预算和至少100GiB可用空间；已有缓存由官方Hub工具复用。不可达时可显式选择`--hf-endpoint https://hf-mirror.com`，固定版本与全部SHA256不变。

```bash
python scripts/prepare_mvbench_original_assets.py --cache /external/hf-cache
python scripts/prepare_mvbench_original_assets.py --cache /external/hf-cache --action download --max-bytes 15796297352
python scripts/prepare_mvbench_original_assets.py --cache /external/hf-cache --action prepare --data-root /external/mvbench-original-data
```

`prepare`先完整校验，再在同文件系统的新目录中建立硬链接普通文件视图，避免额外复制视频，也不放宽公共读取器的路径检查；跨文件系统会明确失败，已有目录不覆盖、不自动清理。原快照供原生执行，普通文件视图供统一准备与研究读取。完整3895文件清单进入公共安装包，只有`test.json`是主任务表，不将原始侧标注重复计为新任务。

安装公共数据包后，外部配置为`{"sources":[{"source_id":"MVBENCH","data_root":"/external/mvbench-original-data"}]}`；使用统一入口准备完整索引，输入引用中不写原题或答案：

```bash
python -m vllm_hust_workload_data.pipeline plan --config /external/mvbench-prepare.json
python -m vllm_hust_workload_data.pipeline run --config /external/mvbench-prepare.json --output /external/mvbench-prepared-new
```

```bash
python scripts/prepare_mvbench_original_view.py --official-checkout /external/VLMEvalKit --snapshot /external/hf-cache/datasets--OpenGVLab--MVBench/snapshots/a776e554280b99b70f00cc3eacd69a65e0727efc
```

完整快照取得并核验后，加`--execute`使用固定原生成函数生成原表；已有表只复用或拒绝异常内容，不覆盖。原生验证入口同样默认仅计划；执行前必须核实设备空闲、研究任务优先及共享服务窗口，并配置原环境的FFmpeg库路径和HF缓存路径。`--frame-cache`须为已存在的外部目录。

```bash
python scripts/run_mvbench_original.py --official-checkout /external/VLMEvalKit --snapshot /external/hf-cache/datasets--OpenGVLab--MVBench/snapshots/a776e554280b99b70f00cc3eacd69a65e0727efc --frame-cache /external/mvbench-frames --output /external/mvbench-fixed60-new-run --model Qwen/Qwen3.5-9B
```

仅在确认窗口后加`--execute`；原始请求、响应、预测及评分保存到新的私有输出目录，不加入Git。来源总勾须等上述全部交付环节通过。

上述原生入口默认固定60题验证；完整正式评测可增加`--full --execute`运行全部4000原题及同一原生评分。当前实际证据只覆盖固定60题，不把全量可运行入口写成已完成全量评测。

### Video-MME视频理解（固定原生协议交付完成）

- [x] 固定发布元数据、README原Git身份和2700题原Parquet完整SHA256已核验；900视频，短/中/长各900题；官方工具包已有8帧无字幕和带字幕两profile，不是自行削减64帧配置。
- [x] 推理前固定各时长原顺序前三个不同视频，保留每视频全部3题，共27题/profile、计划54次调用。不能按模型结果换题，字幕只按原始采样帧时间提取。
- [x] 公共原表准备入口已在服务器生成/复核2700题完整原TSV，与作者MD5一致；全部题目/选项/答案及视频/字幕路径关系通过，原注释不改写。只有准备证据，不冒充完整媒体或原生执行。
- [x] 字幕库pysubs21.8.0按官方PyPI轮子SHA256独立安装，未改基础客户端/模型服务；原subtitle.zip完整身份及744份原SRT、275142条事件解析通过，0模型调用。镜像无可用版本的失败保留，官方PyPI安装成功，未静默换版本。
- [x] 候选安装包阶段核验64载荷，原2700条全部11字段可完整重建、完整2700条索引通过；固定27题在提供侧与研究消费者回读一致。这一历史阶段仅为注释数据链，0模型调用，不能独自证明媒体/原生完成或实际CI工件验证。
- [x] 实际CPU
  CI工件12795044334在服务器独立安装，64包载荷字节相等，全部2700条原始11字段/索引及固定27题提供侧与研究main消费者逐值回读通过；不是仅源码或候选wheel测试。该次核验0模型调用，范围仅注释/索引/消费，媒体与原生证据另列，见[交付回执](validation/videomme-original-readiness-20261028.json)。
- [x] 完整24原文件共101002235118字节已在服务器逐文件SHA校验，与固定发布清单一致。原整包任务末尾Hub缺文件失败保留；只恢复最后缺失的videos_chunked_20.zip（2241427390字节），其他23份复用，不重下整包、不修改资产。
- [x] 公开媒体准备入口已完成原解包及逐字节核验：900视频、744字幕，共101427428166解包字节，真实退出码0。容器干净inactive文件缓存造成的内存误拒已修复，不清全局缓存、不改模型或原解包余量；失败证据保留。
- [x] 两个原profile各27题的原解码/8帧/采样字幕/提示预检完成；随后54次NPU推理和未修改的exact_matching评分全部完成、全54
  stop、0未决评分，各13/27。没有按结果改题、缩减原流程或调用付费judge。
- [x] 独立保存审计重建原提示和API编码，核对全部8帧摘要、原请求/响应/usage/参考关系，并重新调用原评分验证54条逐题一致；审计禁用IP网络/模型生成，0新模型调用。原库对未抽中类别的nan警告保留，不改成0或隐藏未覆盖语义域。
- [x] 当前CI40663611039与已验证CI包64载荷按来源复核：63份逐字节不变，唯一变动的source_manifests资源仅新增BIGCODEBENCH获取描述，VIDEOMME描述/适配/读取代码不变；2700原字段/索引和27条消费证据有效复用。研究main与受维护工作树的消费者身份再次核对相同，不将其他未合并分支算作接通。

来源总勾对应54个profile任务、27道独立问题、9个原视频的可用性验证。固定样本在各时长内只抽到Knowledge/Humanity &
History，未验证其他语义域的模型质量；不代表2700题全评、论文模型分数或引擎性能提升。完整环境和从原资产到原生评分的命令见[组件操作说明](../../workload-preparation/README.md#video-mme-%E5%8E%9F%E7%94%9F%E8%A7%86%E9%A2%91%E4%BB%BB%E5%8A%A1)。

配置见`configs/workload-native/videomme-original-protocol.json`。原项目明确仅限学术研究、禁止商业用途及未经许可的再分发/修改；本次按科研用途私下验证，不公开原视频、题目、字幕或模型输出。[原项目使用条款](https://github.com/MME-Benchmarks/Video-MME#-dataset)。64帧及变fps配置不在当前已验证范围，不能以这份计划宣称它们已跑通。

获取入口默认只做离线计划；完整下载须显式确认学术限定、预算和外部缓存，预留至少500GiB供原归档、后续解包及服务余量。下载依赖单独环境中的`huggingface_hub==0.36.0`，其余计划/校验/注释视图使用标准库。已有同一缓存的下载任务存活时不另起重复任务；官方SDK复用固定原始对象和断点。原站不可达可显式加`--hf-endpoint https://hf-mirror.com`，仍核验同一固定版本与文件身份，不绕过访问门槛。

```bash
python scripts/prepare_videomme_original_assets.py --cache /external/hf-cache
python scripts/prepare_videomme_original_assets.py --cache /external/hf-cache --action download --accept-academic-only --max-bytes 101002235118
python scripts/prepare_videomme_original_assets.py --cache /external/hf-cache --action prepare-data --accept-academic-only --data-root /external/videomme-annotation-data
```

`prepare-data`只校验并硬链接原注释到同文件系统的新目录；即使视频缺失也可用于原记录核验，绝不声称原生媒体已就绪。完整归档另用`--action verify --accept-academic-only`逐文件校验，仍不是解包、视频解码或推理。

安装公共数据包后，使用外部配置`{"sources":[{"source_id":"VIDEOMME","data_root":"/external/videomme-annotation-data"}]}`走统一prepare/index/resolve。原始11字段、原视频/问题ID、选项顺序及答案均保留，参考答案不混入输入，媒体和字幕路径只是原生布局引用，读取器不伪造请求或实际媒体就绪状态。

```bash
python -m vllm_hust_workload_data.pipeline plan --config /external/videomme-prepare.json
python -m vllm_hust_workload_data.pipeline run --config /external/videomme-prepare.json --output /external/videomme-prepared-new
```

原表准备默认只输出离线计划；核验原Parquet与固定作者代码后加`--execute`，原生成函数生成/核对整表，不检查或假装存在900视频。已有异常表拒绝覆盖，原数据目录必须在仓库外：

```bash
python scripts/prepare_videomme_original_view.py --official-checkout /external/VLMEvalKit --snapshot /external/hf-cache/datasets--lmms-lab--Video-MME/snapshots/ead1408f75b618502df9a1d8e0950166bf0a2a0b
```

完整归档到齐后，可用以下媒体准备入口。默认只计划，显式执行须加`--execute --accept-academic-only`：先验证全部原文件，再检查ZIP原始成员名、重名覆盖、解包预算、内存和磁盘余量，使用固定作者原解包函数，最后逐字节比较全部输出与原归档流/CRC并生成外部回执。已有video/subtitle目录拒绝跳过或覆盖，未知/其他任务的提取锁不清理；失败半成品保留待核查。原函数软件夹具和部分已到齐归档头已在服务器预检，不代表完整媒体已解包或原生任务完成。

内存余量检查只保守计入干净inactive文件缓存，不把active页、shmem、dirty/writeback或匿名模型内存当作可用；仍受宿主MemAvailable和容器上限约束，保持最大原成员3倍加2GiB的原保护余量。首次仅按上限减当前占用而误拒绝的失败已保留，修正后真实容量前置检查通过；不通过全局清缓存或停模型来规避检查，不代表媒体或推理已完成。

```bash
python scripts/prepare_videomme_native_media.py --official-checkout /external/VLMEvalKit --snapshot /external/hf-cache/datasets--lmms-lab--Video-MME/snapshots/ead1408f75b618502df9a1d8e0950166bf0a2a0b --output /external/videomme-native-media-receipt.json
```

字幕依赖仅装到新的外部目录，不升级已验证的基础环境。原生调用方仍须显式加载该目录并核验版本；安装依赖不是原生评测：

```bash
python -m pip --isolated install --no-deps --only-binary=:all: --require-hashes --target /external/videomme-subtitle-overlay -r configs/workload-native/videomme-subtitle-requirements.txt
```

原生执行入口已提供，但尚未真实执行，不据软件测试打来源总勾。完整媒体回执通过后，分别选择作者已有`Video-MME_8frame`、`Video-MME_8frame_subs`配置；默认只输出计划，显式执行另加`--execute --accept-academic-only`。每个配置固定27题，覆盖短/中/长视频各前三个视频的全部原问题；`--full`运行该配置全部2700题。入口保持原提示、抽帧/字幕、API编码、2048生成预算及exact-matching评分，不更换失败题，不调用付费judge；未决评分和错误保留，不写成完成。使用专用外部帧缓存和新的私有输出目录，完整媒体回执、当前选定媒体SHA及作者源码均检查；原SDK缓存缺失的main元数据引用显式绑定固定版本，已有冲突引用拒绝覆盖。执行前仍须协调研究侧资源，不能把示例命令视为获准占卡。

可在完整媒体回执通过后加`--execute --accept-academic-only --prepare-prompts-only`先做CPU原生输入预检：使用相同原始抽帧/字幕/提示函数，保存选定原题的私有提示及8帧摘要，0模型调用、无评分、不产生来源完成勾。默认不加`--execute`仍只计划。预检与正式推理须使用不同的新输出目录；原题和帧缓存可复用，不因解码或输入失败换题。

```bash
python scripts/run_videomme_original.py --official-checkout /external/VLMEvalKit --snapshot /external/hf-cache/datasets--lmms-lab--Video-MME/snapshots/ead1408f75b618502df9a1d8e0950166bf0a2a0b --media-receipt /external/videomme-native-media-receipt.json --frame-cache /external/videomme-frames --output /external/videomme-8frame-run --model Qwen/Qwen3.5-9B --profile Video-MME_8frame
```

运行环境须使用独立CPU解码/API客户端，不能往在用NPU服务中装评测依赖。服务器实测Python3.12客户端：torch2.10.0+cpu、Decord0.6.0、pandas3.0.5及外部FFmpeg动态库，原工具包和评分/API类导入通过；`/v1/models`只读确认既有Qwen3.5-9B服务存在，不代表已执行视频。原工具包找不到可选`.env`时会打印提示，本地服务不需要因此创建外部凭据或付费judge。

带字幕配置显式传入已按上述哈希安装的独立字幕目录，入口检查pysubs2==1.8.0及实际模块来源，不自动升级。无字幕配置不加载该overlay。先建立外部帧缓存、在进程启动时提供已准备好的FFmpeg库位置，再使用原配置；以下仍默认计划，不发模型请求：

```bash
mkdir -p /external/videomme-frames
LD_LIBRARY_PATH=/external/ffmpeg/lib python -B scripts/run_videomme_original.py --official-checkout /external/VLMEvalKit --snapshot /external/hf-cache/datasets--lmms-lab--Video-MME/snapshots/ead1408f75b618502df9a1d8e0950166bf0a2a0b --media-receipt /external/videomme-native-media-receipt.json --frame-cache /external/videomme-frames --subtitle-overlay /external/videomme-subtitle-overlay --output /external/videomme-8frame-subs-run --model Qwen/Qwen3.5-9B --profile Video-MME_8frame_subs
```

### LongVideoBench长视频交错理解（原协议已审，数据授权待完成）

- [x] 2026-10-28复核[官方数据页](https://huggingface.co/datasets/longvideobench/LongVideoBench)及[原生软件](https://github.com/longvideobench/LongVideoBench)：6678题，validation1337/test5341；长视频、字幕与图像交错关系属于原任务，不能用Video-MME替代，不构造纯文本题冒充。
- [ ] 官方文件访问要求登录、分享账号信息并接受条款，本任务未接受条款、未取得受控原数据，不绕过门槛。CC-BY-NC-SA-4.0及原视频权利边界须保留；已有公共工具包源码不等于原数据就绪。原文件身份、视频/字幕关系、合格NPU执行配置、原生推理/评分及实际CI消费均未完成，来源总勾保持未选。

### TemporalWiki原基座探针PPL已交付，其他演化语料模式缺口保留

- [x] 原GPT2-large基座PPL全部8CSV／35,948行、1126个NPU批次完成，原评分标量独立重算精确一致；296cc656补齐分类API入口后两项CI通过，实际工件13073205020的78提供方载荷独立安装核验，全部原字段／索引和24条公共／当前研究回读通过，与原生全部输入文件绑定；TemporalWiki原生评测和LJSpeech原生TTS可通过分类API查询，751项CPU同套检查通过，先前漏登失败保留。现有9B／14B未重启，保护租约已释放。[交付回执](validation/temporalwiki-base-probes-native-delivery-20261101.json)

- [ ] 持续训练checkpoint、T5、生成、全文／差分语料等模式未完成，另2份语料归档和发布方许可审查仍待补；不编造更新事件。下列准备与失败记录为历史检查点，不表示当前声明的基座探针协议仍未交付。

- [x] 官方GPT2-large全部8份模型文件／3,250,026,117字节在服务器核验通过，权重整文件3,247,159,078字节与发布方LFS
  SHA256一致，未写入Windows原始数据或权重。公开候选入口`run_temporalwiki_original_npu.py`的完整8文件／35,948行计划核验通过，保留原25-token输入／输出和batch32；明确为empty-checkpoint基座原探针困惑度，不冒充持续训练论文模型。

- [x] 全部35,948行／1126批次原GPT2基座NPU评测完成，8份原困惑度CSV均返回；保存loss按原float32顺序在NPU仅重算标量评分，8份结果逐一精确一致，0新模型调用。第一轮自设4GiB分配器上限失败保留，5GiB配置成功，显存增量峰值4926MiB，9B／14B不重启；当前19项直接依赖版本及隔离`pip check`通过，继承CANN工具元数据的失败另存。仍需当前CI及公共／研究消费交接后补来源总勾，下列更早准备记录按历史检查点保留。

- [x] 隔离兼容环境安装、`pip check`及真实依赖导入通过；未替换系统Torch。执行未修改的原`Datasets.py`与原GPT2分词器，在1,776条changed探针中核验固定前三条，保留原25-token输入/输出、特殊token和左侧padding。证据`temporalwiki-original-dataset-cpu-20261101-001/qualification-public.json`；仅CPU数据处理，0模型调用，仍需权重、NPU评测和官方困惑度结果。

- [x] 重新核验原`evaluation_ppl.py`及GPT2
  baseline配置：原作者支持`checkpoint_path=""`时直接使用基座模型，配置模型为`gpt2-large`，原input/output长度25、batch32。可以继续适配这一明确的原生基座探针协议，不必把未公开的持续训练checkpoint当作该分支的必需前置；不能据此声称复现持续训练模型或改用Qwen聊天。证据`temporalwiki-original-base-probe-review-20261101-001/qualification-public.json`，0模型调用，模型/分词器及NPU原生执行仍待验证。

- [x] 原GPT2-large模型快照固定到`32b71b12589c2f8d625668d2335a01cac3249519`，7份原分词器/配置/模型卡共2,867,039字节通过Git内容身份核验，仅在服务器保存。服务器直连官方模型源的网络不可达失败保留；配置经原官方URL和既有代理有界RAM转发，未下载大权重、未加载模型，不能计原生执行。

- [x] 官方代码版本已固定，177份原软件/配置共382,667字节在服务器逐文件Git blob及SHA256核验；13份Python仅AST解析，162份原配置保留，不导入或执行作者代码。

- [x] 原协议已区分：56份探针困惑度配置、56份语料困惑度配置，以及44份微调/6份预训练配置。README示例并不是普通聊天问答，不能换成当前Qwen API调用冒充复现。

- [x] 官方历史上传版本的完整TWiki_Probes.zip已在服务器取得，620,459字节，与发布方LFS SHA256一致，未在本地下载数据；获取回执已保存。

- [x] 公共获取入口复用整包原文件，安全解包并校验全部9文件：8份月份/changed或unchanged探针CSV及原不透明.DS_Store。全部35,948行三元组/索引验证通过；未按实体猜测跨文件关联，也不构造更新事件。

- [x] 提供侧及当前研究消费者固定每文件前3条、共24条完整记录回读一致；这是候选代码验证，0模型调用，不冒充原生困惑度评测。

- [x] 实际CI工件12785028172在服务器独立安装，60个包载荷字节一致；全部35,948原字段/索引及24条提供侧/当前研究完整记录回读通过，0模型调用，不是原生困惑度结果。

- [ ] 原数据许可待核验；另外两种归档尚未取得，不得把Wikipedia全文、差分语料和subject/relation/object探针强行合并为同一种行结构。

- [x] 2026-10-31通过发布方dataset info和不可变版本tree
  API补齐三种归档身份，登记在现有探针profile的`publisher_archive_inventory`，计划CLI直接显示。另两包分别为932070039和24655216607字节，仅登记官方LFS身份，未下载、未重算整包SHA；原探针取得版本与内容不变。许可和剩余数据获取仍未完成，不计来源总勾。

- [ ] 官方模型/checkpoint、合格NPU环境及原生评测待完成。旧环境为Python3.8、Torch1.9 CUDA11.1等，不在当前NPU环境盲装或静默修改作者配置。

  仅准备/核验作者软件（外部目录；不下载数据或权重、不执行训练/推理）：

  ```bash
  python scripts/prepare_jsonschemabench_author_source.py --source temporalwiki --author-root /external/temporalwiki
  python scripts/prepare_jsonschemabench_author_source.py --source temporalwiki --author-root /external/temporalwiki --action prepare --max-bytes 382667
  python scripts/verify_temporalwiki_original_contract.py --author-root /external/temporalwiki
  ```

  原始探针资产准备（默认离线计划；实际下载和解包均须显式选择，目录必须在本仓库以外）：

  ```bash
  python scripts/prepare_temporalwiki_original_assets.py --archive /external/TWiki_Probes.zip
  python scripts/prepare_temporalwiki_original_assets.py --archive /external/TWiki_Probes.zip --action download --max-bytes 620459
  python scripts/prepare_temporalwiki_original_assets.py --archive /external/TWiki_Probes.zip --action extract --data-root /external/temporalwiki-probes
  python scripts/prepare_temporalwiki_original_assets.py --archive /external/TWiki_Probes.zip --action verify --data-root /external/temporalwiki-probes
  ```

  原站不可达时可显式加`--hf-endpoint https://hf-mirror.com`，仍核对同一发布方固定版本/整包SHA256。已有原包复用，不重下；已有解包目录只核验，不覆盖。subject/relation作为输入字段，object单独保留为参考；官方likelihood流程须读取完整三元组，不能只拿输入字段声称测过困惑度。

  安装公共数据包后，沿用统一prepare入口。外部配置示例（不是模型执行配置）：

  ```json
  {"sources": [{"source_id": "TEMPORALWIKI", "data_root": "/external/temporalwiki-probes"}]}
  ```

  ```bash
  python -m vllm_hust_workload_data.pipeline plan --config /external/temporalwiki-prepare.json
  python -m vllm_hust_workload_data.pipeline run --config /external/temporalwiki-prepare.json --output /external/fresh-temporalwiki-index
  ```

  服务器GitHub匿名配额及原代码域名连接失败记录保留；最终只通过已有认证读取原小型软件并内存中转到服务器，没有发送凭据、原软件入Git或本地落数据集。HF镜像元数据403与原站连接超时仍是数据获取缺口。软件无LICENSE文件，Wikipedia衍生数据许可应另外核对。[阶段证据](validation/temporalwiki-original-readiness-20261028.json)。

### NatureBench科学代码工作流（进行中，不打来源总勾）

- [x] 固定官方代码与数据版本，服务器取得56份原始运行/评估/环境/任务列表文件，共514543字节，Git blob及SHA256核验通过；30份Python只做语法解析，未执行科学任务代码。

- [x] 完整90任务的CPU3/GPU-low70/GPU-high17分组，以及NatureBench-25的1/22/2分组全部核验；25任务轨道是原90任务的按资源一致子集，不新增来源计数。

- [x] 未改的官方`run_naturebench.py --help`返回0；这只是准备客户端中的CLI可用性，不代表原始conda环境复现、Agent执行或科学评分通过。

- [x] 官方全部3个CPU任务的576原文件已取得，共2,754,532,128字节，500份LFS SHA256和76份Git
  blob逐份核验通过。复用574份已有文件，两个超时对象以固定原URL整文件回退取得，未截断、改内容或换任务；58,059,841字节的重复临时副本已清理，规范数据和回执保留。

- [x] 独立准备环境13个固定依赖/14个已安装分发，`pip check`及官方CLI
  help通过；公开只读入口核验576文件，官方`--skip-download --download-only`离线准备返回0。这不是科学任务执行；其他87个Full任务数据未取得。这里CPU是科学任务资源组，后续LLM仍须用NPU，未启用CPU模型推理。

- [x] 3个CPU原始任务包的关系读取已在服务器逐包核验：576文件读取前后校验，保留7项原始元数据，隐藏评估与参考结果不进入题目输入；这是候选代码验证，不代表已安装CI工件、统一索引或研究消费者接通。

- [x] 候选代码统一索引3个原任务引用、提供侧回读3条、当前研究消费者回读3条均在服务器验证通过，原输入和outcome逐值一致；研究消费者版本及摘要保存在阶段证据，0模型调用。

- [x] 实际CI工件12783786040在服务器独立安装，59个包载荷字节一致；3个原任务的索引、提供侧/研究侧完整记录回读均通过，0模型调用。

- [ ] 隔离环境、完整原生Agent流程、官方评估器与有效性judge仍待完成，不以静态问答或`--skip-judge`替代。

  任务包关系核验（不执行科学代码、不调用模型）：

  ```bash
  python scripts/verify_naturebench_original_packages.py --data-root /path/to/original-naturebench-data
  # 可选：在全新、与原始数据分离的目录建立引用索引并逐条回读
  python scripts/verify_naturebench_original_packages.py --data-root /path/to/original-naturebench-data --prepare-output /path/to/fresh-index
  ```

```bash
python scripts/prepare_jsonschemabench_author_source.py --source naturebench \
  --action prepare --author-root <仓库外官方源码目录> --max-bytes 600000
python scripts/verify_naturebench_original_contract.py \
  --author-root <同一官方源码目录> --output <源码目录外的新证据文件>
python scripts/prepare_naturebench_original_assets.py --mode download \
  --author-root <同一官方源码目录> --data-root <仓库外服务器新数据目录> \
  --cache-root <独立新缓存目录> --max-bytes 3000000000
python scripts/prepare_naturebench_original_assets.py --mode verify \
  --data-root <上述原始数据目录>
```

下载客户端依赖见`configs/workload-native/naturebench-preparation-client-py312-observed.txt`，是已验证的Python3.12
ARM准备环境，不是原始科学任务conda环境。默认优先调用未改的官方download-only入口；确需镜像时显式指定`--hf-endpoint https://hf-mirror.com`，通过[官方HTTP后端扩展](https://huggingface.co/docs/huggingface_hub/v0.36.0/package_reference/utilities#configure-http-backend)仅保持同版本分页请求的传输域名，保留路径和参数。中断后可用相同目录加`--resume`，先完整校验已有文件再续传；不要与其他下载器同时写同一目录。对象超时可显式加`--transport publisher-files`，仅获取缺失的完整原对象，仍须全部576文件的原始摘要/大小通过，不更换数据源或任务。所有已取得数据原样保留在服务器，未下载到本地工作站或Git。

官方数据是科学代码任务包：`problem/`向Agent可见，`evaluation/`与隐藏真值不可泄漏；任务环境、评估器和有效性审查均属于原生流程。当前ARM/NPU服务器没有Docker/Podman/bwrap，官方conda导出又包含linux-64/MKL构建，不能悄悄换包便宣称原环境已复现。代码/任务规范的MIT许可不覆盖全部第三方科学数据，必须保留每任务许可与署名；公开可下载不等于可随意再分发。[原始版本与资源分组](../../configs/workload-native/naturebench-original-release.json)；[阶段证据](validation/naturebench-original-readiness-20261028.json)。

### OpenCodeInstruct原始代码训练语料（固定原始profile已交付）

- [x] 核验官方固定版本元数据：50个原始train分片、6,861,113,102字节，公开CC-BY-4.0。复用已有原字段适配，问题与教师代码、原测试、历史测试状态和判断信息分开保存，不生成替代题目。
- [x] 公共获取入口支持固定原文件，公共引用准备入口及微型夹具回归已补齐；这些只是软件能力验证，不代表真实数据或模型已跑通。
- [x] 服务器已取得预先固定的0/24/49原始分片，共422,115,928字节，发布方LFS摘要及大小全部复核通过；每份Parquet元数据100000行，schema及原前3条字段类型已检查。实际`average_test_score`为double/float，不按HF卡片中的string描述强转。[阶段证据](validation/opencode-original-readiness-20261028.json)
- [x] 三份可用原分片全部300000条九字段、引用索引及9条当前研究侧回读通过；使用候选目录和已验证CI适配运行库，不能冒充新目录的实际CI工件验证。
- [x] 固定9条原问题在现有9B BF16 TP1 NPU服务生成并保存核验：7 stop、2 length，预算8192，未换题或重跑。实际3bf633eb CPU
  CI包57载荷核验和9条当前研究回读通过，未重建索引。
- [x] 3bf633eb的tests、lint及CPU
  smoke全部通过，[最终交付范围与证据](validation/opencode-native-delivery-20261028.json)。原`unit_tests`及生成代码未执行；如需功能评估，必须另行核验隔离环境，不把生成跑通当代码正确率。

```bash
python scripts/prepare_reused_research_assets.py OPENCODE --mode plan \
  --files data/train-00000-of-00050.parquet data/train-00024-of-00050.parquet data/train-00049-of-00050.parquet
python scripts/prepare_reused_research_assets.py OPENCODE --mode download \
  --data-root <仓库外原始目录> --max-bytes 450000000 \
  --files data/train-00000-of-00050.parquet data/train-00024-of-00050.parquet data/train-00049-of-00050.parquet
python scripts/prepare_reused_source_records.py OPENCODE \
  --data-root <仓库外原始目录> --output <仓库外新引用目录> \
  --files data/train-00000-of-00050.parquet data/train-00024-of-00050.parquet data/train-00049-of-00050.parquet
```

这是合成代码训练语料，不是独立代码质量benchmark。教师答案和原历史测试分数不代表本地模型结果；实际三份文件的`average_test_score`为double，与HF卡片的string描述不同，必须原样保留数值，不能擅自改类型或补造评分器。

原问题生成复用已测的训练语料执行流程，默认只显示计划；`--full`为流式执行所选原文件全部行，默认固定原前3条/文件，不按结果换题。可用模型回退profile如下，已完成9条原问题生成与保存核验，但不是作者质量评测配置。

```bash
python scripts/run_opencode_original.py --mode run \
  --data-root <仓库外原始目录> --output <仓库外私有新结果目录> \
  --files data/train-00000-of-00050.parquet data/train-00024-of-00050.parquet data/train-00049-of-00050.parquet \
  --model Qwen/Qwen3.5-9B --max-tokens 8192 --context 262144
python scripts/verify_opencode_original_saved.py \
  --data-root <仓库外原始目录> --saved <上述私有结果目录> \
  --files data/train-00000-of-00050.parquet data/train-00024-of-00050.parquet data/train-00049-of-00050.parquet
```

### OpenMathInstruct-2原始推理训练语料（固定原始profile已交付）

- [x] 完整主train32分片的固定版本/URL/LFS摘要/大小已登记，共7,576,089,560字节；不把与主train重叠的1M/2M/5M子集重复计为独立问题来源。
- [x] 运行前固定的0/15/31原分片已在服务器取得并核验，共710,707,907字节。三份Parquet元数据行数为436650/436650/436649；已检查原schema及各原前3条字段类型，未据此宣称所有行值或索引均验证通过。
- [x] 三个可用分片全部1,309,949条原字段及索引完成核验；实际CI28997404包的56个载荷与工件一致，9条公共/当前研究消费者回读通过。初次私有核验器误比较逐文件与三文件全量选择范围，已明确校验各自范围并恢复只读核验；失败记录保留，未重建索引或重跑模型。
- [x] 原前3条/分片共9条原问题已在现有Qwen3.5-9B BF16 TP1 NPU3服务生成：3 stop、6
  length，显式预算8192，未改问题、换题或重试。保存的输入、参考、来源、预算和响应已用公开核验入口复核，0新增模型调用；不独立判断数学正确性，不冒充作者教师模型复现。
- [x] 当前a70da63a的tests、lint及CPU
  smoke全部通过；独立安装实际CI工件并核验56个包载荷，复用全量索引后9条公共/当前研究侧回读通过，未重建索引或重跑模型。`problem`独立作为输入，教师解答、参考及空答案均保留在参考侧。[最终范围与证据](validation/openmath2-native-delivery-20261028.json)；[历史阶段证据](validation/openmath2-original-readiness-20261028.json)。

```bash
python scripts/prepare_reused_research_assets.py OPENMATH2 --mode plan \
  --files data/train-00000-of-00032.parquet data/train-00015-of-00032.parquet data/train-00031-of-00032.parquet
python scripts/prepare_reused_research_assets.py OPENMATH2 --mode download \
  --data-root <仓库外数据目录> --max-bytes 800000000 \
  --files data/train-00000-of-00032.parquet data/train-00015-of-00032.parquet data/train-00031-of-00032.parquet
```

这是发布方公开的合成推理训练语料，不是独立质量benchmark，不能把训练语料续写说成官方独立评测、训练复现或真实线上分布证明。当前只取得主train三个固定分片，其他29份仅登记；原始数据不放入仓库或本地工作站，运行配置须明确模型与预算，不能补造作者未给出的质量评分器。

原问题生成默认只展示计划；运行必须显式给出模型、上下文和输出预算。以下为当前已验证的本地模型回退profile，不是作者官方质量配置：

```bash
python scripts/run_openmath2_original.py --mode run \
  --data-root <仓库外原始目录> --output <仓库外私有新结果目录> \
  --files data/train-00000-of-00032.parquet data/train-00015-of-00032.parquet data/train-00031-of-00032.parquet \
  --model Qwen/Qwen3.5-9B --max-tokens 8192 --context 262144
```

默认原前3条/文件，`--count`明确增加固定前缀；`--full`流式遍历所选文件全部原行，不按结果过滤。输入只使用原`problem`作为user内容，模型原生模板由服务处理；不加入教师解答。`verify_openmath2_original_saved.py`使用相同原始目录/文件选择及`--saved`结果目录复核，无模型调用、无独立数学评分。原字段准备复用`prepare_reused_source_records.py OPENMATH2 --data-root ... --output ... --files ...`；全量模式可生成三个所选分片的全部引用索引，不复制原载荷。

### DRAGOn同版本原始RAG资产（未交付，不打总勾）

- [x] 复用公共获取入口，服务器取得发布方1.15.0同标签四份原文件并核验SHA256/大小，共2,277,123字节：600个公开问题、542篇公开文本、600条参考QA及542条参考文本映射。
- [x] 全量原关系已核验：问题和参考问题逐字一致、全部公开/参考文本ID映射及正文一致、933条参考文本关联有效。四份原文件均未修改，不把参考答案混入模型输入。
- [x] 原作者16份协议/核心库文件、44366字节已从固定Git
  blob取得并核验；复用现有作者源码获取入口，不复制作者代码入仓库。原始Git连接失败、HF主通道失败及改用原始同摘要传输的过程保留。
- [x] 原始四文件关联适配已实现；服务器独立安装候选wheel，55个包载荷与wheel一致。全部600条原问题/参考字段及索引通过，原前3条公共/当前研究消费者回读逐字段一致，0模型调用。当前研究main版本已在线核对；未更改研究线程默认依赖。
- [x] 原作者核心库环境已在Linux aarch64/Python3.12.13独立安装112项依赖；清除客户端继承的CANN
  PYTHONPATH后，pip一致性检查、四个原模块导入及原评分器软件夹具通过。实测约束与原lib requirements的dry-run无变更；初次旧环境缺依赖与继承路径污染记录保留。
- [x] 实际CPU CI工件已独立安装复核：55个载荷文件与工件一致，全部600条原字段/索引和原前3条当前研究消费者回读通过，未重跑模型或升级研究默认依赖。旧候选记录保留，不冒充实际CI工件。
- [ ] 官方检索/生成与原题评分尚未完成；需要适配NPU的embedding和reader环境。实际CI读取/回读也不替代benchmark运行。[准备与CI消费证据](validation/dragon-original-preparation-20261028.json)

```bash
python scripts/prepare_reused_research_assets.py DRAGON_PUBLIC --mode plan
python scripts/prepare_reused_research_assets.py DRAGON_PUBLIC --mode download \
  --data-root <仓库外数据目录> --max-bytes 3000000
python scripts/prepare_jsonschemabench_author_source.py --source dragon --action prepare \
  --author-root <仓库外作者源码目录> --max-bytes 100000
```

准备好原始四文件后，可使用公共原字段读取入口（不是模型执行或评分入口）：

```python
from pathlib import Path
from vllm_hust_workload_data import iter_workload

records = iter_workload("DRAGON_PUBLIC", data_root=Path("/external/dragon-original"))
```

保持发布方600个问题的原始顺序；`task.input`只含原id/question，`outcome.original_reference_qa`保留完整原始8字段参考，`runtime.corpus_file`指向完整原语料。四份文件逐一检查版本/摘要及完整关联，不能按参考文件行号拼答案；协议版本与各数据仓库版本分别保留。读取前后均核验原文件，参考变更会失败，不从gold生成输入。

核心库准备可使用外部Python3.12环境，以`configs/workload-native/dragon-core-client-py312-observed.txt`作为约束安装不修改的原`lib/requirements.txt`。这是实测兼容版本，不是作者冻结锁；模型客户端另见`dragon-npu-model-client-py312-observed.lock.txt`，固定原前三题的NPU嵌入/现有reader适配已验证，不能只安装核心库就宣称完整基线可运行。客户端须排除未经核验的继承`PYTHONPATH`，不要修改在用NPU服务环境。

仅登记既定113项中的`DRAGON_PUBLIC`，不新增交付范围；参考仓库虽名含private，已通过官方无凭据元数据实查同版本非gated且Apache-2.0，不能仅凭名字推断公开/私有。四份资产保持独立版本身份，均固定同一发布标签。其月度新闻快照不是生产文档增删事件或逐请求版本轨迹，不补造不存在的在线事件。

### VisionArena-Chat原始多模态会话资产（未交付，不打总勾）

- [x] 固定官方版本完整文件元数据已登记：43份原Parquet分片，共84,565,530,886字节（约84.6GB），每份记录发布方LFS摘要、大小和原始URL，见`visionarena_original_files_v1.json`。仅元数据，不含图片或会话正文。
- [x] 官方自定义协议边界已补充：获取/使用须接受协议，禁止向第三方转移原数据；未代用户或机构接受协议，未启用自动下载profile。非gated不等于无限制开放再分发；授权使用者从发布方直接取得原始数据。
- [ ] 原图像与会话绑定、完整历史的多模态续写、声明生成预算及实际CI/研究消费尚未验证。保留现有原生字段适配，不用文本代理替代图片，不把历史模型回复当标准答案，不另造官方质量评分。

该来源是已发布历史会话语料，不是自带完整任务评分器的benchmark。现有适配状态仍为`CONTRACT_MAPPED_ASSET_VALIDATION_PENDING`；43分片的登记不等于取得数据或原生运行成功。[固定版本官方协议](https://huggingface.co/datasets/lmarena-ai/VisionArena-Chat/blob/1394b4f59ab6f1f2e5aff6bc15b448e15960e170/README.md)

### AppWorld固定原三题原生执行、官方评分与消费交接

- [ ] 2026-10-31研究消费端完整性回归待修复：当前研究消费文件与既有回查证据的SHA256一致。小型结构夹具确认，修改原数据库资产后公共入口拒绝读取，研究旧入口却重新计算引用并返回。此前固定三题的原生执行、官方评分和未改资产的完整记录回读仍有效，但不能据此声称防篡改校验已闭环。拟仅让研究AppWorld消费委托现有公共reader，并补数据库/参考资产负例；研究线程已暂停，正在等待用户确认最小接入修复，不启动研究实验。机器记录见`dataset-validation-status.json`的`current_research_consumer_integrity_followup`。

- [x] 2026-10-30经授权恢复原专用CPU工作区，原模板、持久卷及资源预算不变，实际Pod运行且两容器就绪。既有732条索引与历史实际CI提供方复用；固定原三条经未修改的研究消费入口读取，完整记录与历史保存结果一致，原specs及每题6份参考资产保持分离。0模型调用、0
  Agent或评分重跑。[研究消费回查证据](validation/appworld-research-consumer-20261030.json)

- [x] 官方0.1.3.post1核心客户端已在服务器Linux
  aarch64、完整CPython3.11.17隔离环境安装；官方`appworld install`、依赖一致性检查和CLI帮助均成功，83个安装发行包。受保护代码仅解包到服务器私有环境，未放入Git。

- [x] 官方`appworld download data`取得14696文件、169528045字节，全部留在服务器；公共只读入口复用同一数据验证train90、dev57、test_normal168、test_challenge417，共732项原任务的划分ID和目录关系，无重下、无改题。[数据准备证据](validation/appworld-original-data-preparation-20261028.json)

- [x] 公共入口固定原三题完成141步ReAct及官方评分：28条精确缓存响应复用、113次新增NPU请求全部HTTP200。三题均1/5测试通过、任务未成功，失败保留；这表示原生流程可用，不表示高质量。实际CI71运行载荷与当前提供方一致，732条原索引不变，研究侧三题完整记录、原specs及每题6份独立参考资产回读通过。详见[最终交付回执](validation/appworld-native-delivery-20261030.json)。不代表全732题推理、原70B复现、确定性环境重放或性能结论；历史准备回执不升级为执行证据。

2026-10-30历史首次只读复核（早于上述原生交付完成，以下为当时结论，不代表当前勾选状态）：原专用CPU
Pod已不存在（Kubernetes返回`NotFound`）；Coder最新build为`stopped`、原因`autostop`、停止job成功，但本任务原80GiB持久卷仍为`Bound`，UID与既有记录一致。因此不能认定历史数据丢失，也不能认定旧进程仍在运行；恢复自有工作区后，应先从持久卷回查既有结果，不重复下载或执行原任务。本次未挂载卷、未修改平台配置、未新增模型调用，AppWorld保持未勾选。

公共入口：`python scripts/prepare_appworld_runtime.py --output <仓库外新目录> --uv <已有uv> --python <完整Python3.11>`默认只输出计划。显式加`--execute --install-protected-code`才安装官方包并在私有环境解包；可用`--uv-cache <已有自有软件缓存>`避免重复下载。工具将`APPWORLD_ROOT`、`APPWORLD_CACHE`及临时目录都指向外部目录，不改HOME或模型服务环境。该版本声明范围安装不是作者冻结锁；ARM需要Python.h。历史缺头文件和根盘缓存失败均保留，未删数据掩盖失败。

原始数据入口：`python scripts/prepare_appworld_original_assets.py --output <外部新目录> --client <上步client目录> --cache <上步cache/appworld目录>`默认计划，加`--execute`调用官方CLI；拒绝已有目标，避免作者下载器覆盖旧数据。已有资产用`--verify-only`只读核验，不重下不覆盖。不把这些任务改造成静态prompt，也不把目录关系核验当作原生任务执行。

本次核心客户端的83项实际依赖版本记录在`configs/workload-native/appworld-core-client-py311-observed.txt`。需要约束重建时向环境入口显式传`--constraints <该文件>`；这是本次ARM/Python3.11.17环境的观测版本，不是作者冻结锁，也未证明其他平台兼容。本次已安装环境直接复用，不为增加证据重复安装。

### BixBench原问题与科学环境资产准备（未交付）

- [x] 最新原生终态：固定三题均完成原40动作，客户端退出0。第一题原轨迹完整复用，第二、三题新增共160次请求/HTTP200响应，三题最终答案均为`null`，原样保留，不重新生成或替换。三题原CSV输入已按`str/str/range`模式准备，尚未正式评分。
- [ ] 评分语义待用户决定：五个独立合成控制例仅3个符合预期。已在未改官方源码上复现空规范化字符串的部分匹配可直接判对；区间分支也被传入普通而非发布方已有区间模板。不是把两个问题都归因于小模型；诊断本身零新模型调用。已请求保留原版并增加显式最小修复版的授权，未改评分规则、原题或原分数。不得把控制例计入benchmark成绩，来源总勾仍保持未完成。

以下准备和失败记录保留其发生时的范围；与最新终态不同时，以上述最新终态及机器清单为准。

- [x] 固定官方数据版本，服务器取得285,842字节原JSONL，按发布方Git
  blob身份和SHA256核验；实际为205条扁平问题、17字段，问题ID唯一。数据卡的嵌套`questions`结构不能用于本版本适配。原始正文、参考答案和canary没有进入本机文件或Git。
- [x] 注册原JSONL及64个ZIP的完整文件身份，实际205条问题引用其中59个ZIP，约5.20GB；另5个已登记ZIP不因此变成额外任务。全部`data_folder`关系核验通过。固定原前三题为`bix-1-q1`、`bix-1-q2`、`bix-10-q1`，不为覆盖更多capsule而换题。[原始资产清单](../../workload-preparation/data/catalogs/bixbench_original_files_v1.json)、[准备证据](validation/bixbench-original-metadata-20261030.json)
- [x] 固定原前三题对应的两份原ZIP已在服务器取得，共5,962,719字节，发布方LFS摘要及16个成员CRC全部核验；未解压或执行代码，没有原始数据落入本机文件。全体语料仍缺另57份引用ZIP，不能将两份算作全量资产取得。
- [x] 原生字段适配器安装候选软件包后，服务器全部205条原字段可还原、205条索引及固定前三条公共/未修改研究消费完整记录回查通过，原ZIP引用与摘要验证通过。输入只含原问题和数据目录名，答案/结果/未知扩展字段保持参考侧；不是API
  prompt生成器。[候选包交接证据](validation/bixbench-candidate-handoff-20261030.json)
- [x] 实际CI工件12953366374的71个运行载荷与Git及安装文件逐字节一致；服务器全部205条原记录还原、复用未改变的205条索引及固定原前三条公共/研究消费与ZIP校验通过，无索引重建和模型调用。[实际CI交接证据](validation/bixbench-actual-ci-handoff-20261030.json)
- [x] 官方ARM64 Notebook镜像16层摘要与最终配置身份核验通过；原作者Notebook
  API的可信`1+1`夹具在非root、离线、只读根容器执行成功。保留第一次运行副本权限失败，只调整本任务副本的组权限及客户端umask，未改作者源码或共享数据权限。[环境夹具证据](validation/bixbench-notebook-preparation-20261030.json)
- [ ] 原生科学任务与官方评分仍待完成，不打来源总勾。固定原前三题的40步Agent流程使用单NPU
  Qwen3-14B和专用CPU隔离Notebook；原始消息不重排，温度1.0及4096-token预算保留。此前9B拒绝原始系统消息位置；14B后续收到6次HTTP200，但客户端60秒超时，0题完成。输出文本位于reasoning字段，需核查解析器与非思考绑定，不能搬移字段伪造响应。失败环境、请求和响应均保留，原评分尚未执行。

本来源保留科学Agent执行路线，不能用发布方另有的零样本问答模式替代。固定原配置默认40步，依赖`fhaviary`、`fhda`、`fhlmi`和`ldp`；作者入口为`bixbench/generate_trajectories.py`，随后按作者后处理/评分协议评判。原ZIP须保留为来源；运行副本按作者预处理移动Data内容、移除参考Notebook，不能将整个含参考Notebook的归档直接暴露给Agent。官方软件准备入口已补充，27份原文件在专用CPU工作区校验通过、13份Python文件通过静态解析；原锁文件客户端安装仍在进行，尚未验证Agent导入、隔离容器或原生执行，不打来源总勾。命令与边界见[组件说明](../../workload-preparation/README.md#bixbench-%E5%8E%9F%E7%A7%91%E5%AD%A6%E4%BB%BB%E5%8A%A1%E5%87%86%E5%A4%87%E4%B8%8E%E8%AF%BB%E5%8F%96%E5%8E%9F%E7%94%9F%E6%89%A7%E8%A1%8C%E6%9C%AA%E5%AE%8C%E6%88%90)。

2026-10-30后续环境核验：上述安装过程已经结束，Python3.12.13按原锁文件安装、依赖检查及安装后源码复核通过。显式指定原源码目录后，官方生成入口`--help`通过；不指定目录时模块未找到，不能把安装成功等同入口可用。隔离Notebook夹具随后通过，固定原题14B尝试已超时结束；原评分仍待验证，不打来源总勾。[客户端环境证据](validation/bixbench-official-client-20261030.json)

后续绑定修复正在验证：仅重启本线程服务并移除不适合非思考模式的reasoning解析器，保留同一模型、精度、上下文和原生成预算。公共入口可显式指定传输等待600秒，默认不改作者设置。新的固定三题原生尝试已观察到正文、工具调用及后续工具结果消息；尚无题目完成或评分成功结论，不打总勾。此前失败请求和响应没有被改写或覆盖。

### DiscoveryBench原始资产准备（未交付，不打总勾）

- [x] 固定官方版本的完整原树1499文件、54085184字节已在服务器按Git
  blob身份、大小、文件集合核验，包含原real/synth数据、Agent及评判源码；公共只读核验复用同一目录，无重下。原始数据和作者代码不纳入Git。[证据](validation/discoverybench-original-assets-20261028.json)

- [ ] 原生科学Agent、代码执行环境/隔离、评判模型与官方评分、公共提供方和研究侧消费尚待接通，不能把1499文件当作1499任务或已完成样本。

- [x] Linux
  aarch64/Python3.11.17独立客户端按作者原`requirements.txt`安装110项依赖，`uv pip check`及原`discovery_agent.py --help`、`discovery_eval.py --help`均通过；之后复核原资产未变化。没有构造或运行Agent，不代表安全代码执行环境或评判模型已就绪。

核心环境准备：用完整Python3.11在外部新目录建venv，再以`uv pip install --python <venv/bin/python> --link-mode copy -r <原source/requirements.txt>`安装，`uv pip check --python <venv/bin/python>`检查；软件缓存和临时目录均设置到外部目录，不改模型服务环境。从原source目录以该venv运行两个CLI的`--help`可检查导入。作者requirements不是完整传递依赖锁，未声称其他平台或未来依赖解析结果已验证。

原metadata扫描仍有缺口：982份可按UTF8
JSON解析，80份原文件解析失败，不能以CLI帮助通过声称数据可运行。解析成功文件中的1122条数据文件关系有1121条存在；唯一缺链来自`.ipynb_checkpoints`辅助元数据，不作为正式任务数据缺失的结论。正式任务目录及编码兼容还须单独核验；保留原字节，不猜编码、忽略坏字节、替换题目或直接转写原文件。这些是文件检查数，不是任务数。

入口：`python scripts/prepare_discoverybench_original_assets.py --output <外部新目录>`默认只展示计划；`--execute`从固定官方版本获取并核验，拒绝已有目标；已有原目录用`--verify-only`只读复用，不改原题，不下载替代数据，不执行Agent。保留作者原许可证与归属说明；数据库层面的许可不自动覆盖所有内容权利，不公开原始资产。

原始real任务的只读输入预检入口（不是Agent或评分入口）：

```bash
python scripts/verify_discoverybench_original_contract.py \
  --native-assets <publisher-native-real外部目录> \
  --answer-assets <publisher-answer-keys外部目录>
```

先核验全部185份固定real资产和原real答案键身份，再按原dataset/metadataid/query_id检查原题及精确文件关系，不按CSV行号配答案，不输出参考答案。发现原始问题时退出码1并列出失败引用，不自动改编码、补文件别名或过滤任务。服务器已用该公开入口检查全部239条引用，复现229条关系通过、5条UTF8失败及5条文件关系失败；这是预检真实结果，不是原生执行或来源交付成功。

DiscoveryBench还提供两个官方HF准备profile：向同一准备入口传`--release publisher-task-index`取得固定数据卡及原默认train/test
CSV；`--release publisher-answer-keys`取得原real/synth答案键。默认官方HF通道；服务器原通道明确失败后，显式`--hf-endpoint https://hf-mirror.com`可按同一官方版本、Git
blob和大小校验获取，不带凭据、不用改造数据。`--reuse-assets <已有外部资产目录>`仅复制校验一致的文件到新目录、缺什么取什么，不覆盖旧目录；只读复用仍用`--verify-only`。

当前5份官方文件字节身份已通过：默认CSV原train25、test239行，其JSON查询字段均可解析；real答案键239行可解析，synth答案键仍有原UTF8解析错误，已保留字节与失败，不猜编码或宣称已解决。正式任务/答案/数据文件关系及两个发布版本的一致性还须核验；默认CSV行数不等于Git仓库所有元数据文件或完整synthetic任务数。参考答案只用于评分，不作为模型输入。

官方原生real数据使用`--release publisher-native-real`及固定pins，当前185文件、13825701字节已取得核验；`--reuse-assets <旧Git source目录>`仅复用165份字节一致文件，20份发行版本不同的元数据按HF固定版本获取，不覆盖旧缓存。任务对应使用同一HF版本的原metadata和答案键dataset/metadataid/query_id，不按CSV行顺序配对，不把不同发布版本当作相同对象。

原生输入合同仍未全通：239条原test答案引用中229条通过原JSON题目/数据文件关系及原作者默认提示构造；10条失败引用中5条为原metadata的UTF8解码失败、5条为原metadata引用的精确文件名不存在。后者引用连字符文件名，而目录只有下划线形式，未擅自重命名或补别名。这里未运行Agent、生成代码或评分；不得过滤这10条换题。两个官方版本的20份不同文件，即便做严格、可逆的旧代码页诊断，其完整JSON对象也均不相同；未采用运行时猜编码或转写。安全执行隔离、原生调用/评判及公共/研究消费仍待完成，来源总勾保持未完成。

### Copilot 2026嵌套会话元数据（已交付，非原始请求回放）

- [x] 固定[官方release](https://github.com/Azure/AzurePublicDataset/releases/tag/ghcp-coding-agent-2026)身份400522856及七份原包，共3218886136字节，全部发布方SHA256和大小匹配。后补说明/schema/card另固定文档版本215becda，不假称文档属于旧release标签。
- [x] 公共代码独立核验七日606个原gzip分片、301026个原会话及七份官方manifest，逐片身份、行数、完整嵌套结构和每日总量均一致。最初检查器遇到夹在包中的manifest而停止的失败回执仍保留，已复用原分片纠正检查，不重下或覆盖。
- [x] 公共准备入口完成全部301026条引用索引；七日各固定首片首3/末片末3，共42条候选研究回读，session/turn/call/tool及未知扩展字段均与原JSON一致。相同session_id不去重，使用文件SHA和行号身份，不把元数据变成请求。
- [x] 实际CI提交1923bbfd的49个安装载荷与候选包逐字节一致；复用完整索引，再用实际CI包完成42条固定研究回读和单片首3条公共准备smoke。来源按元数据资产打总勾，0模型调用。[交付回执](validation/copilot-2026-metadata-delivery-20261027.json)

取得/展开计划：`python scripts/prepare_copilot_original_assets.py --output /private/copilot`；加`--execute`才获取七包，已有归档可加`--reuse-archive-dir /private/existing-archives`。已有完整目录只读核验：`python scripts/prepare_copilot_original_assets.py --verify-data-root /private/copilot/data --output /private/fresh-copilot-audit`。后者已在服务器全量完成；公共展开曾对第六日22片/10674会话真实验证，不将该单日检查冒称整套七包下载/展开命令端到端重跑。

统一准备：`python scripts/prepare_reused_source_records.py COPILOT_AGENT_2026 --data-root /private/copilot/data --output /private/fresh-copilot-prepared`。输出只含引用索引与准备证据；原数据留仓库外，失败现场保留。

边界：只有匿名模型标签、token/缓存计数、消息片段类型和工具状态等元数据，没有原prompt、代码、模型响应或工具输出。`request_body()`明确拒绝，不造prompt，不映射真实模型，不将缓存计数推成精确KV块；作者绘图、特征和研究分析不迁入。研究侧固定消费者回读已验证，不表示其默认依赖已自动更新。

### DocVQA原始文档视觉问答（固定原生抽样交付完成）

- [x] 核查仓库现有入口后，取得并按Git blob/大小/SHA256核验维护方Inspect
  Evals固定版本的五份任务/依赖声明文件；固定版本`46e05d2efb7046553a9489329ecdff22f21c6788`，服务器目录`docvqa-original-runtime-discovery-32028-001`。这是原DocVQA验证集路线，不以不同任务的DocVQA2026竞赛替换。
- [x] 后续取得并逐Git
  blob核验固定框架全部4071文件/158827759字节，原zip99837538字节；原`uv.lock`冻结安装的独立CPU客户端101个依赖导入和依赖检查通过，全部原文件和锁未变。只准备API客户端，不安装或执行CPU/GPU模型，0推理调用。
- [x] 官方HF域名连接失败后，通过镜像取得并用Git对象完整性核验固定原commit/tree，读出六个原validation分片的LFS
  SHA256及大小，共1055835028字节；版本仍是维护方固定的`539088ef8a8ada01ac8e2e6d4e372586748a265e`，不是替代数据集。
- [x] 首次原文件下载返回HTTP429，旧失败日志保留；冷却3090秒后单次探测恢复，再分次取得全部六份原validation文件，共1055835028字节，逐份与原Git-LFS大小及SHA256一致，不换题、不换版本。
- [x] 服务器完整检查5349条、9个原始字段、5349个唯一questionId、1286个原图路径及全部PNG字节；固定首三题ID为49153、24580、57349。公共目录已登记原文件身份，新增原字段适配：原图不缩放，answers仅作参考，不当作模型输入。相关55项软件测试通过，不替代真实接入或原生运行。
- [x] 原生工厂首次在Inspect底层缓存写入时返回空间不足，失败日志保留；仅将该进程的XDG_CACHE_HOME指向共享盘新目录、复用HF已取得的原缓存后，未修改的工厂成功加载全部5349题，固定首三题身份及框架缩略图已核验，原六片SHA不变，0模型调用。
- [x] 服务器候选包的公共准备命令完成全量5349条绑定与索引，全部原字段逐条核对一致；六片各固定首3/末3、共36条经固定研究消费者回读，与公共结果全字段一致，原PNG字节及参考答案保真。首个候选检查器误在绑定前直接读raw_source目录的失败现场保留，修正检查顺序后通过，未修改数据或源角色语义。该结果是候选包证据，不冒称实际CI工件。
- [x] 原生固定首三题已完成，仅使用现有NPU
  Qwen3.5-9B/BF16/TP1及16384输出预算，无thinking覆盖。三次原始请求、三条原生评分、样本错误0；ANLS依题ID49153/24580/57349分别为0.75/1/1。这只是固定可用性抽样，不外推全量或模型质量。
- [x] 三条保存结果独立复核完成：由原PNG重建原框架1024缩略图，完整原生model-input图像附件、原模板、原API输出和原ANLS重算一致，新增模型调用0，不清洗答案。官方API日志会把图像base64替换为固定标记；不声称由该脱敏日志证明线上图像字节，只核验日志中完整保留的原生模型输入附件。
- [x] 实际CI提交10b219f6的50份安装载荷与wheel逐字节一致，完整5349条原字段再次比对、公共准备及36条固定实际CI研究回读完成。主CI和CPU
  smoke均成功；复用三条原生推理和保存复核，不重跑模型，按固定原生抽样范围打总勾。[交付回执](validation/docvqa-original-native-delivery-20261027.json)

保存复核入口为`python scripts/verify_docvqa_original_saved.py --run-root <原运行目录> --data-root <原六片目录> --framework-source-root <固定原source> --output <新外部审计目录>`，只核查保存结果，不调用模型。该来源完成不表示全5349题模型评测、作者模型质量复现或控制界面通过。

环境补充：冻结基础环境缺少可选API
SDK，按同一未修改uv.lock中五个版本及制品哈希补齐，执行`uv pip install --python <独立venv/bin/python> --no-deps --require-hashes -r configs/workload-native/docvqa-api-client-requirements.txt`后再`uv pip check --python <独立venv/bin/python>`。服务器检查106个环境包兼容；不修改模型服务环境。运行时将固定Inspect
Evals原source/src放入PYTHONPATH，保留其原资源文件，避免上游非editable安装缺漏数据资源。

原生入口：`python scripts/run_docvqa_original.py --data-root <原六片data目录> --framework-source-root <固定Inspect原source> --output <新外部结果目录> --inspect-cache-root <共享盘缓存目录> --image-cache-root <共享盘图像缓存目录> --endpoint http://127.0.0.1:19466/v1 --model Qwen/Qwen3.5-9B --api-server-pid <已核实API进程PID> --npu-engine-pid <已核实引擎PID> --max-tokens 16384`。沿用原生工厂、模板、图像处理及ANLS；每次传输前等服务空闲且无其他NPU所有者，SDK/框架/样本重试均关闭。原日志缓冲和trace通过进程级XDG_DATA_HOME/XDG_CONFIG_HOME定向本次外部结果目录。

首次两次启动均在模型调用前失败，已保存现场：可选SDK缺失；随后/root独立98GB卷已满，原生samplebuffer默认写入失败。未删卷、未重启，后续只修客户端依赖和本进程外部目录。第三次完成时控制界面提示AF_UNIX路径过长，原生任务仍完成；不声称其交互控制界面可用。

原数据获取使用发布方HF入口：`hf download lmms-lab-encoder/DocVQA --repo-type dataset --revision 539088ef8a8ada01ac8e2e6d4e372586748a265e --include 'DocVQA/validation-*.parquet' README.md --local-dir /private/docvqa/data`。只能选择该固定原验证集；镜像需显式设置HF_ENDPOINT，仍校验相同SHA，不能用同名其他版本替换。

公共准备入口（服务器候选包全量准备及36条研究回读已核验）：`python scripts/prepare_reused_source_records.py DOCVQA --data-root /private/docvqa/data --output /private/fresh-docvqa-prepared`。原文件目录应包含`DocVQA/validation-00000-of-00006.parquet`等六片；准备输出是原位置引用索引，不是替代benchmark。原生执行仍使用固定Inspect
Evals任务的模板、1024缩略图和ANLS，准备完成不等于任务通过。

需明确协议归属：[维护方文档](https://ukgovernmentbeis.github.io/inspect_evals/evals/docvqa/)描述其原生验证集实现及ANLS；其固定代码包含1024像素缩略图和`ANSWER:`模板，这是该框架的原生协议，不得冒称数据集作者原始实现或静默改成其他协议。正式运行前须固定完整环境、原题/原图及该预处理的边界；原数据读取不能用缩略图覆盖发布方图像，也不将发现源码计为任务跑通。

### Azure LMM 2025服务轨迹元数据资产（已交付，非视觉benchmark）

- [x] [微软固定原发布说明](https://github.com/Azure/AzurePublicDataset/blob/215becdacba1ce682c7368642ce97d5a332de7a6/AzureLMMInferenceDataset2025.md)与gzip身份已核验；服务器取得8524196字节原压缩CSV，完整检查100万行四字段。原文件不解压复制到本地或Git；原timestamp字符串和所有数字字符串保持不变，时区未声明，不推断全局绝对时间。
- [x] 实现原`.csv.gz`流式读取及独立`AZURE_LMM_2025`字段契约，保留TIMESTAMP/NumImages/ContextTokens/GeneratedTokens。元数据没有原prompt或图像内容；`request_body()`拒绝，不造prompt、不把它当视觉任务或模型质量benchmark。官方绘图分析不迁入。
- [x] 候选包47安装载荷逐字节一致；公开取得/核验/准备命令已实际执行，100万原行四字段全部保真并建索引，固定研究消费者首3/末3共6条回读与公共消费者一致，6条均拒绝`request_body()`。新增模型调用0、无研究观测/分析迁入。
- [x] 实际CPU CI
  `040046b5`成功，47安装载荷与CI包及全量核验候选包均逐字节一致；复用100万行保真与索引，实际CI首3/末3研究回读6条一致，公开准备另做3条smoke，不冒称重跑全量。以服务轨迹资产打勾，不宣称请求回放或视觉推理。[交付回执](validation/azure-lmm-2025-asset-delivery-20261027.json)

取得/核验入口：`python scripts/prepare_reused_research_assets.py AZURE_LMM_2025 --mode plan`默认离线计划；新外部目录取得原文件用`--mode download --data-root <目录> --github-api`，已有原数据用`--mode verify --data-root <目录>`。准备统一索引用`python scripts/prepare_reused_source_records.py AZURE_LMM_2025 --data-root <原目录> --output <新外部索引目录>`，不下载模型或执行推理。它是服务轨迹资产，不是原始多模态请求回放；发布方描述的按长度/图像数构造流量也不在这里冒称原请求。

### RepoBench原生仓库级代码补全（固定原生抽样交付完成）

评分可复现性：本次实测CodeBLEU0.7的Python结果受Python哈希种子影响，同一批保存输出总CodeBLEU出现19.93/20.13两值，EM/ES及Java分数不变。原生入口现以新评分子进程固定`PYTHONHASHSEED=0`，不修改作者评分代码；旧无种子报告必须显式加`--freeze-scoring-output <新外部评分目录>`由保存审计生成独立评分文件并重复核验两次，原报告保留，不能反复重算挑高分。已完成90题的固定种子评分审计，新增模型调用0；样本Python/Java的EM均为0，负面结果保留，不代表质量达标或作者CUDA基线复现。

- [x] 历史运行中部分保存审计：前43题核验通过、当时索引43仍未知；这一快照保留，其后90题已全部审计完成。旧进程曾沿用初始“输入已准备”标签至终态，公共入口已修正后续状态标签；未为改标签重跑有效结果。

作者模型取得方式：在服务器已有授权的HF客户端环境运行`hf download deepseek-ai/deepseek-coder-1.3b-base --revision c919139c3a9b4070729c8b2cca4847ab29ca8d94 --include LICENSE config.json generation_config.json pytorch_model.bin special_tokens_map.json tokenizer.json tokenizer_config.json --local-dir <外部模型目录>`。固定版本七文件会在原生入口再次核对大小/SHA256；不依赖浮动main，不下载自定义Python代码，不把模型放进Git。主站网络不可用时可显式设置同发布方镜像`HF_ENDPOINT=https://hf-mirror.com`，仍核对同一固定身份。已有模型直接复用，不重下。

保存审计入口：`python scripts/verify_repobench_original_saved.py --source-root <原source> --archive <固定zip> --data-root <原数据目录> --model-root <固定原模型目录> --run-root <已保存run目录>`只重建原题及重算作者评分，不加载模型或调用推理API。运行中可加`--partial`检查已保存前缀，未知/失败尝试保留且不计完成；整项交付仍须全部固定题、原生评分及公共/研究接入核验。不得仅凭这个工具通过或读取成功就打来源总勾。

原生入口：`python scripts/run_repobench_original_npu.py --source-root <原source> --archive <固定zip> --data-root <10原parquet所在目录> --model-root <固定DeepSeek-Coder-1.3B-Base目录> --output <新外部结果目录> --per-stratum 3`默认只展示计划；加`--prepare-only`仅检查资产并构造固定原题，确认研究/NPU窗口后加`--execute --resource-window-confirmed --npu-device npu:1`实际生成和运行作者`eval.py`。原题、prompt、生成token和完整响应只落外部私有目录，已有目录拒绝覆盖，失败不自动重跑。显式BF16/eager是NPU适配配置，不冒称作者CUDA质量复现；原prompt构造、下一行提取、128-token/temperature0.2/top_p0.95采样与EM/ES/CodeBLEU评分不改。

- [x] 官方默认日期区间2023-12-01至2023-12-31（沿用作者UTC端点语义）、五长度档2k/4k/8k/12k/16k，2语言×3设置×5长度共30层各首3题、90原题已在服务器构造并独立复核。去参考字段后prompt与作者完整原行构造完全一致，623至14420
  tokens，加128输出均未超16384窗口；新增模型调用0。`--full`表示此**官方默认过滤配置**下不抽样，不代表覆盖原发布版全部九长度档或复现旧v0论文分支。

- [x] 90题原生NPU生成、token/原prompt/参考关系保存审计及固定种子的作者评分均完成；复用49684原行/60研究回读的实际CI证据并重新核对47安装载荷一致，不重跑原生有效结果。Python/Java的EM均0，原报告及评分不稳定失败保留；不是质量验收通过。[交付回执](validation/repobench-native-delivery-20261027.json)

公开源码入口：`python scripts/prepare_repobench_original_assets.py --output <新外部目录>`默认只展示计划，加`--execute`取得固定作者归档并校验；已有同版本原zip可加`--reuse-archive <zip>`避免重复下载。已有目录用`--verify-only`只读核验，服务器实际12原文件通过；不会修改作者CUDA代码或在此步骤执行模型。

环境入口：`python scripts/prepare_repobench_runtime.py --source-root <原source> --archive <固定zip> --environment <新外部venv> --uv <uv> --platform-python <已有NPU平台Python> --grammar-source-dir <外部语法源码目录>`默认只展示计划，加`--execute`安装；已创建但依赖未完成的venv可显式`--install-dependencies`续装，已有环境用`--verify-only`。此配置显式继承平台torch/torch-npu/Transformers，不修改平台包；作者没有锁文件，不能称官方环境复现。Python/Java语法包0.21.0的官方固定源码通过API取得、按固定大小/SHA256校验后在ARM构建，未换评分器。服务器已实际通过原生模块导入、两语言EM/Edit
Similarity/CodeBLEU小型自一致夹具及venv依赖检查；这些是环境检查，不是原题NPU执行。

公开资产入口复用`python scripts/prepare_reused_research_assets.py REPOBENCH --mode plan`，取得原数据用`--mode download --data-root <外部目录> --max-bytes 1200000000`，已有数据用`--mode verify --data-root <外部目录>`只读核验。默认550MB预算会拒绝意外全下约1.05GB；必要时显式选择同固定版本`--hf-endpoint https://hf-mirror.com`，仍逐文件校验原SHA256。准备统一索引用`python scripts/prepare_reused_source_records.py REPOBENCH --data-root <同原目录> --output <新外部索引目录>`。这两个准备命令本身不下载模型、不执行推理；已在服务器用实际CI工件核验，原生生成/评分另见上述入口与交付回执。

- [x] 固定[作者v1.1源码](https://github.com/Leolty/repobench/tree/e0cfd34c9e7cd8f057efd32b9280b9719ae12a91)，12文件逐文件大小/Git
  blob核验；原生入口为`run.py`和`eval.py`，保留EM/Edit Similarity/CodeBLEU评分，不改为聊天问答。
- [x] Python/Java固定版本10份原始Parquet全部在服务器取得并核验大小/SHA256；完整检查49684行、11原字段及context的path/snippet关系。覆盖2语言×3设置×9长度档共54层；HF主站不可达时使用同发布方固定版本镜像，不改任务。不在本地或Git保存原始数据。
- [x] 实际CI
  `e241accd`的47份安装载荷逐字节一致，全部49684原行字段重建和索引、固定研究消费者60条回读一致；语言及其独立数据版本不丢失，next_line/all_code/gold_snippet_index不混入模型输入。新增模型调用0，不等同原生benchmark跑通。
- [x] 作者示例DeepSeek-Coder-1.3B-Base七个固定文件已核验，约2.69GB权重身份一致；独立venv与ARM语法构建、NPU原生生成及作者评分均完成。作者`run.py`的CUDA绑定由公开NPU入口适配，作者prompt/日期和长度过滤/下一行提取不改；`all_code`、`next_line`和`gold_snippet_index`不进入请求，不改变现有服务环境。

### StructEval剩余渲染/VQA环境（未交付）

- [x] 公开只读入口`scripts/prepare_structeval_rendering_plan.py --data-file <原parquet> --official-archive <固定官方zip> --package-root <原structeval包目录> --output <新元数据json>`已在服务器执行；2035原行、27官方模块身份通过，明确950非渲染/1085渲染行。只输出原协议计数与固定位置，不调用模型或渲染器。
- [x] 现有950条非渲染结果保留不重跑；历史NPU宿主隔离不可用问题由已授权的专用CPU
  Docker工作区承接，不在NPU宿主执行生成代码。当前Matplotlib最小镜像及隔离控制器已实测，其他渲染格式依赖仍待补齐。
- [x] 原Text-to-Matplotlib固定前三题000802/000803/000804已通过生成、官方代码提取、CPU隔离渲染及完整VQA，7/8/8个原问答对保留。6次实际模型调用均stop，生成提示、裁判提示和图片字节已独立核验；使用现有Qwen3.5-9B、显式关闭thinking，不宣称作者模型复现或引擎性能。27个官方模块未改，未增加来源完成数。
- [ ] 提供合格隔离worker后执行原渲染及完整VQA评测。Python/JavaScript/TeX生成代码不能直接在宿主机执行；worker应无网络、凭据或NPU设备，原输入只读、输出限额且有CPU/RAM/PID/时间边界。VQA可用明确配置的本地NPU视觉模型，不默认授权付费judge；渲染成功不能代替VQA完成。

### Mooncake官方存储回放（固定原生抽样交付完成）

- [x] 实际CI工件32e2e975的46个安装载荷和既有35639条原元数据保真/索引证据复用；固定研究消费者首3/末3共12条回读完成，所有原记录拒绝`request_body()`，不编prompt。
- [x] 两份原轨迹共12031/23608条、九份固定官方模块已校验；旧目录七份不同身份模块未修改或冒用。模型名glm5仅指原工具的KV存储布局，不是部署GLM5模型。
- [x] 在已授权、已记录的研究空闲独立I/O窗口，conversation/toolagent各固定原始首3条运行exit0。保留默认512-token页、2000页、单线程、unpaced、single文件、无fsync。每场景原生日志均为3请求/23100
  token/2读页操作/42写页操作，独立保存审计重算并核对覆盖，没有重放模型或磁盘任务。
- [x] 原清理器因官方工具实际多嵌套一层scenario目录而保护性拒删，旧失败回执保留。核对真实路径、所有者、文件类型、单硬链接、inode及无打开FD后，仅删除本次两个明确的生成data.bin并移除空目录；原始数据/源码/日志不动。逻辑文件上限224919552000字节，实际分配4723318784字节，不能把逻辑容量当已回收物理空间。
- [x] 原生日志、原始文件身份、实际CI研究回读及清理证据闭环完成，按官方存储回放资产打勾。[交付回执](validation/mooncake-official-storage-delivery-20261027.json)

准备入口：`python scripts/prepare_mooncake_storage_plan.py --help`；原生执行入口：`python scripts/run_mooncake_storage_original.py --help`。显式指定原工具源码、原trace目录、新测试页/结果目录、容量预算和已确认的独立I/O窗口，先检查计划再运行，不能在研究测量时抢I/O。

范围：仅固定原始抽样，不称全35639条磁盘回放；无原prompt、无NPU推理、无引擎性能结论。完整trace无冲突映射所需约9.37TiB的容量边界仍在，不能由首3条通过外推全量；这次首3条的哈希页号没有触发modulo碰撞。

### LAB-Bench官方执行环境（固定原生抽样交付完成）

原生进展：固定31子集各首3条共93题已完成，Qwen3.5-9B/BF16/TP1，16384输出预算。显式`--disable-thinking`只覆盖本地vLLM模型模板的隐藏thinking，作者CoT提示/图像编码/选项及评分保持不变，不改服务全局设置。93条保存输出的原请求、图像、选项、解析和原评分已完整独立复核，审计0新增模型调用、不换题、不清洗答案；每题前等待服务空闲和无其他NPU占用。这不是作者GPT-4o质量复现。

- [x] [交付回执](validation/labbench-native-delivery-20261027.json)：8类别、93题完整执行及保存结果审计；87条stop、6条原预算length保留。原解析器解析45题，答对14题；其余原输出和各类别分数原样保留，coverage不是解析成功率。完整1967条原字段/图像及索引、固定CI46份载荷和124条研究回读证据已核对复用，不重新推理、不迁入观测或分析。仅验证固定原生抽样可用，不宣称全1967题推理或当前模型达到质量验收线。

- [x] TableQA/ProtocolQA/FigQA/LitQA2各3题共12题保存结果独立审计：从原始任务重建请求、原图编码、选项、官方解析和评分全部一致，新增模型调用0。12条均以`stop`结束，但作者解析器仅解析出1个选项、答对0题；其他输出含括号/尖括号或缺少规定标签，不做事后清洗凑分。官方coverage=1是作者将`None`也视作非“不确定”选项的结果，**不代表解析成功率**。当前模型配置不能据此推荐为质量基线。

保存核验入口：在同一作者环境运行`python scripts/verify_labbench_original_saved.py --source-root <原source> --archive <固定zip> --run-root <外部结果目录> --categories TableQA ProtocolQA FigQA LitQA2`；仅重放已保存响应，不发送API请求、不执行模型生成代码，检查失败/缺失响应会报错，不跳过凑通过。类别完整完成后才核验对应类别；部分审计不打来源总勾。

- [x] 实际CPU CI工件换锁复核：commit `32e2e975`、run
  `40411348417`成功，安装46份载荷与wheel逐字节一致；不依赖候选目录覆盖，全部1967条保留及索引、124条固定研究消费者回读一致重新核验通过。模型调用0，仍须原生推理/官方评分及保存审计，总勾未完成。

- [x] 候选统一适配完整保留1967条原字段并建立索引；图片引用能定位原文件，ideal留在参考侧。固定研究消费者对31个原子集的首3/末1条共124条回读，与公共消费者完整记录一致。该项用已审计provider代码及新候选目录元数据核验，尚待新版本实际CI包换锁复核，也不能代替原生推理。

ARM兼容补充：作者冻结锁安装后，`uv pip check`发现torch缺少setuptools；用同一环境命令的`--install-compat`显式安装`setuptools==75.1.0 --no-deps`，不改作者源码/锁及其他依赖。实际复核123个已安装包、锁内版本不符0项、依赖检查通过；此补充不是作者原锁的一部分。重复`uv sync`可能移除补充，之后须重新核验。先前版本检查脚本误把作者锁内多版本分支压成单值的问题也已修正，环境核验不代表模型推理完成。

环境入口：`python scripts/prepare_labbench_runtime.py --source-root <原目录/source> --archive <原目录/publisher-original.zip> --environment <独立venv目录> --uv <uv可执行文件> --python <Python3.12可执行文件>`默认展示计划，加`--execute`执行作者`uv sync --frozen --no-dev`，不更新锁；已有环境用`--verify-only`。核验隔离venv、原始源码、作者包导入、已安装包与锁的版本和`uv pip check`。已测试的平台为Linux
ARM64/Python3.12，无CUDA推理；模型仅通过另一个NPU服务入口执行，不在客户端加载权重。

资产入口：`python scripts/prepare_labbench_original_assets.py --output <仓库外目录>`默认只展示计划；加`--execute`按固定官方API归档下载并安全解包，或加`--reuse-archive <已有zip>`复用校验一致的归档，避免重复下载。已有目录使用`--verify-only`只读核验，服务器已实际验证553文件通过、0模型调用；不覆盖既有目录。原归档245275217字节，解包原文件283413040字节，均仅放外部目录。固定归档身份若变化应重新审计，不能跳过校验。

- [x] [官方固定版本](https://github.com/Future-House/LAB-Bench/tree/998a8e0a40cf116c80e1b0e7a805ebb5fb9fa838)的11份运行代码/说明/依赖文件在服务器按大小和Git
  blob校验；未下载任务数据，未调用模型。
- [x] 确认原生入口为`labbench.Evaluator(Eval.<category>).score_agent(agent_fn)`；按原选项精确比较，保留“不确定”选项，报告accuracy/precision/coverage。八个官方类别须分别保留，不能只做纯文本类别就称全部完成。
- [x] 官方固定归档已完整取得，553文件/283413040字节逐文件大小及Git
  blob一致，原任务和图片保持原样；清点公开多选题1967条、31子集（更正此前32的计数）。按原`uv.lock`的独立环境安装exit0，不修改推理服务或作者锁文件。
- [x] 官方环境导入、全部1967条原任务加载和93条固定样本的原生输入构造通过，图表原图可读取；沿用作者对ProtocolQA/SuppQA的上下文拼接和原选项随机化，不把question单字段冒充完整输入。ARM客户端torch无CUDA，推理只走NPU服务。
- [ ] 各类别固定抽样的官方推理/评分及公共/研究接入核验。输入构造成功不算模型执行；模型及预算仍须明确记录，不宣称复现官方模型质量。

公共原生入口：`scripts/run_labbench_original.py --source-root <原仓库解包目录> --archive <固定官方zip> --output <新外部结果目录> --endpoint http://127.0.0.1:19466/v1 --model <本地NPU模型别名> --max-tokens <明确预算> --per-subset 3`；全量改为`--full`，可用`--categories`分批运行，不覆盖已有结果。运行于作者`uv.lock`的独立环境，程序先核验固定归档和全部553原文件，再复用作者OpenAI
agent、CoT模板、图片编码、选择解析及Evaluator评分。固定随机种子只保证选项排列可复现，不改原题；调用请求/响应、作者task
buffer和评分结果只保存在外部私有目录。声明的Qwen9B原生配置已开始执行及官方评分，完整93题和保存审计尚未完成；不把运行中状态或首个类别成绩升级为来源交付。

### OpenThoughts3原对话语料（固定范围交付完成）

- [x] code/math各首3条共6条原题执行完成并独立保存审计通过；全部以`length`耗尽32768-token预算，不能当作最终答案或质量成功。原题/teacher分离、token计数和原文件身份核验通过，审计新增模型调用0。

- [x] science首3条独立续跑并完成保存审计，同模型/原题/32768预算不变，前6条未重跑。原队列父进程消失原因未确定，旧现场保留；science三条也均为`length`，不是最终答案质量成功。

- [x] 沿用已登记发布方固定版本，120原分片/28188197544字节的身份全部登记。它是作者发布的合成推理训练语料，不冒称真实用户请求、独立质量评测集或作者训练复现。

- [x] 生成前固定原分片0/25/110，均取得并二次校验；完整检查30000原行及4字段，实际code/math/science各10000行，均为原human/gpt两轮。首分片用NFS硬链接复用，未重复下载/复制286MB。[准备回执](validation/openthoughts3-original-readiness-20261027.json)

- [x] 复用已有原资产获取、绑定/索引及历史对话续写入口；原`conversations/from/value`保留，只在API传输视图明确映射human→user、gpt→assistant，原问题不改，原teacher回答不进入对应目标请求，也不当质量金标准。

- [x] 固定三域各3条原题共9条NPU续写及保存审计完成；最终实际CI工件46载荷逐字节核验，30000原行的全部字段/前缀视图/索引一致，固定研究消费者18条回读一致。按原语料续写范围打勾，其他117分片仍仅登记，不宣称全120万条执行或独立质量benchmark。[交付回执](validation/openthoughts3-native-delivery-20261027.json)

- [x] 三个固定分片完整30000行已通过公共入口绑定并生成统一索引；固定研究消费者对每片首3/末3条共18条回读，与公共消费者完整记录一致。原human/gpt字段保留，原teacher回答只在参考侧、不混入请求。此项为接入核验，不替代原生续写和保存审计。

执行边界：首个code原题300秒客户端读超时、无有效保存回答，旧失败保留；后以1200秒等待重试原题，题目/预算不变。九条有效保存结果全部耗尽预算、没有最终答案，不能写成答案正确或推荐质量基线。生成时复用已验证45载荷的旧CI包；最终数据/研究交接另以实际CI的46载荷工件复核，未为换包重跑模型。

获取先用`scripts/prepare_reused_research_assets.py OPENTHOUGHTS3 --mode plan --files <原分片路径>`查看；显式`--mode download --data-root <外部目录> --files ... --max-bytes <预算>`执行，`--mode verify`只读核验。默认550MB保护不允许意外全下28GB；同一固定版本需要镜像时显式`--hf-endpoint https://hf-mirror.com`，原SHA256仍必验。绑定沿用`scripts/prepare_reused_source_records.py OPENTHOUGHTS3 --data-root ... --output <独立目录> --files ...`。原始数据/长推理回答不进入Git、wheel或本地PC。

### SchemaStore原始schema资产（交付完成）

- [x] 核查官方定位与原生校验方式：这是schema集合及原正/负例，不是用户请求/模型质量benchmark，不造prompt冒充原请求。官方流程为`npm clean-install`后`node ./cli.js check`，保留原schema、引用和正负例。[官方说明](https://github.com/SchemaStore/schemastore/blob/b3c6a70f2a535d14e8ad2bf460b3aa8f5c0efb27/CONTRIBUTING.md)
- [x] 固定版本原归档通过官方API取得；4810文件逐文件大小/Git blob校验及实际根tree独立一致性核验通过，含1006
  schemas、2286正例文件、1402负例文件。Git连接失败历史保留；只有资产取得，不是原生校验完成。[准备回执](validation/schemastore-original-readiness-20261027.json)
- [x] 独立Node22.14.0
  ARM64归档与官方发布校验和一致，公开`scripts/prepare_schemastore_runtime.py`实际执行exit0，npm10.9.2；原锁`clean-install`190包成功，没有系统安装或更改推理服务。锁中安装hook为0，显式禁用hook执行。
- [x] 官方完整CLI校验exit0，pre-checks和Ajv validation均完成；执行前后4810原文件及重建Git根tree一致。报告实际测试976
  schemas，742有正例、289有负例，未改作者筛选规则，不冒称独立验证全部2681个catalog外链或全部1006原JSON文件。
- [x] 公共获取/原归档复用入口和数据组件读取、共享索引、fixture原字节读取已实现，纯数据边界与夹具检查通过。数据包不导入Node、评分器或观测依赖，原schema对象不改，正负例单独作为原验证资产引用。
- [x] CI实际工件45文件一致；全1006原schema、3688正负例引用及索引核对，固定研究端16处回查和70份原例子字节读取通过。[最终回执](validation/schemastore-native-delivery-20261027.json)。资产可用不等于模型质量/性能结论，原正负例不生成prompt。

数据获取：`python scripts/prepare_schemastore_original_assets.py --output /external/schemastore`默认只规划；`--execute`才获取固定官方归档。已有同一原归档可加`--reuse-archive /external/original.zip`，先验身份再复用；`--verify-only`只读校验。实际数据根为`/external/schemastore/data`，不把原schema/例子复制进仓库。

公共使用：`iter_workload("SCHEMASTORE", data_root=Path("/external/schemastore/data"))`返回1006份按原schema路径排序的资产；`record.task["input"]["schema"]`是原JSON对象/boolean，`record.outcome_record()`的`positive_examples`/`negative_examples`保持官方目录关联规则和原路径/字节数/校验和。用`vllm_hust_workload_data.schemastore.read_original_fixture(data_root, record.source, reference)`读取不改写的JSON/YAML/TOML等原字节；不默认联网解析`$ref`、不生成prompt、不能调用`record.request_body()`冒充原始请求。共享准备入口仍按既有`sources:[{source_id,data_root}]`配置，不需科研专用配置。

准备运行时：从`https://nodejs.org/dist/v22.14.0/node-v22.14.0-linux-arm64.tar.xz`取得原归档，再用`python scripts/prepare_schemastore_runtime.py --archive /external/node-v22.14.0-linux-arm64.tar.xz --output /external/private-node --execute`校验并解包。需要Python3.12+的tarfile数据过滤器及lzma模块，不依赖系统xz程序；本服务器初次系统tar解包因缺xz失败，改用标准库解包，原归档不变。校验入口默认只规划，显式提供`--source`原源码、`--original-tree`官方完整recursive
Git
tree元数据、`--node`私有node路径、`--output`独立日志目录和`--execute`才运行`node cli.js check`；源码/例子全保留，不以自定义JSON校验替代官方流程。

### NarrativeQA原始summary-QA（来源交付完成）

- [x] 原三表关联已接入共享`iter_workload`、准备索引及公共/研究读取协议；保持问题原物理顺序，两个原答案及tokenized参考与输入分离，文档/摘要原字段独立保留。433项组件/合同测试及代码对应两项CI通过；服务器精确工件审计通过。

- [x] 复用原3表/46765问答关系，不重新获取数据；明确summary-QA与full-story不同，不拿summary替代缺失全文。

- [x] 固定AllenAI评分源码/许可证并在服务器获取、二次只读核验；原BLEU-1/4、METEOR、ROUGE-L接口在隔离py-rouge1.1及固定WordNet下通过合成夹具预检。[评分身份与差异](../../configs/workload-native/narrativeqa-metric-pins.json)。不是作者官方聊天运行器或论文数值复现；该实现的BLEU-4权重及逐参考取最大/两位舍入明确保留。

- [x] 原始test物理顺序前三题已生成、双参考评分与保存审计exit0；3条stop，完整输入695/698/700
  tokens。9B非思考/256输出预算为声明式适配，不冒称作者配置；审计零模型调用。[阶段回执](validation/narrativeqa-summary-native-checkpoint-20261027.json)

- [x] 全46765条原问答、1572组文档/摘要关系和全部原字段、统一索引核对；精确CI44文件一致，固定研究端18处回查包含已跑3题。旧单表映射不再阻塞原CSV使用，未改默认研究依赖或增加模型调用。[最终回执](validation/narrativeqa-native-delivery-20261027.json)。只验summary-QA，不声称full-story/全文检索或原模型复现；初次WordNet缺失失败保留，未降级METEOR。

原任务入口：`python scripts/run_narrativeqa_original_summary.py --data-root /external/narrativeqa-original --metric-source /external/narrativeqa-metric/scoring/narrativeqa.py --output /external/narrativeqa-run --execute`。默认原test前三题，`--full-test`才跑全10557题；保持完整摘要与原问题，不包含参考答案。默认256输出token为适配预算。审计用同参数，另加`--reuse /external/narrativeqa-run`并将输出改成新的`--output /external/narrativeqa-audit`，不调用模型或覆盖旧结果；依赖环境同下段。

评分源准备：`python scripts/prepare_narrativeqa_original_metric.py --output /external/narrativeqa-metric`默认只规划，显式`--execute`获取，`--verify-only`只读校验。`py-rouge==1.1`必须安装在独立target/环境，用`PYTHONPATH`选择；不要覆盖其他任务正在使用的`rouge==1.0.1`。`NLTK_DATA`显式包含已有punkt/punkt_tab及固定WordNet目录；运行评分禁止隐式联网下载。现已验证环境为Python3.12.13、NLTK3.10.4、py-rouge1.1；这不是原2018模型环境复现。

### LongBench V2容量更新（来源仍未完成）

- [x] 既定Medium原第3/5/8行，用当前9B原tokenizer和256K部署窗口重新核验，完整输入149662/138653/68762
  tokens均容纳原128输出预算；未截断、未换题、无模型调用。[容量回执](validation/longbench-v2-medium-9b-capacity-20261027.json)
- [x] 同三题已原生执行/评分完成exit0：完整输入149662/138653/68762
  tokens，原128输出预算保留；length/stop/stop，正确1/3，不换错题或补偿分数。请求/响应保存在服务器。
- [x] 独立保存结果审计exit0，逐题核对完整作者提示、原参考/字段、响应usage及作者extract_answer评分；未调用模型/网络/judge，未修改原结果。可用`scripts/verify_longbench_v2_original_saved.py --help`查看外部路径参数，默认只规划，显式`--execute`执行。
- [ ] 精确工件/研究端闭环待补；原Short有效结果不重跑，Long组仍超当前窗口，来源不增加总勾。

### PAWS分步骤（来源交付完成）

原生运行及保存结果复核见[阶段回执](validation/paws-native-checkpoint-20261027.json)；不以阶段完成替代来源交付完成。

- [x] 复用已登记的发布方HF版本身份，6个原Parquet全部取得、逐文件字节数/SHA256二次校验；751450行、4字段、三配置及其原split数量与发布方一致。[准备回执](validation/paws-publisher-readiness-20261027.json)
- [x] 默认只规划、显式获取/恢复/只读校验入口及5项CPU合同测试；原数据只留服务器。旧GCS403记录保留，不声称与旧tar逐字节相同，不新增QQP/PAWS-X来源。
- [x] 原释义判别任务执行与适用评分已完成并离线复核；人工标注final test、人工swap辅助训练与noisy辅助训练分别统计，不混为同一人工测试集。
- [x] 已补声明式生成模型接入与离线原句对/响应/评分复核入口，配套CPU夹具通过；输入不含原ID/标签，严格0/1输出不修补，噪声标签只算一致率。软件能力不等于原生运行终态。
- [x] 全8000条人工test＋其他5文件各原前三，共8015题已执行并保存，全部stop、无非法输出；正式test正确6774/8000（84.675%），仅报告该模型配置结果，不设质量通过结论。保存结果复核无新增模型调用；来源总勾仍等待下项闭环。
- [x] 751450条原记录4字段及全量索引核对、精确CI工件44文件一致、固定研究消费21处回查完成；无新增模型调用。[最终交付回执](validation/paws-native-delivery-20261027.json)。21处回查不冒称逐一覆盖全部8015次推理。

### TraceLab分步骤（来源总勾未完成）

- [x] 复用665453轮全部元数据、全量索引、原DuckDB及官方确定性CSV转换、独立enwik9文本身份；不是原始用户prompt正文。
- [x] 官方Rust1.90归档和校验文件核对，恢复到持久化私有前缀；公共构建入口实际按原锁编译exit0，原源码不变，避免下次容器切换丢失二进制。[恢复回执](validation/tracelab-original-locked-build-20261024.json)
- [x] 公开原生回放入口及7项离线合同测试：默认只规划，不绕过原缓存资格探测，超上下文在调用前阻塞、不截断或跳过，核验全部原session/轮次/预算/工具等待。
- [ ] 官方原生回放：已到达原缓存资格探测，但512-token探测在当前1024-token
  Mamba对齐缓存块下未命中，原程序exit1，20轮任务均未开始。[失败证据](validation/tracelab-original-dry-run-20261024.json)。缓存相关开关已开启，不能误称未配置；未改探测、换session或重启服务，待合适模型/配置后续验证。
- [ ] 保存原生日志/summary独立复核与公共提供者/固定研究消费交接；不迁入研究采集和分析，不报告性能提升。

### MMMU分步骤（固定原validation抽样交付完成）

- [x] [最终交付回执](validation/mmmu-native-delivery-20261027.json)：11550条原字段/索引完整核验；90道原题推理、原评分和独立离线复核通过；成功CI精确软件包与固定研究消费者152条往返通过，包含全部90道已运行题。非全量推理或质量结论。

- [x] 官方固定版本30学科、92个原Parquet分片的路径/容量/SHA256完整登记，包含两个双分片test学科；不把原题和图像写入仓库。

- [x] 默认只规划、显式执行/匹配恢复/只读校验入口与5项离线合同测试通过。

- [x] 服务器92/92原始分片取得、全部字节数/SHA256与第二次只读校验通过，原数据不落本地；[准备回执](validation/mmmu-original-readiness-20261026.json)。直连网络不可达后以相同身份镜像恢复，没有绕过校验；不是原生benchmark完成。

- [x] 官方源码Git
  tree核验；30学科validation各原0/1/2共90题（含4多图）NPU执行与30份原评分通过，公共保存结果独立审计通过、0新增调用；原128预算下59次length、31次stop保留，不加预算或换题，不冒称LLaVA复现/全量质量通过。

- [x] 复用已有MMMU原生字段适配，更新92文件身份并保留旧登记revision；全16字段/11550行及索引完整，答案/解释与输入分离。精确CI的44载荷逐字节核验、152条公共/研究往返通过，未迁入研究观测或分析。

当前机器清单计算：**57/113项已交付，56项待补**（另5项仅分类，不计入分母），已超过一半。新增LAB-Bench固定93题原生执行、原评分及保存审计，复用已核的公共/研究接入证据；不是作者模型质量复现。Azure
LMM
2025仍仅按服务轨迹元数据资产交付，不是视觉benchmark或原始请求回放。RepoBench样本EM均0、OpenThoughts3九条预算耗尽的边界不变。以下历史统计保留其原时点，不代表当前总数。

### ChartQA分步骤（来源总勾已完成：固定原测试抽样交付）

- [x] [最终交付回执](validation/chartqa-native-delivery-20261026.json)：原测试2500条全字段/图像字节及索引完整核验；两个类别各固定前三条共6条原生NPU运行、原评分、公共离线复核，以及CI产物到固定研究消费6条往返均通过。非2500条全量推理或性能结论。

- [x] 新实例复用已验证编译产物，正确私有venv内Decord导入、隔离Python `pip check`与132包版本快照通过；修复后构建器尚未完整重新编译，不能冒称已验证全新端到端安装。

- [x] 旧构建终态已取回：编译成功，但旧版解释器符号链接缺陷导致Decord安装到旧容器系统Python，随后pip
  check失败；不是“系统未修改”。新版前缀保护已修，新实例根环境未发现Decord，未擅自全局回滚。

- [x] 复用原测试分片与已有前三条图像/问题的CPU处理证据；不重复下载，不冒充推理。

- [x] 固定VLMEvalKit版本`160e8a7d`，服务器897文件重建Git tree与原版本完全一致；未修改原提示、评分或运行器。

- [x] 原HF测试2500条无损转为工具输入视图；完整问题、答案列表、顺序、类别与图像原字节逐项相等，未声称是预设TSV的同一文件。

- [x] ARM官方Decord0.6.0源码及两个原子模块树身份核对、私有FFmpeg/Decord编译与导入通过；原失败记录保留，不改系统库。

- [x] 完整原生模块导入；原始0/1/2三条真实视觉NPU生成、原评分与独立复核通过。Qwen3.5-9B
  BF16/TP1显式非思考、原API封装默认预算2048，三条stop、抽样评分66.67%；仅原类别0，不外推其他类别或完整测试集。

- [x] 通用公开原生执行入口、固定原始位置/全量模式、原请求与响应保存、显式非思考参数及CPU smoke合同测试已补；此勾仅表示软件能力，不代表新增服务器验证。

- [x] 已保存结果的公共离线复核/原评分入口与CPU合同测试已补，禁止模型/judge调用、改写原运行或把缺失任务变为通过；公共命令的服务器真实复核仍待恢复后执行。

- [x] 解释器入口防误解析、真实虚拟环境身份/禁止继承系统包、pip隔离与显式安装前缀保护已补，真实临时venv回归通过；不是对旧服务器安装影响的核验结论。

- [x] 旧构建终态与影响已核查，见上述事故说明；不再沿用“影响未知”。

- [x] 私有纯客户端安全恢复、隔离导入和pip check、完整版本快照通过；全新重编译范围另列未验证，不替换旧失败记录。

- [x] 公开入口完成未运行过的原1250/1251/1252，原类别1、三条stop、原评分100%；已有0/1/2只离线复核、不重新生成。

- [x] 最终公共数据/研究消费审计通过，2500条原字段/索引与6条固定往返闭环，CI安装44文件逐字节相等；原始数据未公开。

### LongMemEval-V2分步骤（来源总勾未完成）

- [x] 固定官方53文件源码身份，服务器只读verify成功；不套用旧LongMemEval协议。
- [x] 公共数据准备入口、原41文件身份与容量预算已补，调用未修改作者脚本；软件夹具通过。
- [x] 原41文件完整下载与发布方内容身份校验通过，7120369667字节全部留服务器。
- [x] 截图展开及small/medium官方完整性校验均exit0：每档451问题、1870轨迹、451问题映射，未关闭截图检查；见[原始准备回执](validation/longmemeval-v2-original-readiness-20261026.json)。
- [x] 官方9B完整16资产校验、空闲窗口服务切换及HTTP健康检查通过；BF16/TP1/262144上下文，NPU3运行。配置支持图片不等于真实视觉题已验证。
- [x] 原enterprise/small前三题`no_retrieval`负对照完成NPU生成、原本地评分与独立复核；thinking开启、原预算20000保留，三个0分不删改；不冒称正向记忆检索。
- [x] 独立HTTP客户端依赖锁、隔离后pip check与两份原CLI导入预检通过；RAG图像预处理的PyTorch另列未完成。
- [ ] 原生记忆方法执行与原评分；视觉reader、RAG embedding及judge条件仍未齐备。
- [ ] 公共数据关系适配与固定研究消费闭环；准备成功不等于原生benchmark交付。

### InfiniteBench分协议进展（2026-10-28；来源总勾仍未完成）

- [x] 全部12份原文件校验，2499838795字节；正确旧文件复用，未重复下载。
- [x] 全3946条记录引用准备：数值答案数组800条、字符串答案数组3146条，不强制转类型。
- [x] 全3946条原字段、答案类型、文件/行关系与索引核对；每文件原前3条共36处公共/固定研究回查通过，研究默认依赖未升级。
- [x] `code_run`原始0/1/2：原预算5，NPU执行与作者评分、保存结果独立复核通过。
- [x] `passkey`原始0/1/2：原预算6，同上。
- [x] `number_string`原始0/1/2：原预算12，同上。
- [x] `longbook_qa_eng`原始0/1/2：原预算40，同上。
- [x] `longdialogue_qa_eng`原始0/1/2：原预算40，同上。
- [x] `math_find`原始0/1/2：原预算3，同上。
- [x] `math_find`额外固定原题3–5与6–8：完整原输入约11.6–11.7万token、原预算3，六次生成及作者评分独立复核通过；未重跑0–2，不扩大整来源完成结论。
- [x] `math_find`额外固定原题9–11：三条原输入分别116865、116700、116762
  token，原预算3；已保存生成与作者评分1.0，独立回查请求、答案、用量及摘要一致。此次回查未调用模型，不扩大整来源完成结论。
- [x] `kv_retrieval`原始0/1/2：新9B服务262144上下文、BF16/TP1，逐请求显式非思考；完整输入169100/169118/169254
  token，原预算50，三条stop、作者评分1.0，独立回查通过。旧27B/131072阻塞记录保留，不混作同一模型的性能结果。
- [x] `code_debug`原始0/1/2：新9B显式非思考，完整输入214956/214962/214964
  token、原预算5，三条length、作者评分0，生成和评分独立回查通过；不加预算或换题凑分。
- [ ] `longbook_sum_eng`：当前9B原前三题完整输入402028/88408/240818
  token，保留原输出预算1200时至少需要403228上下文；当前262144不能覆盖固定样本。不跳过第一题或截断。
- [ ] `longbook_choice_eng`：当前9B原前三题268007/268000/268007输入token，原输出预算40，固定三题至少需要268047上下文；当前262144不足。
- [ ] `longbook_qa_chn`：当前9B原前三题339312/339331/339312输入token，原输出预算40，固定三题至少需要339371上下文；当前262144不足。
- [x] `math_calc`旧9B兼容性核查：原前三题均以原`system,user,system,user`消息顺序在tokenize阶段被拒绝400，失败保留；未删除/移动后置system或更换作者模板，后续用14B保持原协议完成验证。
- [x] `math_calc`新增14B绑定的原样预检通过：同一原前三题的消息角色不变，输入54280/54301/54350
  token，加原30000输出预算均在当前131072窗口内。预检零生成，旧9B拒绝记录保留。
- [x] `math_calc`14B原生固定三题完成：原第0/1/2题全部返回，均保留原30000-token输出及length终态。原问题/提示/请求、保存哈希、原数据及官方/独立计分一致，三题及聚合得分均0，不重跑凑分。完整只读审计回执为共享任务根`infinitebench-mathcalc-complete-audit-20261031-001/qualification-public.json`，新增模型调用0。复用此前合格的隔离评分环境、原ROUGE缓存和客户端组件，未重装或修改全局服务环境；第一次缺少`evaluate`的模型调用前失败保留。实际执行目录为`infinitebench-mathcalc-14b-native-20261031-002`，原进程已结束，不再标为在途。未改原角色、题目、输出预算或模型自动重试；与其他正确性客户端共享过服务，不用于性能测量。此为新增一个子协议完成，来源总勾仍待三个长书协议。

本次只读预检12次tokenize、0次生成，原代码/数据身份核验通过，服务器回执为`infinitebench-remaining-9b-original-preflight-32028-001/receipt-public.json`。上述上下文需求只是当前固定三题的token算术下界，不代表扩大服务上下文即可部署或已验证硬件可行性；未重跑已有原生结果，来源总勾仍未完成。

当前为9/12子协议、27道原题经原生执行及保存结果审计：旧27B六协议18题、9B两协议6题及14B数学协议3题分别保留，不合并为统一模型质量或性能结论。旧11次、9B新增3次及14B新增3次length终态和低分均保留，不加预算重跑。三个长书协议仍未完成，不给来源总勾。既有阶段证据见[原复用审计](validation/research-acquisition-reuse-audit-20261025.json)，新增数学完整回执见上项。

### StructEval分步骤核验（2026-10-26；不增加来源完成数）

- [x] 公共审计入口在服务器重评950条历史原题输出，19个非渲染协议各50条；官方分数一致，新增模型调用0次。
- [x] 完整2035条记录及索引、每条10个原字段和参考隔离核验通过。
- [x] 44种原协议组合各取原始前3条，132处公共入口与固定研究消费者回查通过；研究默认依赖未改变。
- [x] 2026-10-29完成Text-to-Matplotlib全50题原生流程：前三题复用，新增47题，历史950题未重跑。总计99次模型调用均stop；49题完成渲染及完整VQA，000815保留原代码AttributeError的渲染0分，按作者缺图规则跳过裁判，不换题、不重生成。原解析代码、完整问答与裁判图片字节已核验。
- [x] 原Typst模块在专用CPU隔离容器中通过正常1/非法语法0两个控制，非原题、不算原题验证。
- [x] Typst公共分派通过正常、语法错误、多页返回1但指定图片缺失及Matplotlib兼容性四项实测控制；保留原返回/缺图语义，不改题或补造图片。
- [x] 原Typst固定前三题001401/001402/001406已生成并由原流程评分，三次stop、渲染均0、原最终分数均0.1。原输入与解析关系已核验；代码使用Markdown式标题，编译诊断为语法错误。仅复用失败代码补查编译stderr，未重新生成或覆盖原分数。
- [x] Typst全部50道原题执行及条件评分已审计：复用前三题，新增原位置3..49的47次生成均stop，原解析/隔离渲染返回及全部原字段核验一致。50题均render0，最终分数48题0.1、1题0.09、1题0.07；按作者缺图规则跳过全部视觉裁判（实际0次）。新增47份保存编译日志均为语言错误，47份代码均带Markdown式标题、无外部import；不改代码或为提高分数重新生成。已核对执行数包含这些原失败结果，不代表成功答案或来源交付。[50题阶段证据](validation/structeval-typst-full50-audit-20261031.json)。
- [ ] Typst成功图片及实际视觉裁判分支仍未建立，不能把条件跳过或渲染控制夹具说成该分支已验证。此前更强27B设想未执行；当前9B输出不满足Typst语法，失败保留。
- [x] 原Text-to-SVG固定前三题001200/001201/001202已完成原生成、官方解析、隔离浏览器渲染与完整VQA；7/8/10个原问答对保留，6次模型调用均stop，输入/原提取代码/实际裁判图片字节已核验。原最终分0.9/0.65/1.0如实保留，不作模型或引擎性能结论。
- [x] SVG入口通过内联图、空输入、外部HTTP资源阻塞和Matplotlib兼容性四项服务器控制；外部资源缺失不伪造模型0分。
- [x] 公共浏览器配方实际构建退出0，在非root/离线/只读容器中完成7个扩展依赖和浏览器资产核验，四项隔离控制另行通过；原题使用的准备镜像与公共配方镜像分开记录，未为了公共配方重跑有效原题。
- [x] HTML、Canvas、Markdown共用隔离分派与原相对导入；10项CPU控制通过，覆盖正常、空输入、内部HTML调用中的外部资源阻塞及SVG兼容。Markdown仅在既有浏览器镜像上增加观察版本3.11.2，不重下浏览器。控制夹具不计原题或来源完成。
- [x] HTML、Canvas、Markdown六组协议各固定前三题已完成原生生成、隔离渲染和完整VQA；18道原题、36次stop调用、158对原问答保留，十个原字段、实际提示、原提取代码与裁判图片字节独立核验。Canvas三题VQA均0、原最终0.29/0.3/0.3如实保留，不换题凑分。
- [x] Text-to-HTML全50道原题已完成；前三题复用、新增47题，合计100次调用均stop，345对原VQA、原字段、提示、提取代码和实际裁判图片字节独立核验。
- [x] Markdown-to-HTML全50道原题已完成；前三题复用、新增47题，合计100次stop调用、439对原VQA，原字段、提示、提取代码和实际裁判图片字节独立核验；原转义警告及回退保留，不修补生成。
- [x] React-to-HTML全45道原题已完成；前三题复用、新增42题，90次stop调用、408对原VQA、十个原字段/提示/提取代码/实际裁判图片字节独立核验，不重跑有效前三题。
- [x] Vue-to-HTML全40题已完成；前三题复用、新增37题，新增74次stop调用和317对原VQA，原字段、提示、提取代码和实际裁判图片字节独立核验。
- [x] Text-to-Canvas全50题已完成；前三题复用、新增47题，新增94次stop调用和374对原VQA，低分原样保留，不修题或更换样本。
- [x] Text-to-Markdown全50道原题完成；前三题复用、新增47题，新增94次stop调用、377对原VQA，原字段、提示、提取代码和实际裁判图片独立核验，全部六组浏览器协议原队列正常结束。
- [x] 原LaTeX/TikZ渲染器公共分派和配方已通过五项隔离CPU控制：片段、完整文档、LuaLaTeX、缺失文档类0/缺图、Matplotlib兼容。非root/只读/断网软件核验通过，原dpi300/45秒预算、引擎回退和首PDF页转换不改。
- [x] Text-to-Latex、Text-to-Tikz、Matplotlib-to-TikZ各固定原前三题完成原生流程和独立审计；9次生成/8次实际裁判、17次调用均stop。001303原渲染0/缺图/最终0.1如实保留，不补造裁判调用。
- [x] Text-to-Latex全50题完成原生执行和独立审计：复用前三题，新增47次生成/47次完整VQA，共94次stop；47张实际裁判图片和343对新增原问题、原提示/解析代码/评分关系均核验。审计0次新模型调用。[三组增量进度](validation/structeval-latex-full-protocol-progress-20261031.json)。
- [ ] Text-to-Tikz新增47题运行因传输超时终止：34份成功生成保留，第35个逻辑请求遇到SDK内部2次重试后失败，第36个请求取消/结果未知。未重放这些请求，未把超时填成0分；Matplotlib-to-TikZ剩余97题尚未启动。已有6道原题继续复用。
- [x] 修正外层重试0未传到SDK的接口缺陷：新入口显式关闭内部重试，提供独立`--request-timeout`和外层`--max-retries`配置，不改原题、生成token预算或模型精度。真实客户端对本地503软件夹具，旧配置3次HTTP请求、修复配置1次，0神经模型调用；回执为共享任务根`structeval-sdk-retry-fix-32068-001/qualification-public.json`。失败原运行不改写；输出预算变更尚未执行。
- [x] Text-to-SVG全50题已完成原生生成、隔离渲染及完整VQA：前三题复用，新增47题，合计100次模型调用均stop；原字段、原提取代码关系、397个原问答对和实际裁判图片字节已独立核验，未重跑有效前三题。
- [ ] 其他15种部分或尚未覆盖渲染协议及完整来源交付；当前累计1441道原题真实执行和保存结果核验（含上述50道失败分支），剩余594道。上述另两组144题尚未完成，不提前计入；失败目录为共享任务根`structeval-original-latex-window-32068-001`。来源仍未打勾。

以上使用[原阶段回执](validation/structeval-nonrender-checkpoint3-20261025.json)记录；前三项是证据复用与数据链路核验，不是整套原生benchmark完成。

### Spider原数据与官方测试套件（固定原生抽样交付完成）

- [x] 复用此前固定官方协议来源记录；原作者GoogleDrive数据包已在服务器取得，205800266字节、2234文件、ZIP
  CRC通过，Windows无原数据落盘。SHA256仅为接收端观察值，不冒称发布方校验和。

- [x] 官方Test Suite数据库原包已取得，1269456098字节、3944文件、全部ZIP CRC通过；原包保留，未重新下载已完成数据包。

- [x] 两原包在新目录完成受控解压，6178文件/约6.99GB；原包不变。全11840条原记录与原schema/SQLite文件关联核验，372份原数据库文件头/哈希核验；166/206两个原目录同名DB不合并。

- [x] 独立CPU评分软件环境实际构建，并通过非root/只读/断网检查；固定NLTK官方分词资源离线安装，不将依赖准备算作原SQL、模型或评分完成。

- [x] 公共准备入口在服务器绑定dev原前三题、隔离问题/schema与七字段原参考记录，并核对完整32份官方测试数据库；原评分器参考控制在隔离容器运行，3题Test Suite
  Accuracy为1.000。此为参考控制，不是模型成绩或来源完成。

- [x] 物理SQLite schema、原SQL及固定dev前三题的实际官方评分已核验：公开dev的20个DB均有测试套件；public
  test使用的40个DB无对应套件，不以单库评分替代，也不宣称已执行public test。

- [x] 公共生成/评分入口完成原dev位置0/1/2的真实NPU验证：Qwen3.5-9B三次响应均stop，原响应与SQL预测逐字节一致；官方完整32份测试库评分1.000。接入提示及4096输出预算明确非官方模板/预算，不据三题作质量结论。

- [x] 固定CI工件及研究消费闭环；实际CI安装工件的全11840条原记录与原索引核对、12次公共/研究回查通过，原生三题生成/官方Test
  Suite评分结果直接复用。来源按固定抽样交付打勾，不是全量原生评分，见[阶段回执](validation/spider-original-preparation-20261029.json)。

- [x] Spider读取模块在服务器核对全部11840条、四个原始划分、七个原字段及分划分schema/数据库资产关系；数据库同名不跨命名空间合并。统一registry、候选及实际CI安装验证均已完成，新增模型调用0。

- [x] Spider统一registry和候选wheel已接通：服务器独立环境安装后，从site-packages统一入口核对全部11840条原记录，原问题、独立参考答案、schema及两个数据库命名空间均保持；候选验证不替代后续实际CI验证，二者均有独立回执。

- [x] Spider候选工件全11840条索引已核对，四个原划分各前三条共12次公共/研究消费回查完全一致；冻结研究代码身份与当前研究checkout一致，未改默认依赖。首次审计的单文件/全文件选择范围差异已查清并按同范围纠正，不重建有效索引。

- [x] 固定提交366f8323的两项CI成功，实际工件12876710598的67份载荷与精确Git
  blob一致、独立安装字节一致；全11840条原记录与原索引核对、12次公共/研究回查通过，零新增模型调用。Spider按固定原生抽样交付整项打勾，不是全dev质量或性能结论。

- [x] 后续入口导航修复已通过固定提交9b36fb93的两项CI，并在服务器独立安装实际工件12879026361；67份包载荷一致，准备、生成、隔离评分三个入口已核验。与先前工件相比仅分类资源变更，原生结果和研究回查证据直接复用，不重复调用模型。

### SWE-bench／Verified共用官方软件与环境（两来源分别固定抽样交付）

- [x] SWE_BENCH自身固定版本原测试Parquet已在服务器取得并按官方LFS大小/SHA核验，2,294条/12字段、12,097,227字节；原前三题11693/12057/12318均不在Verified中，不能复用其模型结果。复用公共获取/只读核验入口，Windows不落原数据。

- [x] SWE_BENCH原三题官方仓库镜像经公共入口冷构建完成；参考补丁环境对照3/3修复通过，保存原测试脚本/参考ID/评分重新核对一致，0重复测试/模型调用，原Docker历史配置保留。仅环境对照，不是模型任务完成。

- [x] 实际CI39636工件12910016155独立安装，67份载荷与Git/安装文件逐字节相同。SWE_BENCH全2294条/12字段和Verified全500条/13字段完整适配、索引及各固定3条研究回读通过；研究代码/默认依赖不变，无原数据重下载。

- [x] 最新实际CI工件12912171225再次核验两来源全字段/索引与各固定3条研究回读；67份软件载荷与b3e4
  Git对象相同。该提交两项CI均通过（40910826104、40910833864），仅同步137条既有基线行号，不新增豁免、不改摘要或审核标记。

- [x] SWE_BENCH固定原三题原生链路完成：首题11693的非法补丁结果保留不重跑；连接中断的原12057/12318恢复后均提交并完成官方测试，均未修复，原失败尝试全部保留，不挑题或重跑质量失败。原模板、任务、提交、测试脚本/参考ID与保存评分独立核验；两轮146+238=384次NPU后端请求均200，模型窗口结束、租约释放，15分钟原生队列仍活。当前代码9399两项CI均绿，67份已核验软件载荷逐字节相同。[来源独立交付回执](validation/swebench-classic-native-delivery-20261030.json)；非全2294题评测，组合验证不冒称单轮官方榜单分数。

- [x] 固定Verified原顺序前三题为Astropy12907/13033/13236，保留原13字段。发现已准备的新官方harness要求额外image/eval_script/log_parser/eval_type，不能直接套用旧数据。另固定官方v4.1.0发布版原87文件/594174字节及Git/SHA身份，`prepare_swebench_author_source.py --profile v4.1.0`计划与软件夹具通过，默认配置不变；没有补造字段或更换原题。

- [x] 服务器实际执行v4.1.0公共准备入口，87份原文件核验通过；只读复用已有客户端满足原14项直接依赖，原模块导入及固定前三题ARM规格构建通过。原FAIL_TO_PASS数量2/1/2、PASS_TO_PASS数量13/20/644保持不变；这是源码、依赖和规格核验，不是仓库测试或Agent执行。

- [x] 官方Ubuntu22.04的ARM基础镜像已从官方仓库按manifest/config/压缩层SHA及解压DiffID校验后导入独立实例。服务器直连DockerHub超时、默认断网下apt解析失败的两次原构建记录保留；仅对可信官方软件安装的第三次构建指定进程级host网络，原Dockerfile不改，已观察到原apt源正常下载索引。Windows不保存镜像大文件，不变更全局网络，生成代码评分不能沿用该联网配置。

- [x] 未修改的v4.1.0原基础Dockerfile在独立实例构建成功，镜像约1.29GB；原三题共享Python3.9环境和全部三题仓库镜像按原脚本构建完成。原依赖和安装脚本不改，镜像构建不等于任务执行和评分完成。

- [x] 公共准备器新增可选`--profile mini-swe-agent`，固定官方原78文件/302226字节，服务器实际公共入口逐文件Git/SHA/大小核验通过。保留原配置与Agent实现，不在数据包中夹带模型、原题或研究观测。软件准备、客户端及原题执行分别核验，不能据此新增来源完成数。

- [x] 官方Agent独立客户端已只读复用原核心依赖，36项差量包逐项SHA固定后安装，组合依赖检查和未修改的原CLI帮助通过。原Docker环境类使用首题原镜像的只读控制通过：原Python3.9激活正常，容器内cap0/NNP、8GiB内存、2CPU、512PID限额生效，无网络、宿主挂载、NPU或模型凭据；容器内root与可写文件系统如实保留，不冒称非root/只读根。无模型生成或原题执行。

- [x] 经明确授权建立只监听回环的临时SSH转发，保留原公钥、禁止命令执行和其他转发端口，带UTC到期时间；两项拒绝控制及CPU到原NPU模型发现接口通过。原模型服务未重启、不改监听、不开放公网。旧SSH到期格式兼容和一次传输超时均保留证据；这只是客户端连通，不是Agent原题运行。对应软件两项CI通过。

- [x] 原Agent公共调用入口及无模型计划验证已补；原题字段与物理顺序不改，部署绑定与原Agent设置分离，参考补丁不拼入任务。每次NPU请求先检查原服务身份、空闲与NPU所有者，定时队列协作让出窗口，保留原250步预算；完整原生结果和当前软件CI状态分列于下，不把计划验证等同任务执行。

- [x] 保留真实执行暴露的可选依赖缺失：原LiteLLM在首个模型请求前找不到`mcp`，未改原SDK/Agent，而是仅补官方`mcp==2.3.0`等11个新包并逐项SHA核验，所有已有依赖版本不变、依赖检查通过。原生执行随后真实进入NPU；不把此前CLI帮助或静态检查冒充运行时完整性。

- [x] 固定三题全部由原Agent提交，144次NPU请求均HTTP200；独占窗口正常释放，评分仅用专用CPU。官方参考补丁环境对照3/3修复通过；模型实际为1题修复、1题未修复、1题非法补丁被原工具拒绝。五份实际测试日志重新判分一致，三份轨迹的初始原模板/任务、提交和原预算逐项核对；第三题失败原样保留，不修输出、不换题、不重跑模型。原1423份容器/108份镜像配置哈希及1365个卷名仍未变。

- [x] 最新实际CI工件12900186474的67份软件载荷与精确Git对象、wheel及安装文件逐字节一致；原500条/13字段、500索引和固定3条当前研究消费再次通过。研究当前HEAD与冻结consumer源码身份一致，未修改研究代码或依赖。新增直接原Parquet入口全500条与原文件一致，前三题导出字节与真实运行JSON快照相同；公共镜像入口实际复用3镜像，Agent/评分的原Parquet计划通过，没有新模型调用、原数据重下载或重复测试。

- [x] 已导出实际CPU客户端的76包评分/120包Agent官方PyPI
  SHA锁定清单，120个最终版本与真实叠加环境逐项核对；不安装进NPU服务或原Python3.9测试镜像。未额外声称重跑过全新扁平环境安装。公共镜像入口显式传原生ARM规格，避免原CLI默认x86；原生任务/模板/测试脚本保持不变。

- [x] 当前批公共镜像、Parquet和评分入口的c5859eda两项CI均通过（40893299266、40893307702）；67份已核验数据载荷与该提交Git对象逐字节相同，无归一化。Verified固定原顺序三题原生链路和研究回读完成，已打来源勾；模型失败保留，非全量500题评测。SWE_BENCH自身数据/执行仍未完成，不共用完成勾选。

- [x] Verified固定三题的原作者源码部署、隔离客户端/ARM仓库镜像、官方Agent和原评分已完成；不以此替代SWE_BENCH自身的数据、任务与执行验证。

- [x] 固定上游02e7a74f的605份原软件、2950700字节；服务器公共准备入口及独立只读verify均通过，逐文件Git对象/SHA256/大小一致，不下载或复制任务数据、原仓库镜像或生成补丁。

- [x] 新隔离Python3.12/ARM客户端按原17项核心依赖解析76项官方PyPI确定哈希安装；`pip check`及未修改的原评测CLI帮助均通过，不安装CUDA推理extra，不调用模型，不执行任务或评分。

- [x] Verified已有原始记录取得/读取证据保留，两个来源复用同一准备入口，不重复建设。专用CPU工作区自动停止后按原配置恢复；原PVC身份及Spider评分回执仍在，没有变更TTL或共享模板。

- [x] 实际CI工件12880396493的67份载荷已核对精确Git对象并独立安装；复用原Verified文件核验全部500条、13个原字段、500条索引及固定前三条公共/研究消费一致。参考补丁与测试不进入模型输入，不改研究代码或默认依赖，无新增模型调用。这是数据接入闭环，不是原生仓库修复或评分完成。

- [x] 经明确授权，在同一专用CPU工作区启动独立Docker29.9.1，采用19GiB容量受限文件系统和独立套接字；真实非root/只读/断网/cap0容器及内核CPU/内存/PID限额探测通过。原daemon未重启、1423份容器和108份镜像配置哈希、1365个卷目录名未变；只清除了已保存证据的本次临时探测容器。不是原daemon根因修复，也不是原生benchmark运行完成。

- [x] 独立实例已精确恢复此前合格的Matplotlib与浏览器镜像：分别8/11层，完整层SHA256及镜像配置不变；流式导入，不另存镜像大包。两个环境校验器通过，公开隔离控制器的SVG正常/空输入、Matplotlib正常/超时四项控制通过；超时如实保留为阻塞，非模型得分。原环境配置哈希和卷名再次核验不变，没有新增模型调用或来源完成数。

- [ ] SWE_BENCH自身的数据取得、精确工件消费与原生任务/评分仍待补，复用已合格的公共软件与环境；Verified不能替代它的完成勾选。ARM支持属上游实验性，当前证据仅固定原三题，不是完整500题或任意任务容量保证。独立运行时受19GiB文件容量上限约束，另为辅助文件预留1GiB，总新增预算20GiB；37GiB是恢复前历史观察，1.80GiB是仅恢复两运行时后的历史占用，都不是当前任务占用。实际任务镜像和评分已在此限额内执行，当前容量与结果见[阶段回执](validation/swebench-original-readiness-20261029.json)。

### LJSpeech原始TTS语料与VITS原生合成（已交付）

- [x] 核定原1.1版本、13,100条原配对语料定位及官方TorchAudio校验依据，补充无模型依赖的公共获取/核验入口和离线夹具测试；只允许来源记录匹配的精确范围续传，不覆盖原文件，不自动换来源。
- [x] 公共获取器在服务器实际下载完整2,748,572,632字节原包，SHA256与官方TorchAudio固定源码一致；约203秒，零重试，Windows没有原始音频落盘。
- [x] 全13,100条原三字段元数据与13,100份WAV一一对应，原文本不改写；全部RIFF长度、单声道16-bit/22050Hz音频头核验通过。仅流式读取原包，没有再解压一份数据或生成模型输入。
- [x] 实际CI工件12945880401安装后，69个载荷与Git及安装文件逐字节核验；完整13,100条索引及固定原前三条公共/未修改研究消费入口回查通过，完整记录一致、参考音频分离。复用完整索引恢复中断的回查，不重下、不重建索引。[原数据交接证据](validation/ljspeech-actual-ci-handoff-20261030.json)
- [x] 公共原归档适配器在服务器复用同一原包核验全部13,100条，最终代码字节单独复验，保留原文本、发布方规范化文本和参考WAV关系；不解压、不合成。原顺序、配对完整性、索引回查及既有研究消费的人工夹具测试通过；夹具不算真实数据研究消费。[公共读取证据](validation/ljspeech-public-reader-20261030.json)
- [x] 发布方VITS原软件23文件和配置固定并通过公共准备入口复验；官方链接的145,599,717字节权重已在服务器取得，记录观察SHA256并通过全部ZIP成员CRC核验。文件完整性检查未执行模型，后续安全加载见下一项。模型数据未写入本地磁盘。详见[准备证据](validation/ljspeech-vits-preparation-20261101.json)。
- [x] 独立环境安装官方phonemizer 2.2.1及兼容依赖；eSpeak NG
  1.51与动态库仅解包到本任务目录，原对齐扩展编译、原模型模块导入及原清洗函数控制测试通过。权重另经`weights_only=True`在CPU安全加载，未执行模型；控制测试不是原题合成，初次构建和导入失败记录保留。
- [x] [原生入口](../../scripts/run_ljspeech_vits_original_npu.py)按原前三条规范化转写在Ascend 910B2执行VITS
  FP32合成；原生成参数和完整输出保留，3份波形经独立RIFF/float32/22050Hz、有限非静音及输入/参考关系核验，无神经重跑。两套常驻服务未变、租约已释放。[实跑证据](validation/ljspeech-native-fixed3-20261101.json)
- [x] 当前软件CI工件13067137328的78载荷与Git、wheel和安装字节一致；复用13,100条原索引，原前三条公共/当前研究完整记录及实际合成输入、参考音频对应关系通过，无研究代码改动、无神经重跑。CPU
  smoke 41265280781和完整CI 41265277911均通过，来源按明确的原生合成协议交付；全量13,100条合成与人工自然度/MOS评价不在当前声明中。

### DRAGON原生RAG（固定原生抽样交付完成）

- [x] 新增无模型加载的依赖预检，在服务器现有服务环境实际执行；torch_npu可发现，但缺sentence_transformers及三项LangChain组件，不把包存在等同NPU执行合格。
- [x] 原示例CUDA专用初始化与可复用核心链路已核查；CLI默认top_k5/字符chunk1000/overlap100，核心使用完整语料的Chroma
  MMR，不能换成任意词法检索或随机embedding。
- [x] 固定原前三题的原提示传输、实际NPU嵌入、完整检索/生成/官方评分及无推理保存核验通过：完整542篇/1157原分块，5次直接验证输出NPU张量的编码调用，3次reader请求。3次均为作者512预算下的length终止，低质量结果保留；未缩提示、减少语料或替换题目，不算正确作答或性能证明。
- [x] 新客户端56项确定版本/原PyPI哈希安装及核心模块真实导入核验；Torch/NPU等六项保护版本未变。保留全套失败记录，修复仅限新客户端；全局已有11项依赖问题仍在，但未新增问题。不是干净全SDK或NPU模型执行合格，来源总勾不变。
- [x] 发布方列举的`intfloat/multilingual-e5-large-instruct`已固定官方版本`274baa43b0e13e37fafa6428dbc7938e62e5c439`，11份原模型/配置/分词器文件共1,142,121,329字节直接传到服务器；官方LFS
  SHA256或Git对象及独立全文件SHA256复核通过。391个safetensors张量范围完整，原Transformer/Pooling/Normalize三模块不改。既有网络失败保留，未换代理节点、未绕过访问限制；Windows未保存权重。此获取阶段没有加载模型或运行NPU，后续原生验证独立记录于上项。
- [x] 发布方原始reader配置已核对：此Qwen版本没有独立generation_config，显式从原config派生；作者预算fallback512不变。无效物理设备掩码、HOME满盘日志错误、缺配置及未触发的ST6
  hook均有记录，最终只调整部署适配和私有目录；不改作者核心、数据或在用服务。旧零hook计数未追认为有效设备证明，最终直接检查输出NPU张量。
- [x] 精确提交675b7da2的两项CI均成功；实际CI工件12890070997的67份载荷与Git对象、独立安装文件一致。原生入口/保存核验器与实际服务器执行代码字节相同，模型身份清单只补公开哈希误报注释。安装后的四类导航、全部600条原字段/索引和固定3条当前研究回读通过，0新增推理。来源总项按原生固定抽样交付打勾，不是全600或质量/性能完成；见[交付证据](validation/dragon-original-preparation-20261028.json)。

### Video-MME-v2资产与原协议准备（尚未交付）

- [x] 四份固定发布方原文件共9046844字节已在服务器取得，Git/LFS身份核验通过；包含全3200题和原800份字幕，不含视频，Windows无原始文件落盘。
- [x] 全3200题的11字段、800个完整四题组与原顺序、字幕对应关系及ZIP CRC核验通过；未拆散四题组或用旧Video-MME数据替换。
- [x] 三份官方协议源码共49432字节核验；原非线性分组评分器通过参考答案/CPU控制，不是模型评分，不计来源完成。
- [ ] 六个固定完整原组共24题的原视频、64帧原提示、合适NPU配置、实际原生推理与评分；当前32图服务配置不能冒充64帧配置，不为跑通缩帧。
- [ ] 原字段/媒体/分组适配与固定安装工件消费闭环；[阶段回执](validation/videomme-v2-original-preparation-20261029.json)明确区分资产、软件控制和实际执行，来源仍不打总勾。

## 本轮完整交付清单（2026-10-24）

2026-10-25补充复核：原37项复核与研究人员七批23项成果去重为46个有历史获取/准备/验证记录的来源（含本轮仅分类项及受限子集，不是46个完整benchmark）。其中此前遗漏的8项已有48份原文件身份，本轮已复用固定版本、URL与校验和，并提供统一离线计划/原文件校验入口；没有重新下载9.51GB原数据，也未复制研究特征、观测或分析。详见[复用审计](validation/research-acquisition-reuse-audit-20261025.json)。这些来源改标“历史资产已复用、原生待补”，不再用“未开始”掩盖已有成果；准备映射、原生执行和固定工件消费未收口前，完成勾保持不勾。

40 类共登记 110 个公开来源、8 份企业资产，合计 118 项；五个指定来源只分类，本轮推进 113 项。下方历史 19
个勾保留，不自动提升为本轮完成，也不要求已验证项目重复跑。当前交付可用48/113项、65项待补（新增TULU3原始对话抽样续接交付，详见具名回执）；原有分类统计说明如下：2项服务轨迹资产、30项原任务/固定评测协议交付（JSONSchemaBench原11配置、BBH、MATH500、GSM8K、AI2-ARC、NI登记任务、BFCL登记类别、QuALITY原NYU多选集成、LongBench登记Qasper200、tau2登记telecom_full、LoCoMo完整上下文文本QA、RULER原13种任务集成、GPQA原三split
zero-shot集成、BEIR
SciFact/NFCorpus原检索与重排、STSB原句对相似度、XLSUM原45变体摘要、LibriSpeech原完整波形抽样ASR、FLEURS原102语言306条完整波形ASR、XSum原BART抽样摘要、Multi-News原PEGASUS多文档抽样摘要、BillSum发布方HF原法案抽样摘要、HumanEval原三题功能评分、DS1000七库24题评分、MBPP
full原三题CodeEval评分、FLORES原三语向抽样翻译与原指标评分、NQ-open原dev3610纯文本及原生聊天3题集成、DuReader原search demo
dev100完整文档生成与原评分、原版QASPER
dev前三题答案/证据及作者CLI评分、CrossCodeEval四语言原默认检索补全及作者评分）、Dolly、OASST1、LongWriter、UltraChat、WildChat、ToolACE及LongAlpaca七项语料及8份私有企业请求语料。并非47个全量带官方质量评分的benchmark，也不代表这些模型质量达标，边界见逐项回执。

完成要求以主方案第零节及机器清单 delivery_contract 为准：原题、完整原生流程与适用评分必须跑通；每个独立协议预计执行和评分不超过 30
分钟则全跑，否则事前固定至少三条，不足则全跑。不得改题、删工具、截断或缩减原生预算，失败不得换题。资源不足保留未勾并记阻塞；模型答错不等于链路失败。服务轨迹无原始请求回放入口时，只能交付数据资产，不能编
prompt。

仅使用获授权 NPU，优先让现有研究任务完成，不向观测服务混入请求。原始数据、企业载荷及完整索引不公开；CI 仅 CPU 正确性 smoke。PR 不合并。以下逐项状态与同目录
dataset-validation-status.json 的 delivery_inventory 对齐。

### JSONSchemaBench 子流程进度（不增加来源完成数）

- [x] 固定官方34资产、44作者文件、真实11配置loader/33处原formatter和隔离客户端依赖核验。
- [x] WashingtonPost原38题完整作者bench/evaluator及逐题保存结果审计；复用6题，32次新请求尝试，30支持/8不支持，27条符合原validator；未改题或换题。
- [x] 严格流式编译拒绝分类及真实断点复用，保留失败与原输出；独立重评0新增模型调用，不作连续性能结论。
- [x] 单独default原始0/1/2的原生bench/evaluator及保存结果审计；3条符合原validator，不是全2867题。
- [ ] 其他9个具名配置原生验证；已保存20条，后端配置混用阻塞，待明确服务配置并核查复用兼容性。
- [x] 官方33文件、19,100配置/split视图行原字符串/ID逐行核验及全量索引；固定CI提供者44文件与99处公共/研究回查通过，旧入口不变。
- [x] 新准备入口deb68808的CPU smoke及主CI均成功；数据消费核验与原生执行证据分开保留。
- [ ] 整来源原生验证收口；其余分区已保存20条，单后端配置混用阻塞，未算整来源完成。

### 先行复核：已有37项证据与能力（与完成交付分开）

本表勾选只表示**已核对历史证据、代码登记、研究消费记录及后续缺口**，不表示该来源的原生 benchmark 已完成。本轮静态复核没有重跑原数据或占用
NPU。详细历史样本、运行状态及边界见总体方案第8.5.2节同名来源行。

复核对象原为 benchmark 草稿 PR
链（#379→#432）；现按用户授权，#432已作为首批汇总合入main，其余五个草稿关闭并注明承接关系，历史和分支保留。下表仍是既有能力复核的历史截面，逐源勾选表示该阶段审计完成，不代替上方唯一当前交付清单。研究侧
#37 的来源消费回执按固定提交核对。

复核结果：29个公开来源都在来源清单、权威来源目录和主适配器/扩展适配器注册；8份企业资产在来源清单、适配器注册和独立私有资产目录。37项均可复用现有原始读取/准备能力。36项有可解析且带对应来源ID的脱敏JSON回执；QReCC没有独立JSON回执，但固定准备说明记录了归档/test成员SHA、原始16451条及服务器prepare/index/resolve三条，并有研究侧消费回执。固定研究PR
#37中有23项来源级消费回执（含此前漏计的StructEval、TauBench）。另用固定研究消费代码和数据包，对剩余14项完成本地CPU-only原始数据prepare→resolve：8份企业资产各3/3，6个公开来源共24/24（含AI2-ARC两个test子集）；脱敏回执已记录，但不冒充PR
#37原有回执。现37项均有来源级消费证据，但不代表原生任务、评分或性能完成。

历史固定提交上的benchmark PR #432及研究侧PR
#37曾通过完整CI和无模型消费工作流；这只证明对应软件构建、安装和跨仓消费链路，不替代原生任务执行、评分或NPU验证。最新提交的CI需单独查看。研究侧#37已于2026-10-24 14:45:39
UTC合并为`da48172b57f14d0f7762201e08bbf026a808476b`，其消费源码树与固定研究提交`6ca466e9`一致；benchmark
#432仍为Draft且不合并。

历史模型运行分层仍为：13项有任务评分/validator样本（含BBH
0/3）；11项只证明响应或原始关系完整性、没有独立语义gold；4项虽有运行但任务链仍有缺口；9项停在读取或本来就是非prompt资产。MMLU-Pro在本表仅核查历史成果和分类，本轮不新增实现或运行。旧19个勾照旧保留，不能替代本轮113项原生交付状态。

**初始复核阶段历史快照（非当前进度）**：37项历史审计均已完成；当时Azure数据资产1项可用、BBH
1项正在原生验证、32项已有缺口待补、HumanEval、QReCC与ChartQA共3项受资源/环境阻塞、76项未开始，另5项仅分类。QReCC官方网页集合需tens of
TB；服务器/data约1.42TB可用，未缩小语料。ChartQA当前两卡主机没有Qwen3-VL权重且SSH已拒绝重连，未发模型请求。只有Azure数据资产计1个交付勾；这不是模型benchmark完成数。

| 来源 ID                    | 证据与能力复核 | 历史实证阶段                                   | 研究侧消费证据                         | 复用与待补                                                                     |
| -------------------------- | -------------- | ---------------------------------------------- | -------------------------------------- | ------------------------------------------------------------------------------ |
| AGENTBENCH                 | [x] 已复核     | 原生任务已有抽样评分                           | 有固定研究侧逐源回执                   | 复用已验证子集；剩余范围见8.5.2                                                |
| AI2_ARC                    | [x] 已复核     | 原生任务已有抽样评分                           | 本地补充消费9/9（Challenge 6、Easy 3） | 复用已验证子集；剩余范围见8.5.2                                                |
| AZURE_LLM                  | [x] 已复核     | 轨迹/历史工具数据资产                          | 有固定研究侧逐源回执                   | 按数据资产复用；不造prompt                                                     |
| BBH                        | [x] 已复核     | Boolean原生评分0/3复用；余下26项78条已评分复核 | 有固定研究侧逐源回执                   | 全27变体81个唯一原题覆盖；最终交付工件待收口，不改stop                         |
| BFCL                       | [x] 已复核     | 原生任务已有抽样评分                           | 有固定研究侧逐源回执                   | 复用已验证子集；剩余范围见8.5.2                                                |
| BIGCODEBENCH               | [x] 已复核     | 运行过，原生闭环有缺口                         | 有固定研究侧逐源回执                   | 复用已保留的成功/失败；按8.5.2补原生闭环                                       |
| BURSTGPT                   | [x] 已复核     | 轨迹/历史工具数据资产                          | 有固定研究侧逐源回执                   | 按数据资产复用；不造prompt                                                     |
| CHARTQA                    | [x] 已复核     | 已读原资产，原生任务未跑                       | 有固定研究侧逐源回执                   | 主机已恢复；已有Qwen3.5视觉张量，但当前服务text-only，待验证视觉配置与原图评分 |
| DOLLY                      | [x] 已复核     | 响应/原始关系完整性                            | 本地补充消费3/3                        | 复用完整性结果；不补造语义分                                                   |
| DS1000                     | [x] 已复核     | 运行过，原生闭环有缺口                         | 有固定研究侧逐源回执                   | 复用已保留的成功/失败；按8.5.2补原生闭环                                       |
| GSM8K                      | [x] 已复核     | 原生任务已有抽样评分                           | 本地补充消费3/3                        | 复用已验证子集；剩余范围见8.5.2                                                |
| HUMANEVAL                  | [x] 已复核     | 运行过，原生闭环有缺口                         | 有固定研究侧逐源回执                   | 复用已保留的成功/失败；按8.5.2补原生闭环                                       |
| JSONSCHEMABENCH            | [x] 已复核     | 原生任务已有抽样评分                           | 有固定研究侧逐源回执                   | 复用已验证子集；剩余范围见8.5.2                                                |
| LOCOMO                     | [x] 已复核     | 原生任务已有抽样评分                           | 有固定研究侧逐源回执                   | 复用已验证子集；剩余范围见8.5.2                                                |
| LONGBENCH                  | [x] 已复核     | 原生任务已有抽样评分                           | 有固定研究侧逐源回执                   | 复用已验证子集；剩余范围见8.5.2                                                |
| LONGBENCH_V2               | [x] 已复核     | 运行过，原生闭环有缺口                         | 本地补充消费3/3                        | 复用已保留的成功/失败；按8.5.2补原生闭环                                       |
| LONGWRITER                 | [x] 已复核     | 响应/原始关系完整性                            | 本地补充消费3/3                        | 复用完整性结果；不补造语义分                                                   |
| MMLU_PRO                   | [x] 已复核     | 原生任务已有抽样评分                           | 本地补充消费3/3                        | 仅分类；本轮不新增推理或评分                                                   |
| MOONCAKE                   | [x] 已复核     | 轨迹/历史工具数据资产                          | 有固定研究侧逐源回执                   | 官方回放入口已核；无冲突需9.37 TiB，默认2000页有modulo映射，未运行             |
| NATURAL_INSTRUCTIONS       | [x] 已复核     | 原生任务已有抽样评分                           | 有固定研究侧逐源回执                   | 复用已验证子集；剩余范围见8.5.2                                                |
| OASST1                     | [x] 已复核     | 响应/原始关系完整性                            | 有固定研究侧逐源回执                   | 复用完整性结果；不补造语义分                                                   |
| QRECC                      | [x] 已复核     | 已读原资产，原生任务未跑                       | 有固定研究侧逐源回执                   | 官方端到端还需1000万网页/5400万passages检索集合与原评估器                      |
| QUALITY                    | [x] 已复核     | 原生任务已有抽样评分                           | 有固定研究侧逐源回执                   | 复用已验证子集；剩余范围见8.5.2                                                |
| SCIENCEAGENTBENCH          | [x] 已复核     | 已读原资产，原生任务未跑                       | 有固定研究侧逐源回执                   | 按8.5.2补原生环境与评分                                                        |
| STRUCTEVAL                 | [x] 已复核     | 原生任务已有抽样评分                           | 有固定隔离消费回执                     | 复用已验证子集；剩余范围见8.5.2                                                |
| SWE_VERIFIED               | [x] 已复核     | 已读原资产，原生任务未跑                       | 有固定研究侧逐源回执                   | 按8.5.2补原生环境与评分                                                        |
| TAUBENCH                   | [x] 已复核     | 原生任务已有抽样评分                           | 有固定原生引用消费回执                 | 复用已验证子集；剩余范围见8.5.2                                                |
| TOOLACE                    | [x] 已复核     | 轨迹/历史工具数据资产                          | 有固定研究侧逐源回执                   | 按数据资产复用；不造prompt                                                     |
| TRACELAB                   | [x] 已复核     | 轨迹/历史工具数据资产                          | 有固定研究侧逐源回执                   | 按数据资产复用；不造prompt                                                     |
| PAIO-CHAT-1000             | [x] 已复核     | 响应/原始关系完整性                            | 本地补充消费回执3/3；未在研究PR #37    | 复用私有请求和完整性记录；不宣称语义正确                                       |
| PAIO-CODE-EVAL-1000        | [x] 已复核     | 响应/原始关系完整性                            | 本地补充消费回执3/3；未在研究PR #37    | 复用私有请求和完整性记录；不宣称语义正确                                       |
| PAIO-JSON-500              | [x] 已复核     | 响应/原始关系完整性                            | 本地补充消费回执3/3；未在研究PR #37    | 复用私有请求和完整性记录；不宣称语义正确                                       |
| PAIO-LONG-PREFILL-5000     | [x] 已复核     | 响应/原始关系完整性                            | 本地补充消费回执3/3；未在研究PR #37    | 复用私有请求和完整性记录；不宣称语义正确                                       |
| PAIO-LONGTEXT-4000         | [x] 已复核     | 响应/原始关系完整性                            | 本地补充消费回执3/3；未在研究PR #37    | 复用私有请求和完整性记录；不宣称语义正确                                       |
| PAIO-PREFIX-SHARED-5000    | [x] 已复核     | 响应/原始关系完整性                            | 本地补充消费回执3/3；未在研究PR #37    | 复用私有请求和完整性记录；不宣称语义正确                                       |
| PAIO-REUSE-CONV-5000       | [x] 已复核     | 响应/原始关系完整性                            | 本地补充消费回执3/3；未在研究PR #37    | 复用私有请求和完整性记录；不宣称语义正确                                       |
| PAIO-SEMANTIC-SIMILAR-5000 | [x] 已复核     | 响应/原始关系完整性                            | 本地补充消费回执3/3；未在研究PR #37    | 复用私有请求和完整性记录；不宣称语义正确                                       |

| 来源 ID                    | 本轮勾选                                       | 复用证据或待补事项                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| -------------------------- | ---------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| WILDCHAT                   | [x] 有界原始历史续写可用                       | 原首分片59857条绑定/索引；前三会话全部4处NPU续写，3 stop/1声明2048预算length保留；前三条完整字段与固定工件公共/研究回查一致。[证据](validation/wildchat-original-continuation-20261026.json)；无质量评分/在线Agent承诺，其他13分片和全量生成未验证。                                                                                                                                                                                                                                     |
| LMSYS_CHAT                 | [ ] 历史资产已复用                             | 已复用研究原6文件的版本/URL/校验和，支持离线计划与原文件只读校验；服务器路径、准备映射、原生执行/评分和固定消费待补。不是重新取数，不提前勾选。                                                                                                                                                                                                                                                                                                                                          |
| OASST1                     | [x] 原始会话语料可用                           | 两split原资产、88838行字段核对及各三点历史续接/安装包prepare→resolve通过。[交付回执](validation/oasst-corpus-delivery-20261024.json)；无官方质量评分、非实时会话、非全量生成                                                                                                                                                                                                                                                                                                             |
| ULTRACHAT                  | [x] 原始历史续写可用                           | 复用原适配/准备入口；原test_gen28304条完整字段/索引核验，前三会话全部7处NPU续写正常stop，首尾6处固定工件公共/研究回查一致；[证据](validation/ultrachat-original-continuation-20261026.json)。发布方合成历史语料，无质量评分；不是在线Agent，其他7分片/全量模型推理未验证。                                                                                                                                                                                                               |
| DOLLY                      | [x] 原请求语料可用                             | 八原类别各三条共24条生成/完整性及固定版本CI、安装回核通过。[交付回执](validation/dolly-request-corpus-delivery-20261024.json)；非15011条全量，无官方语义评分                                                                                                                                                                                                                                                                                                                             |
| GSM8K                      | [x] 原生抽样交付                               | 复用204/1319唯一原题执行评分；固定CI组件核对完整原资产及输入/参考隔离、保存原生结果回查。[交付回执](validation/gsm8k-native-delivery-20261024.json)；支持原生全量，未全跑、跨配置不汇总分数                                                                                                                                                                                                                                                                                              |
| MATH500                    | [x] 原生抽样交付                               | H4原前三题完整4轮及评分完成；固定CI wheel全500条字段、3条往返及三项CI通过。[交付回执](validation/math500-native-delivery-20261024.json)；非全500题，不冒充独立裁判质量认证                                                                                                                                                                                                                                                                                                               |
| GPQA                       | [x] 可用                                       | 原三个CSV全部1192位置均有响应及评分审计；原skip69循环分数保留，显式兼容只补3次推理、复用1189结果，原题/提示/预算不改。固定CI、作者Python3.9及研究消费通过，不称GPT-4或其他协议复现。                                                                                                                                                                                                                                                                                                     |
| MMLU_PRO                   | — 仅分类                                       | 复用历史回执，核对完整使用链路                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| AI2_ARC                    | [x] 原生抽样交付                               | 两子集各原前三题及末三题共12题执行、原版似然评分与回查通过；固定CI组件核对六文件。[交付回执](validation/arc-native-delivery-20261024.json)；支持分别全量，未全跑、不合并分数                                                                                                                                                                                                                                                                                                             |
| LONGBENCH                  | [x] 登记Qasper交付                             | 原Qasper200题原生执行、官方评分及完整输入/参考均复核；研究端固定工件三条双路径接通。不宣称其他LongBench子集或原Qasper全库。[交付回执](validation/longbench-qasper-native-delivery-20261024.json)                                                                                                                                                                                                                                                                                         |
| LONGBENCH_V2               | [ ] 原上下文容量阻塞                           | 原Short三题复用；事前固定Medium原3/5/8行中两题超131072窗口，Long原0/2/7行全部超限。未截断、换题或新增推理，不计完成。[容量回执](validation/longbench-v2-length-group-capacity-20261025.json)                                                                                                                                                                                                                                                                                             |
| LONGMEMEVAL                | [x] 已声明原生协议交付                         | Oracle/S完整历史及M原BM25-session/top10/CoN流程各原前三题，共9条生成、原QA评分及独立审计。实际CI工件78载荷匹配，公共／未修改研究消费者9条完整回读；本地9B judge与官方推荐judge分开。M全历史超限及非适用检索分数保留，不冒称全量或质量复现。                                                                                                                                                                                                                                              |
| LONGMEMEVAL_V2             | [ ] 待开展                                     | 查已有入口，缺失再补原生协议                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| QASPER                     | [x] 原版v0.3抽样原生交付完成                   | 复用原5049字段/索引与图像准备；原dev0..2完成3次NPU答案/证据预测和作者原CLI评分，Answer F1约0.4996、Evidence F1=0保留。当前A9精确工件/研究dd三split9处旧索引回查通过。[证据](validation/qasper-native-delivery-20261026.json)。非LongBench200、全dev/test、LED或视觉推理，不筛图表参考，不称质量通过。                                                                                                                                                                                    |
| NARRATIVEQA                | [x] 原summary-QA已完成                         | 原test前三题生成/双参考评分/保存审计完成；全46765问答三表字段关系与索引、精确CI44文件及固定研究端18处回查通过。[回执](validation/narrativeqa-native-delivery-20261027.json)。不是full-story、全test推理或作者模型复现。                                                                                                                                                                                                                                                                  |
| XLAM_TOOL                  | [ ] 待开展                                     | 查已有入口，缺失再补原生协议                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| BIGCODEBENCH               | [x] 固定原题交付闭环                           | 原3条sanitize、参考3/3、候选pass/pass/fail及独立审计通过，0新增模型调用。公共原Parquet获取/摘要、实际CI64载荷、1140原9字段/索引及受维护研究入口3条回读已核验。原Python3.10完整依赖CPU-only镜像已实际评分；新增代码专用构建上下文32原源码+6运行文件，完整锁及公开配方服务器实际构建完成，233锁版本/71原评估要求/32原文件核验通过；复用已有有效原3条评分，不重跑模型或冒充新镜像评分结果。e2b0.17废弃警告保留。[回执](validation/bigcodebench-container-runtime-readiness-20261028.json)。 |
| SWE_BENCH                  | [x] 固定原三题原生链路完成                     | 全2294条/12字段、CI工件和研究回读通过；原Agent三题提交，原评分2未修复/1非法补丁，故障原题恢复且首题不重跑。384次NPU请求，保存审计通过，非全2294题评测；[独立回执](validation/swebench-classic-native-delivery-20261030.json)。                                                                                                                                                                                                                                                           |
| SWE_VERIFIED               | [x] 固定三题原生链路完成                       | 原Agent144次NPU请求、官方参考对照3/3通过；模型1修复/1未修复/1非法补丁失败原样保留。保存结果审计、全500条数据字段/索引和固定3条研究回读通过，c585两项CI通过。非全量模型评测，SWE_BENCH另行待完成；[回执](validation/swebench-original-readiness-20261029.json)。                                                                                                                                                                                                                          |
| VISIONARENA                | [ ] 待开展                                     | 查已有入口，缺失再补原生协议                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| MMMU                       | [x] 原validation抽样交付完成                   | 全92文件/11550行/16字段与索引核验，30学科固定90题NPU推理和原评分（含4多图），152处公共/固定研究消费往返通过。[回执](validation/mmmu-native-delivery-20261027.json)。59次length保留，非完整validation质量或LLaVA复现。                                                                                                                                                                                                                                                                    |
| CHARTQA                    | [x] 原测试抽样原生交付完成                     | 原测试2500条全字段/图像字节/索引核验；双类别固定6题NPU生成、原评分与公共离线复核，CI产物和固定研究6条往返通过。[回执](validation/chartqa-native-delivery-20261026.json)。非全量质量/性能；修复构建器全新重编译未验证。                                                                                                                                                                                                                                                                   |
| VIDEOMME                   | [x] 固定原生协议交付完成                       | 全24原文件及900视频/744字幕字节核验、2700题原11字段/索引、实际CI及受维护研究27条回读已闭环。官方8帧与字幕8帧各27题均原生NPU推理/原评分完成，各13/27、全54 stop，独立原输入/帧/请求/回答/参考/评分审计通过且0新调用。27独立问题、9视频，不是2700题/全语义域质量或性能保证；仅学术研究。[环境与操作](../../workload-preparation/README.md#video-mme-%E5%8E%9F%E7%94%9F%E8%A7%86%E9%A2%91%E4%BB%BB%E5%8A%A1)、[回执](validation/videomme-original-readiness-20261028.json)。                |
| VIDEOMME_V2                | [ ] 资产及原评分器准备；原生待补               | 四份原资产、3200题/800完整四题组/800字幕及原CPU参考评分控制已核验；尚缺视频、64帧NPU配置、原生推理评分和消费闭环。[阶段证据](validation/videomme-v2-original-preparation-20261029.json)，0模型调用，不计完成。                                                                                                                                                                                                                                                                           |
| MVBENCH                    | [x] 固定原生协议与公共消费交付完成             | 完整3895原文件、4000题表/索引已核验；20类各前3题、16帧原生推理/评分34/60；实际CI62载荷及60条研究回读通过。支持全量4000题入口，未声称已全评或性能提升。                                                                                                                                                                                                                                                                                                                                   |
| LONGVIDEOBENCH             | [ ] 待开展                                     | 查已有入口，缺失再补原生协议                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| LIBRISPEECH                | [x] 已完成                                     | 原两split前三+最长共8条完整音频NPU转写和原规范WER通过；固定公开入口另跑未见原位置，5559条保真、CI38文件工件及两split公共/研究消费已核验。不是全5559推理或流式ASR。                                                                                                                                                                                                                                                                                                                       |
| FLEURS                     | [x] 原生交付可用                               | 原102语言test/模型全校验，各原始前3条共306条完整波形NPU转写/CER评分和保存结果独立重评通过；最新成功CI44包3语言9处公共/研究消费一致。非全test或Conformer复现。                                                                                                                                                                                                                                                                                                                            |
| WAVCAPS                    | [ ] 原生路径准备中                             | 固定发布方6份源码/说明身份及3份Python AST已核验；原音频、专用checkpoint、NPU运行与适用评分/消费仍缺，不以文本问答替代。见下方WavCaps准备记录。                                                                                                                                                                                                                                                                                                                                           |
| SCIENCEAGENTBENCH          | [x] 已交付原固定前三题单轮协议                 | 原始生成、NPU CodeBERT、隔离科学执行与102条研究消费核验完成；[最终回执](validation/scienceagentbench-fixed3-native-delivery-20261101.json)，程序失败保留，其他模式未验证。                                                                                                                                                                                                                                                                                                               |
| DS1000                     | [x] 已交付                                     | 通用原题ID／范围／full入口已接通，七库24道原题生成及作者评分完成；1000原字段公共／研究消费一致。ARM移植差异及告警保留，非全1000题成绩。见[回执](validation/ds1000-native-delivery-20261025.json)。                                                                                                                                                                                                                                                                                       |
| MIND2WEB                   | [x] 原生交付可用                               | 原始三split各前3完整任务共9题89动作、846次NPU请求及官方评分；原提示/预算不变，显式非思考模型适配。保存响应复核、实际CI包及双端索引消费通过，非全量/实时浏览器结论。[交付回执](validation/mind2web-native-delivery-20261030.json)                                                                                                                                                                                                                                                         |
| GAIA                       | [ ] 待开展                                     | 查已有入口，缺失再补原生协议                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| STRUCTEVAL                 | [x] 已交付：19个完整非渲染协议                 | 原950题NPU生成/官方语法路径评分复用，111份保存文件哈希一致；公开原数据/源码获取及隔离客户端说明齐，当前CI工件/未修改研究消费者完成2035行原字段与132次回查。仅声明非渲染协议，其他渲染/VQA模式及原生失败单列，不称全44协议完成。[交付范围与证据](validation/structeval-nonrender-native-delivery-20261101.json)                                                                                                                                                                           |
| PAWS                       | [x] 已完成                                     | 全8000条人工test＋15条辅助分类、保存输入/响应/评分复核；全751450条4字段/索引、精确CI44文件和固定研究消费21处回查通过。[交付回执](validation/paws-native-delivery-20261027.json)。声明式模型适配非作者聊天运行器；noisy辅助标签不作人工test。                                                                                                                                                                                                                                             |
| XLSUM                      | [x] 已完成                                     | 原45变体6173个固定位置均核验原输入/编码/解码及作者ROUGE；10变体全6068、其余35各3。固定CI/研究消费与独立环境通过，不代表全114198条NPU推理；限非商业研究。                                                                                                                                                                                                                                                                                                                                 |
| FLORES200                  | [x] 原生抽样协议已交付                         | 原Meta归档204代码、2009公开平行位置保留；复用原NLLB-600M三语向9次NPU翻译/6个唯一平行位置，chrF++与原SPM spBLEU评分、固定CI安装及公共/研究消费通过；[交付证据](validation/flores200-native-delivery-20261026.json)。不重生成、不替换3个缺失模型语言标记、不称全部语向已执行，模型限非商业研究。                                                                                                                                                                                           |
| BURSTGPT                   | [x] 已交付                                     | 三份各原20条官方模拟已核；固定CI提供者及研究消费端三份各原3行消费相等。无新模型调用；第2/3份仅授权官方缩放，不作原时序性能结论。见[回执](validation/burstgpt-native-delivery-20261025.json)。                                                                                                                                                                                                                                                                                            |
| TRACELAB                   | [ ] 缺口已列待补                               | 复用历史证据；原生缺口见37项复核表及方案8.5.2                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| AZURE_LLM                  | [x] 数据资产可用                               | Azure 本轮脱敏回执；非模型 benchmark                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| MOONCAKE                   | [x] 固定官方存储回放可用                       | 两原轨迹35639条读取/索引及12实际CI研究回读；两个官方场景各首3条原生I/O通过并独立核验，测试页已精确清理。无prompt合成、无模型推理或性能结论。[证据](validation/mooncake-official-storage-delivery-20261027.json)                                                                                                                                                                                                                                                                          |
| QRECC                      | [ ] 阻塞                                       | 官方原始collection规模超过当前约1.42TB可用空间；未缩减或替换                                                                                                                                                                                                                                                                                                                                                                                                                             |
| INFINITEBENCH              | [ ] 历史资产已复用                             | 已复用研究原12文件的版本/URL/校验和，支持离线计划与原文件只读校验；服务器路径、准备映射、原生执行/评分和固定消费待补。不是重新取数，不提前勾选。                                                                                                                                                                                                                                                                                                                                         |
| RULER                      | [x] 可用                                       | 原13种任务15组配置2530条真实请求及原评分审计完成，完整输入/预算保留；固定CI工件完成7500条全字段重建、公共回查90条及研究两路径各45条。QA2显式兼容修复已披露；不声称完整长度网格或复现发布方模型。                                                                                                                                                                                                                                                                                         |
| QUALITY                    | [x] 原NYU多选集成可用                          | 原6题/24候选似然及固定harness评分、完整文章token与安装包2086题索引/6条回查通过。[交付回执](validation/quality-native-delivery-20261024.json)；非SCROLLS数据、非发布方训练基线，非全2086题执行                                                                                                                                                                                                                                                                                            |
| BBH                        | [x] 原生抽样交付                               | 全27变体原任务准备、执行及固定评分可用；81个唯一原题已验证，固定CI工件安装回查及三项CI通过。[交付回执](validation/bbh-native-delivery-20261024.json)。不是6511题全量或高分保证                                                                                                                                                                                                                                                                                                           |
| BFCL                       | [x] 登记原生类别可用                           | simple_python原400题生成、官方AST判定及原文/响应/安装包400条prepare→resolve审计通过。[交付回执](validation/bfcl-native-delivery-20261024.json)；静态函数/参数匹配，不代表其他BFCL类别或真实工具执行                                                                                                                                                                                                                                                                                      |
| TAUBENCH                   | [x] 登记telecom交付                            | 复用ENV原三题，新跑ENV＋ACTION原三题并官方重评分一致，双角色/200步/10错误/ALL保留；干净客户端125依赖、89原源码及保存结果重评分核验通过，全部2285原生引用研究侧接通。非其他domain或全2285模型运行完成。[交付回执](validation/tau2-native-delivery-20261024.json)                                                                                                                                                                                                                          |
| TOOLBENCH                  | [ ] 待开展                                     | 查已有入口，缺失再补原生协议                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| AGENTBENCH                 | [x] FC ALFWorld原生协议已交付，其他域未完成    | 原new_std全部109任务、1625次模型交互及原评分46/109已审计；当前保存结果重评分/资产核验通过，0新增模型调用，全部109项研究原生引用接通。非全域AgentBench、视觉环境或官方Docker等价。[协议交付证据](validation/agentbench-alfworld-protocol-delivery-20261031.json)。                                                                                                                                                                                                                        |
| SCHEMASTORE                | [x] 原schema资产已完成                         | 原4810文件全核验，官方CLI测试976 schemas exit0；全1006 schema/3688正负例引用及索引、精确CI45文件和固定研究端16 schema/70例子字节回查通过。[回执](validation/schemastore-native-delivery-20261027.json)。非LLM请求/质量benchmark。                                                                                                                                                                                                                                                        |
| JSONSCHEMABENCH            | [x] 原生固定协议交付                           | 11原配置全部核验：WashingtonPost全38，其余9具名及default各原前3，共68配置视图任务；原bench/evaluator及逐项重评通过。33原文件19100视图行/索引、99处固定提供者/研究回查通过。旧生成复用，不称全量推理、统一后端性能或模型质量达标。[交付回执](validation/jsonschemabench-official-workflow-readiness-20261024.json)                                                                                                                                                                        |
| LIVECODEBENCH              | [x] 原生四场景固定抽样闭环                     | 三份原文件已取得校验；公开入口已完成3题×10候选代码生成/原评分，以及显式兼容自修复23请求/复用7输出/原评分，保存审计通过。另2场景各3请求×10候选/官方隔离评分及保存审计已通过；全479/442条公共字段及索引、各3条维护中研究读取已核验；代码生成原3条索引及实际CI安装包的三类任务/维护中研究回读均通过；自修复显式兼容，不宣称原模板或全量复现。                                                                                                                                               |
| HUMANEVAL                  | [x] 已交付                                     | 原前三题NPU生成结果复用，公共隔离入口官方评分2/3，失败保留；164题原字段公共/研究消费相等，原3题索引消费相等。不是全164题模型成绩。见[回执](validation/humaneval-native-delivery-20261025.json)。                                                                                                                                                                                                                                                                                         |
| MBPP                       | [x] full原生抽样可用                           | 原题11/12/13双NPU生成、原始三测试与setup隔离CodeEval：参考解均通过，预测2通过/1失败原样保留；974原记录完整映射/索引，14处公共/研究回查一致，精确CI安装工件已核。[证据](validation/mbpp-original-native-sample-20261026.json)；非全500题，sanitized仅取得资产。                                                                                                                                                                                                                           |
| REPOBENCH                  | [x] 固定原生抽样交付完成                       | 官方默认30层各3题共90原题NPU生成/原评分及保存审计完成；49684原行与60研究回读复用，47实际CI安装载荷重新核对。固定评分哈希种子0，EM均0保留，不是质量验收。见[交付回执](validation/repobench-native-delivery-20261027.json)。                                                                                                                                                                                                                                                               |
| CROSSCODEEVAL              | [x] 原生协议已交付                             | 69,494配置行全字段/索引核验复用；四语言原0..2生成/评分保留，公共入口原3..5共12条NPU生成及作者评分通过。精确成功CI44文件及12处公共/研究回查完成；非全语料或所有检索配置推理。[证据](validation/crosscodeeval-original-acquisition-20261026.json)。                                                                                                                                                                                                                                        |
| LOCOMO                     | [x] 原生抽样交付                               | 原9题复用、新99题执行及原提示/官方评分复核完成，合计108条；第3类全96题，其他四类各原前三题。原数据不改，第5类显式缺键兼容非上游已合并补丁；研究端接通。不称全1986题、图像或RAG完成。[交付回执](validation/locomo-native-delivery-20261025.json)                                                                                                                                                                                                                                          |
| NATUREBENCH                | [ ] 原任务包准备通过，原生未完成               | 官方CPU3全部576原文件2.75GB、实际CI工件独立安装、索引及当前研究侧3条回读通过；合格隔离/ARM-NPU科学Agent/评估器/有效性judge待完成，不能替换成静态问答。其余87任务未取得。                                                                                                                                                                                                                                                                                                                 |
| DISCOVERYBENCH             | [ ] 完整原始资产已取得，原生任务待验证         | 固定官方版本1499文件、54085184字节已在服务器逐份Git blob/大小及文件集合核验，公共只读入口复用通过，无重下。包括原real/synth数据及Agent/评判协议；1499是文件数不是题数。代码执行环境、原生Agent/评分及消费闭环未完成，不打总勾。                                                                                                                                                                                                                                                          |
| MLAGENTBENCH               | [ ] 待开展                                     | 查已有入口，缺失再补原生协议                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| MLEBENCH                   | [ ] 待开展                                     | 查已有入口，缺失再补原生协议                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| WEBARENA                   | [ ] 待开展                                     | 查已有入口，缺失再补原生协议                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| VISUALWEBARENA             | [ ] 待开展                                     | 查已有入口，缺失再补原生协议                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| OSWORLD                    | [ ] 待开展                                     | 查已有入口，缺失再补原生协议                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| XSUM                       | [x] 已完成                                     | 原BART固定test抽样、官方HF ROUGE与公开入口未见原题/空正文边界验证通过；三split226711条保真、CI38文件工件和研究消费已核验。不是全11334条test推理或2018论文复现；34条原空正文保留。                                                                                                                                                                                                                                                                                                        |
| MULTINEWS                  | [x] 已完成                                     | 原PEGASUS固定test抽样/原HF评分、公共入口未见原题及空正文边界通过；56216条与作者loader逐条一致，原配对/换行/分隔符保留。CI39文件工件和研究消费已核，不是全5622题test或全文长上下文验证。                                                                                                                                                                                                                                                                                                  |
| BILLSUM                    | [x] 原生协议已交付                             | 发布方HF原23455条三字段、两测试集冻结前三条及公共入口3/4/5的NPU推理/HF评分、固定CI39文件工件与研究消费均通过。不代表全量推理、原论文clean_text或完整原始归档。[交付回执](validation/billsum-native-delivery-20261025.json)                                                                                                                                                                                                                                                               |
| TULU3                      | [x] 原始历史续接交付                           | 原6文件、全939343条字段与索引、18处固定研究回查；6分片前3条共18对话21次NPU续接，精确CPU CI工件复核通过。见原生交付回执；非官方评分/全量推理。                                                                                                                                                                                                                                                                                                                                            |
| NATURAL_INSTRUCTIONS       | [x] 登记原生任务可用                           | 五文件/8284实例字段及固定CI回查通过；分类102、翻译250、抽取261、改写1178全覆盖，摘要首尾6题，合计1797唯一实例。[交付回执](validation/ni-native-delivery-20261024.json)；非整个NI，不替代底层数据集原benchmark协议                                                                                                                                                                                                                                                                        |
| OPENMATH2                  | [x] 固定原始语料profile已交付                  | 3个固定原分片全部1309949条字段/索引、9条原问题生成与保存核验、实际CI包及研究回读通过；其余29分片未验证，不宣称数学质量评测。                                                                                                                                                                                                                                                                                                                                                             |
| OPENTHOUGHTS3              | [x] 原语料续写交付                             | 固定三分片30000行/索引保真，三域各3条NPU续写及保存审计、实际CI46载荷/18研究回读通过；九条均预算耗尽无最终答案，不作质量或全120万条执行结论。[回执](validation/openthoughts3-native-delivery-20261027.json)                                                                                                                                                                                                                                                                               |
| OPENCODE                   | [x] 固定原始代码语料profile已交付              | 3个固定原分片全部300000条九字段/索引、9条原问题生成及保存核验、实际CI包57载荷和研究侧回读通过；未执行代码或评判正确率，其余47分片未验证。                                                                                                                                                                                                                                                                                                                                                |
| SWESMITH                   | [ ] 进行中                                     | 11份原分片/59,136条读取通过，固定前三条公共与研究端回查一致；CI wheel和研究回查通过；原生Agent/评分待验                                                                                                                                                                                                                                                                                                                                                                                  |
| WORKARENA                  | [ ] 待开展                                     | 查已有入口，缺失再补原生协议                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| APPWORLD                   | [x] 固定原三题原生流程与消费交接完成           | 公共入口141步、官方评分三题各1/5且均失败；28缓存响应复用＋113新增HTTP200。14696原文件、实际CI71运行载荷及732原索引上三题公共/研究完整回读已核验。不代表全量、高分或性能；见[交付回执](validation/appworld-native-delivery-20261030.json)。                                                                                                                                                                                                                                               |
| SCIENCEWORLD               | [x] 固定原生抽样交付完成                       | 原生9个episode/464次NPU调用及保存作者评分核验完成；7207原始记录全字段/索引、9实际CI研究回读、51安装载荷与CI闭环通过。入口支持全部30类及原生完整test。严格新环境文本重放差异保留，非确定性复现或质量/性能验收。[交付证据](validation/scienceworld-original-native-delivery-20261027.json)。                                                                                                                                                                                               |
| BIXBENCH                   | [ ] 原元数据已取得，原生待接入                 | 原205条字段/索引及实际CI提供方/研究原三条回查、ZIP校验通过；科学Agent/NPU和原评分待完成，仍缺57份全语料ZIP。见bixbench-actual-ci-handoff-20261030.json。                                                                                                                                                                                                                                                                                                                                 |
| LABBENCH                   | [x] 已完成                                     | 31原子集各首3条共93题原生NPU运行、原评分及保存复核通过；87 stop/6 length保留，45题被原解析器解析、14题正确。1967条索引/原图像、CI46载荷和124研究回读已核复用；不是全量推理或作者模型质量复现。                                                                                                                                                                                                                                                                                           |
| PAPERBENCH                 | [ ] 待开展                                     | 查已有入口，缺失再补原生协议                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| SPOKENWOZ                  | [x] 原文本DST协议已交付，音频未完成            | 固定3+3完整对话，共128次NPU原生DST请求，官方/独立评分4/68、6/60，低分保留。全1000对话入口及范围选择可用；实际CI组件与研究原输入接通。音频/其他响应协议未完成，不据此计语音覆盖。[协议交付证据](validation/spokenwoz-text-dst-native-delivery-20261031.json)。                                                                                                                                                                                                                            |
| DAILYTALK                  | [x] 原Guo对话TTS协议已交付                     | 原三段完整对话29条NPU合成及独立PCM审计；23676行原字节保真，CI79文件工件/研究29条回查通过，完整对话准备可参数化；不称全量媒体、MOS或流式时序。[交付回执](validation/dailytalk-fixed3-native-delivery-20261101.json)                                                                                                                                                                                                                                                                       |
| LONGWRITER                 | [x] 原语料生成交付                             | 6000条原字段/索引审计，复用前三条、新范围入口实跑末三条均完成响应完整性复核；研究侧双路径三条接通。无语义或长度达标结论，不是LongBench-Write。[交付回执](validation/longwriter-native-delivery-20261024.json)                                                                                                                                                                                                                                                                            |
| LONGALPACA                 | [x] 原始指令语料使用已交付                     | 原12000条全部字段/索引和12处固定工件公共/研究回查，补回3231条file来源标识于非输入侧；固定原0..2与7324..7326共6题NPU执行并核验，6 stop，含2条原非空input。[证据](validation/longalpaca-native-delivery-20261026.json)。构造训练语料而非独立质量benchmark，不称全量生成、训练复现或每条均为文档分析。                                                                                                                                                                                      |
| TOOLACE                    | [x] 有界原始历史续写可用                       | 全11300条原system/对话字段及索引、首尾6条固定工件公共/研究回查；原三会话全部15处NPU续写正常stop。[证据](validation/toolace-original-continuation-20261026.json)；原历史tool返回保留、不造工具/ID；无在线Agent/BFCL/ToolACE-8B复现或全量模型推理承诺。                                                                                                                                                                                                                                    |
| DUREADER                   | [x] 原demo原生交付完成                         | 原dev100题/453文档完整NPU生成、五种原评分分支与保存输入/响应复核；94条预测格式有效、6条格式错误保留。200题/916文档保真及12处固定工件公共/研究消费复用。[交付证据](validation/dureader-native-delivery-20261026.json)。是Qwen生成式替代配置，不是作者旧基线或完整DuReader/Zhidao；无效Depends/ROUGE-L不填0。                                                                                                                                                                              |
| SPIDER                     | [x] 固定原生抽样及CI工件/研究消费交付完成      | 原dev0/1/2由Qwen3.5-9B生成，完整32库原Test Suite评分1.000；固定CI工件12876710598核对全11840条、原索引及12次公共/研究回查，非全量质量或性能保证。[交付证据](validation/spider-original-preparation-20261029.json)。                                                                                                                                                                                                                                                                       |
| NATURALQUESTIONS           | [x] NQ-open原任务已交付                        | 原dev3610题纯文本全量NPU生成/评分，原生聊天模式另3题；91535条索引、12处固定工件公共/研究回查及干净客户端重评分通过；[交付证据](validation/nq-open-native-delivery-20261026.json)。低分与空输出保留，两profile不混分，不是完整NQ文档、RAG或质量通过。                                                                                                                                                                                                                                     |
| MULTIAGENTBENCH            | [ ] 原五份数据与源码消费链路通过，原生执行待补 | 官方五文件500条原数据取得并校验；统一适配/引用索引及未修改研究消费者500条完整回读、原配置还原一致，数据库评分参考分离。实际CI72载荷独立安装及500条公共/研究回读也通过；原生环境/模型/裁判及官方评分仍待完成，不把读取当执行。详见机器清单。                                                                                                                                                                                                                                              |
| DEEPPLANNING               | [x] 五类原生样本及实际CI/研究消费闭环          | 购物三级、英文/中文旅行共5个profile各原前三题，合计15题原生执行/评分及保存审计通过；360条原字段/索引及15条当前研究回读经实际CI包复核。436次SDK尝试与414条保存消息分记；失败/零分保留，不宣称360题模型全跑或性能。[交付证据](validation/deepplanning-original-native-delivery-20261028.json)                                                                                                                                                                                              |
| STREAMINGBENCH             | [x] 已交付real原视频NPU协议                    | 固定前三完整组原生生成3次、原评分2/3且独立重算一致；原题、参考、时间窗和128预算未改。当前CI工件78载荷独立安装核验，8文件／4948索引及24条公共研究完整回读通过，实际原生输入匹配；原9B恢复，卡6未动。全量、SQA／主动／音频／在线模式未验证，45音频待补，详见交付回执。                                                                                                                                                                                                                     |
| OVOBENCH                   | [x] offline原生协议已交付                      | 12类36题/103响应NPU执行、原评分、完整入口及实际CI消费交接已核验；[交付证据](validation/ovobench-offline-native-delivery-20261031.json)                                                                                                                                                                                                                                                                                                                                                   |
| FULL_DUPLEX_BENCH          | [ ] 官方评分器准备已核验，原生语音待完成       | 原16文件已核验，两个未修改评分器对发布方样例运行exit0；公共11文件准备入口在专用CPU验证通过。尚无本项目原生语音推理或真实输出评分，不能打来源完成勾；详见下方原生准备清单。                                                                                                                                                                                                                                                                                                               |
| LIBRITTS                   | [ ] 原包及完整配对已验证，原生TTS待完成        | 官方test-other完整964502297字节双校验通过；实际CI组件78文件一致，5120个原始话段及原文/规范化文本/参考WAV全量配对还原，公共前三条准备通过。固定三条研究只读回查通过、索引不变；未做合成/质量评价或完整PCM解码，不计来源完成。                                                                                                                                                                                                                                                             |
| LJSPEECH                   | [x] 原VITS文本到波形协议可用                   | 原13100配对索引和公共/研究读取保留；固定原前三条NPU完整合成、独立RIFF与输入/参考关系核验，以及当前CI工件接入均通过。全量合成、MOS和性能不作声明。见ljspeech-native-fixed3-20261101.json。                                                                                                                                                                                                                                                                                                |
| BEIR_SCIFACT               | [x] 已完成                                     | 原test全量检索/top100重排、原评分、最终公开NPU命令及固定CI组件/研究消费均通过。范围仅为登记集合与协议，不是BM25或全部BEIR基线。                                                                                                                                                                                                                                                                                                                                                          |
| BEIR_NFCORPUS              | [x] 已完成                                     | 原test全量检索/top100重排、原评分、最终公开NPU命令及固定CI组件/研究消费均通过。范围仅为登记集合与协议，不是BM25或全部BEIR基线。                                                                                                                                                                                                                                                                                                                                                          |
| MSMARCO_PASSAGE            | [ ] 原候选集合获取阻塞                         | Microsoft原README/许可/评分器/脚本4文件已核验；官方静态下载入口连接重置、旧Blob入口409，公共镜像探测404。已找到IR-datasets规范校验元数据，仍缺原dev候选集合与qrels；不得造负例、改用QA或缩减候选集合。仅非商业研究。                                                                                                                                                                                                                                                                     |
| STSB                       | [x] 已完成                                     | 原dev1500/test1379全量NPU编码与原评分通过；8628行保真、固定CI组件、三split公共回查/研究消费及安装后缓存重算通过。不含训练复现或生产检索流量。                                                                                                                                                                                                                                                                                                                                            |
| DOCVQA                     | [x] 固定原生视觉问答可用                       | 六原文件5349题全字段/原图及引用索引、36实际CI研究回读；固定首3题NPU原生模板/图像处理/ANLS及独立保存复核完成，分数0.75/1/1。非全量模型评测或作者模型质量复现。[证据](validation/docvqa-original-native-delivery-20261027.json)                                                                                                                                                                                                                                                            |
| CODEJUDGE_EVAL             | [x] 原生固定抽样可用                           | 六份发布原JSON共11160配置行核验/索引、18条原题NPU执行和独立评分/保存核验通过；实际CI工件独立安装及18条研究回读通过。[交付证据](validation/codejudge-original-native-delivery-20261027.json)。不是作者未发布regex或论文分数复现。                                                                                                                                                                                                                                                         |
| DRAGON_PUBLIC              | [x] 固定原生抽样交付完成                       | 完整542篇/1157分块的NPU嵌入、原MMR/提示/官方评分和保存审计通过；实际CI67载荷及原600行/索引/固定3条研究回读通过。3次原512预算length保留，不是正确作答、全600或质量/性能结论。[回执](validation/dragon-original-preparation-20261028.json)。                                                                                                                                                                                                                                               |
| TEMPORALWIKI               | [x] 已交付原GPT2基座探针PPL                    | 全部8CSV／35948行／1126批次NPU计算及原float32标量评分独立重算通过；当前CI工件78载荷安装核验、全部原字段／索引和24条公共研究回读一致。原25-token／batch32／float32不变；未复现持续训练checkpoint、T5或其他语料模式，低质量分数与首次资源上限失败保留。                                                                                                                                                                                                                                    |
| COPILOT_AGENT_2026         | [x] 原生元数据资产可用                         | 七日606分片/301026原会话全量核验和索引，49实际CI载荷与候选一致，42固定实际CI研究回读及3条公共smoke完成；不造prompt、不宣称请求回放。[证据](validation/copilot-2026-metadata-delivery-20261027.json)                                                                                                                                                                                                                                                                                      |
| AZURE_LMM_2025             | [x] 服务轨迹元数据资产交付                     | 全100万行四字段保真/索引、47实际CI安装载荷、首3/末3研究回读完成；无原prompt/媒体，不造请求，不称视觉benchmark。见[回执](validation/azure-lmm-2025-asset-delivery-20261027.json)。                                                                                                                                                                                                                                                                                                        |
| HLE_VERIFIED               | — 仅分类                                       | 本轮不补实现、不重跑                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| SWE_BENCH_PRO              | — 仅分类                                       | 本轮不补实现、不重跑                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| FRONTIERSCIENCE            | — 仅分类                                       | 本轮不补实现、不重跑                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| TERMINAL_BENCH_2_1         | — 仅分类                                       | 本轮不补实现、不重跑                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| PAIO-CHAT-1000             | [x] 私有原请求可用                             | 复用原前三条NPU回放；固定CI读取/原响应束/原声明格式回核通过。[交付复核](validation/enterprise-fixed-ci-handoff-20261024.json)；无语义gold，正文不公开                                                                                                                                                                                                                                                                                                                                    |
| PAIO-CODE-EVAL-1000        | [x] 私有原请求可用                             | 复用原前三条NPU回放；固定CI读取/原响应束/原声明格式回核通过。[交付复核](validation/enterprise-fixed-ci-handoff-20261024.json)；无语义gold，正文不公开                                                                                                                                                                                                                                                                                                                                    |
| PAIO-JSON-500              | [x] 私有原请求可用                             | 复用原前三条NPU回放；固定CI读取/原响应束/原声明格式回核通过。[交付复核](validation/enterprise-fixed-ci-handoff-20261024.json)；无语义gold，正文不公开                                                                                                                                                                                                                                                                                                                                    |
| PAIO-LONG-PREFILL-5000     | [x] 私有原请求可用                             | 复用原前三条NPU回放；固定CI读取/原响应束/原声明格式回核通过。[交付复核](validation/enterprise-fixed-ci-handoff-20261024.json)；无语义gold，正文不公开                                                                                                                                                                                                                                                                                                                                    |
| PAIO-LONGTEXT-4000         | [x] 私有原请求可用                             | 复用原前三条NPU回放；固定CI读取/原响应束/原声明格式回核通过。[交付复核](validation/enterprise-fixed-ci-handoff-20261024.json)；无语义gold，正文不公开                                                                                                                                                                                                                                                                                                                                    |
| PAIO-PREFIX-SHARED-5000    | [x] 私有原请求可用                             | 复用原前三条NPU回放；固定CI读取/原响应束/原声明格式回核通过。[交付复核](validation/enterprise-fixed-ci-handoff-20261024.json)；无语义gold，正文不公开                                                                                                                                                                                                                                                                                                                                    |
| PAIO-REUSE-CONV-5000       | [x] 私有原请求可用                             | 复用原前三条NPU回放；固定CI读取/原响应束/原声明格式回核通过。[交付复核](validation/enterprise-fixed-ci-handoff-20261024.json)；无语义gold，正文不公开                                                                                                                                                                                                                                                                                                                                    |
| PAIO-SEMANTIC-SIMILAR-5000 | [x] 私有原请求可用                             | 复用原前三条NPU回放；固定CI读取/原响应束/原声明格式回核通过。[交付复核](validation/enterprise-fixed-ci-handoff-20261024.json)；无语义gold，正文不公开                                                                                                                                                                                                                                                                                                                                    |

### 真实原生执行进展

- [x] LONGBENCH/Qasper：原200题全量原生执行及官方F1评分已复核。复用前三题，仅新跑3..199共197题；全200题输入/参考/请求token及私有响应SHA、逐题和整体官方评分复算一致，预算不变。[原生回执](validation/longbench-qasper-all200-native-20261024.json)。另已完成研究侧固定工件原前三条的双路径交接，不代表其它LongBench子集或200条全部研究侧回查。
- [x] AgentBench/ALFWorld：复用已审计的前41题，新执行冷却29/照明18/双对象21题；全部109个原任务及1625次模型交互已复核，官方评分函数复算为46/109。未改题、动作、步数及输出预算，奖励0或合法轮次上限原样保留。[全量原生审计回执](validation/agentbench-alfworld-all109-native-20261024.json)。另已通过研究侧prepare和verify重验全部109项原生引用及原游戏资产；不代表研究观测验收。其他AgentBench环境仍待完成，不勾整源交付。

### 不占卡准备进展

- [x] LONGBENCH_V2：新增原题容量绑定准备CLI，直接执行官方提示构造、在推理前停止。服务器独立67包客户端环境`pip check`通过；历史1/4/6三题提示SHA及token数完全一致，反例0/1/2原样保留资源超限并整批拒绝，不截断/换题/评分，模型调用0。[准备回执](validation/longbench-v2-original-capacity-cpu-20261024.json)、[复现命令](LONGBENCH_V2_ORIGINAL_ATTEMPT.zh-CN.md)。
- [x] LONGBENCH_V2：移除原生公共入口固定三题的数量硬编码，显式绑定1..503题的软件校验通过，保留原题顺序、身份与原预算；负token数/超限整批拒绝，不过滤成容易子集。服务器相关回归已通过，模型调用0；原三题NPU历史证据不变，修改后入口未作NPU复验。
- [x] LONGBENCH_V2：补齐本地原生入口的 `/v1` 基地址校验；服务器 CPU
  参数/历史回执回归15项通过，模型调用0。历史三题执行结果保持不变，本项不代表全503题或修改后入口已完成NPU复验。
- [x] SCIENCEAGENTBENCH：原verified版102题的全部输入、参考及环境字段、全量索引和102条回查核验通过；30份官方执行/评分及依赖源码已固定并核对Git
  blob，未执行代码。[准备回执](validation/scienceagentbench-cpu-preparation-20261024.json)。
- [x] SCIENCEAGENTBENCH：固定原前三题（1/2/3）的完整科学数据目录、各自参考程序/评价脚本/rubric已在专用CPU服务器取得，共14文件1006008字节；同一原归档ETag的Range请求、原CRC和保存后SHA复核通过，公开资产校验入口实测退出0。未整包下载1.77GB归档，未执行参考程序或生成代码，原始文件不公开；这不是全102题资产取得或来源完成。
- [x] SCIENCEAGENTBENCH：原30份源码在专用CPU工作区复核，24份Python静态解析；原`make_test_spec`在Python3.10构造前三题通过，12个字段保留，ID仅沿用官方转字符串规则，配方选择ARM64。未构建镜像、未运行程序或模型，不代表科学环境通过；细节见唯一机器清单。
- [x] SCIENCEAGENTBENCH：显式本地模型通信接口及固定旧版SDK的两次合成回环HTTP控制通过，保留模型标识、原消息及所选GPT-chat分支2000输出预算；不是原题、不是NPU推理。原Agent价格/tokenizer、完整流程仍未接通，不填虚假价格，也不将Qwen改名GPT。官方OpenHands
  legacy入口亦经检查，不能盲用其validation默认值和旧整包镜像。
- [x] SCIENCEAGENTBENCH：原verified固定前三题单轮生成、NPU
  CodeBERT评分和隔离科学执行终态完成；102条原字段及当前CI工件研究消费验证通过。[最终回执](validation/scienceagentbench-fixed3-native-delivery-20261101.json)。程序错误保留，未执行全102题或其他Agent分支；旧validation入口不混用。
  split，不能与当前verified版混跑，不勾原生通过。
- [x] TAUBENCH/τ²：2285个原telecom任务与引用全量准备/预检，新增3个原ACTION类任务公共引用交接预检通过；补齐显式ID/全量入口，125包实测环境快照和pip
  check通过，模型调用0。[CPU回执](validation/tau2-readiness-cpu-20261024.json)。
- [ ] TAUBENCH/τ²：从零重装、修改后入口的原生执行/评分及研究侧工件联调未完成；原3题不重跑、不算全2285题通过。当前模型服务恢复失败，NPU队列未恢复；CPU准备继续。
- [x] TRACELAB：原压缩文件SHA、665453条调用元数据全字段/顺序核查及前三条索引回查完成；20份官方回放源码/示例/许可文件已在服务器取得并逐文件核对Git
  blob。[数据回执](validation/tracelab-cpu-audit-20261024.json)、[官方入口准备](validation/tracelab-official-replay-preparation-20261024.json)。
- [x] TRACELAB：服务器全665453条原元数据索引、首尾6条回查完成；官方DuckDB资产直接流式传至服务器并核验发布SHA及字节数，本地未保存数据。见[索引回执](validation/tracelab-full-index-20261024.json)及[资产回执](validation/tracelab-duckdb-original-20261024.json)。
- [x] TRACELAB：未改官方转换器、独立Python环境按上游锁版本安装NumPy/DuckDB；完整原DuckDB只读转换两次，各665453行，CSV字节一致、原数据库未变，模型调用0。[转换回执](validation/tracelab-official-csv-preparation-20261024.json)。
- [x] TRACELAB：取得官方回放说明推荐的独立enwik9文本资产，10亿原始字节、发布MD5/SHA1、完整UTF8解码及ZIP
  CRC核验通过；不增加数据集清单条目，不把它冒充原轨迹prompt。[辅助资产回执](validation/tracelab-independent-text-pool-20261024.json)。
- [x] TRACELAB：隔离Rust1.90.0、原Cargo.lock构建通过，绑定现有OpenSSL3.6.4、不改源码/锁；全8058会话/665453轮及固定字典序前三会话完整20轮的官方CPU
  dry-run通过，模型调用0。[构建回执](validation/tracelab-original-locked-build-20261024.json)、[dry-run回执](validation/tracelab-original-dry-run-20261024.json)。
- [x] TRACELAB：独立CPU诊断程序直接引用未改的上游Rust分词器/token池函数，当前Qwen3.5
  tokenizer加载及默认1亿token池构建通过，未降低池容量、模型调用0。[诊断回执](validation/tracelab-original-token-pool-cpu-20261024.json)。回执依据SSH失败前完整采到的结构化stdout恢复，未冒充SCP成功；这不是原回放器或NPU运行。
- [ ] TRACELAB：服务缓存明细探测、真实NPU回放未完成。全量336898轮的完整输入+原输出预算超当前131072，保留阻塞，不跳题；固定三会话20轮容量可容纳但尚未调用模型。新实例已核验连通；近期原生响应的缓存明细为null，官方缓存探测仍未运行，不绕过探测或修改正在使用的服务。[前提复核](validation/runtime-prerequisite-recheck-20261025.json)。来源继续未勾，运行边界见[服务轨迹说明](ORIGINAL_SERVING_TRACES.zh-CN.md)。
- [x] TOOLACE：原11300条system/历史对话、原轮次与工具描述、全量索引及首尾6条回查完成，保持历史对话资产身份。[CPU回执](validation/toolace-corpus-cpu-audit-20261024.json)。
- [ ] TOOLACE：服务器新取固定版数据卡网络不可达，不能将目录中的预期SHA当下载验证；没有附带可执行工具环境/任务oracle，不编prompt冒充Agent原生评测，来源级继续未勾。
- [x] JSONSCHEMABENCH：固定CI包与登记的三个原schema
  prepare→resolve完成，旧原始请求、two-shot提示与官方validator复核一致，保留两条不符合约束的输出。[CPU回执](validation/jsonschemabench-cpu-audit-20261024.json)。
- [x] JSONSCHEMABENCH：原11配置声明范围内原生bench/evaluator及68视图任务审计完成，官方原数据19100视图行/99处公共及固定研究回查通过；不把旧3schema或抽样交付称全库质量通过。
- [x] LONGWRITER：固定CI数据包、6000条原始消息/独立参考、全量索引和首尾6条回查完成，旧三题响应完整性复核；区间/全量入口及环境边界补齐。[CPU回执](validation/longwriter-cpu-audit-20261024.json)。
- [ ] LONGWRITER：修改后入口新实跑及长度/语义质量评估未完成；本来源是作者训练语料，不冒充LongBench-Write评测。
- [x] LONGBENCH/Qasper：固定CI数据包、200条原始数据/参考、全量准备索引及首尾6条回查完成；旧三题原生请求和官方评分已离线复核；补齐原始区间/全量入口及依赖说明。[CPU回执](validation/longbench-qasper-cpu-audit-20261024.json)。
- [ ] LONGBENCH/Qasper：研究端精确工件交接仍待复核；原197题续跑和官方评分已完成，不再列作原生缺口。来源整体未因此提前打勾，也不扩称其它LongBench子集。

#### ScienceAgentBench verified版原生链路准备

2026-10-30复查：固定官方README指向的`benchmark_verified.zip`分享链接在匿名访问（含官方分享下载参数）下返回HTTP401，未取得文件。只记录访问阻塞，不使用旧版资产替代、不绕过授权，也不将102题元数据与预览算作原生执行完成；继续推进其他来源。

数据沿用`osunlp/ScienceAgentBench@9c6e96c9e74572e979b0930ee735041cef528cb7`的原`verified`文件，102题；代码固定为`OSU-NLP-Group/ScienceAgentBench@c26e151ed601ba109dc4d35e057ff8e73fec469d`。原适配保留task_inst、domain_knowledge、dataset_folder_tree和dataset_preview；gold_program_name与eval_script_name仅作参考，其余instance/domain/源码及输出路径字段保留元数据。不要把整个envelope或参考侧直接当模型输入。

原数据URL及SHA可用`workload-data catalog download-plan SCIENCEAGENTBENCH`查看。配置`{"sources":[{"source_id":"SCIENCEAGENTBENCH","data_root":"/private/ScienceAgentBench"}]}`即可用现有pipeline全量准备/回查；数据目录须保留原`data/verified-00000-of-00001.parquet`路径。本次102条均实测回查，不仅前三题，也未修改源Parquet。

[发布方当前说明](https://github.com/OSU-NLP-Group/ScienceAgentBench/tree/c26e151ed601ba109dc4d35e057ff8e73fec469d)要求verified版使用配套`benchmark_verified.zip`，并明确不得在线重新分发解包数据。该包中原前三题的完整科学数据目录、参考程序、原评价脚本及rubric现已取得并核验，14文件1006008字节；其余题资产尚未取得。预览文字不能替代真实数据文件，旧benchmark压缩包不能冒充verified修正版。

原生生成入口为上游run_infer.py/agent.py，评估可使用上游`evaluation.harness.run_evaluation`容器化路径，直接Conda评估为另一官方路径。当前两份直接脚本run_infer.py和run_eval.py仍硬编码`validation`
split；后续必须明确将生成和评估同时绑定原verified版本，并核验instance ID及配套评价脚本，不能静默运行不同split或回退main。原源码未修改，本次未执行任何生成程序或评分。

环境不能写成只装一个requirements就可用：上游主Agent和科学程序评估环境分开，直接路径使用Python3.10及独立sci-agent-eval环境；任务还会配置各自的科学依赖，CodeBERT和视觉judge也是评分链路的一部分。专用CPU的独立Docker已恢复并完成无网络、只读根、内存及PID限制的容器控制；科学任务镜像尚未构建，不能据此计原题通过。本地NPU模型、辅助评分模型、原框架模型表/上下文和loopback配置尚待适配与验证；不得在全局推理环境执行上游会卸载/同步包的评估脚本，不发默认外部付费模型请求，不删指标以凑通过。

2026-11-01依赖核查：官方`code_bert_score==0.4.1`源码分发包已按PyPI
SHA256取得并静态核验；Python默认评分模型为`neulab/codebert-python`、第11层。评分API允许显式device，但无CUDA时默认CPU，后续必须显式绑定并验证NPU，不能直接运行默认设备。官方requirements的`litellm==1.52.8`精确版本查询返回HTTP404，当前查询的包索引未列出该版本；历史发布及失效原因未确认，也未静默安装替代版本。原Agent的真实模型tokenizer和价格绑定仍待补，不将Qwen冒称GPT或填写虚假价格。[依赖与隔离环境准备记录](validation/scienceagentbench-original-library-audit-20261101.json)仅证明准备，不是原生任务交付。

- [x] ScienceAgentBench辅助模型准备：原Python评分模型`neulab/codebert-python`固定版本的8份原文件、502184777字节已取得，逐份核对发布方Git/LFS身份，权重仅保存于NPU共享盘，本地磁盘不落权重。尚未加载模型或运行评分，不算原题通过或来源交付。

- [x] ScienceAgentBench辅助评分环境准备：独立虚拟环境安装原`code_bert_score==0.4.1`及作者Docker配方指定的`transformers==4.46.3`，包依赖和导入检查均退出0；4份包内Python源码与官方原始分发包逐字节相同。该原分发包内部`__version__`仍写0.3.5，已核实不是换包。新Python3.12及厂商Torch2.10只作明确披露的NPU兼容环境，未初始化NPU或加载模型。30份原作者源码和102题原verified元数据也已从已有CPU副本转入NPU服务器并核验，未重新下载、不经本地磁盘；科学任务数据和参考程序仍在隔离CPU侧，未混入模型输入。

- [x] ScienceAgentBench原基线输入控制：使用未改写的原任务格式化/提示函数及现有14B模型的真实tokenizer，在CPU核验全部102题；原单轮、不加知识、不自调试基线的完整聊天输入为280至21059
  token，均小于原26000输入预算，预留2000输出不变。仅为未裁剪输入和真实tokenizer的控制，不等于原LiteLLM运行、模型生成或任务评分；没有初始化NPU。

- [x] 专用CPU磁盘清理：已完成来源LiveCodeBench的原始9.38GB文件逐次核对官方SHA256及未打开/未映射状态，清理两个既有硬链接；首个链接删除未释放空间，查明并清理同任务原消费者留下的最后链接后，实测释放9375645696字节，空余20745592832字节。原消费索引、审计、生成结果、评分及环境均保留；原始数据可按固定官方版本重新获取，未扩卷。这是缓存退役，不是新增数据集验证。

##### ScienceAgentBench原版辅助指标NPU入口

- [x] `scripts/scienceagentbench_codebert_npu.py`在卡4完成一次明确标注的原参考自匹配控制：原函数处理canary前缀，3次原模型NPU前向，控制值0.9999998807907104。未执行生成程序，未生成模型回答，不是科学任务得分或来源完成。显存增量实测采样最高492MiB，原9B/14B服务身份不变，租约已释放；保存结果独立复核不重跑模型。[控制回执](validation/scienceagentbench-codebert-original-npu-control-20261101.json)。

该入口只补原CodeBERTScore辅助评分，不替代原科学执行、成功率或视觉评价。环境沿用上文隔离版本，并使用原Python模型固定版本`c4ff459e2a91d002d9bb4ab8cc2b7ee7125ef213`；入口核对8份模型文件、4份原评分包源码和原`compute_scores.py`身份，禁止CPU/CUDA回退。运行前先取得空闲资源窗口，入口不会替使用者停服务或协调共享卡。

输入JSON为记录列表，每项仅含`task_inst`、`pred_fname`、`gold_fname`：分别是原任务指令、实际生成程序路径、原参考程序路径。原参考仅供评分，不能放进生成提示。任务、代码、权重和输出保存在执行者私有目录，不提交到仓库。

```bash
python scripts/scienceagentbench_codebert_npu.py \
  --original-source /private/ScienceAgentBench/compute_scores.py \
  --model-dir /private/codebert-python \
  --records /private/actual-prediction-records.json \
  --output-dir /private/new-codebert-result \
  --device npu:0
```

使用原作者固定源码版本，`--output-dir`必须尚不存在；生成程序与科学评价另行接原容器流程。本次真实NPU证据仅覆盖辅助组件控制，尚未覆盖实际生成答案的评分或完整科学任务。

##### ScienceAgentBench原单轮本地基线与科学执行进展

- [x] 原前三题`1/2/3`已用现有`Qwen/Qwen3-14B`完成真实NPU生成，原完整消息、代码提取、temperature=0.2、top_p=0.95及2000输出预算保留；输出701/359/329
  token。原GPT-chat协议明确绑定本地非思考模型，不冒称Claude/GPT论文成绩；价格仅记为`null`，不伪造API单价。原作者不加知识、不自调试分支的3个方法沿用，未开放宿主机安装/调试代码路径。
- [x] 三份实际生成程序完成原CodeBERTScore
  NPU评分，9次模型前向，原值0.7892104387/0.7659464478/0.8183770776保留；不是此前自匹配控制，也不是科学任务成功率。生成程序未重跑，参考仅在生成结束后供独立评分。
- [x] 原ARM64科学基础配方的软件安装完成；磁盘保护中断后复用已提交安装层，只补原配方最后一步，不重装、不扩卷、不删旧环境。原Python3.10环境依赖/导入通过，第2/3题原instance配方已构建。第2题补齐原XLSX数据必需的`openpyxl==3.1.5`；上游pipreqs不能推断pandas动态可选reader，首次缺依赖记录保留，不改模型程序。
- [x] 实际容器控制验证只读根、无网络、no-new-privileges、UID1000执行、参考不可读、数据可读及结果可写。原`compute_scores.py`不改；辅助F1经原任务/程序/参考SHA绑定复用真实NPU结果，原900秒执行预算、原成功判定及结果tuple保留。只增加非root执行身份和缓存目录隔离，未在CPU重新计算神经辅助分数。
- [x] 第2/3题到达原程序错误终态，原结果分别为`(valid=0, CodeBERT=0.7659464478, success=0)`和`(0, 0.8183770776, 0)`：程序将set作为pandas列名、预测特征与训练不一致。原成功评价脚本因程序先失败而未进入；不改答案、不降校验、不换题凑通过。第3题不重复执行；第2题只在补reader后重跑科学程序，不重跑LLM或CodeBERT。[实际进展回执](validation/scienceagentbench-original-fixed3-native-progress-20261101.json)。
- [x] 第1题按官方科学执行流程在专用CPU隔离容器执行；任务内部科学网络允许使用CPU，LLM与CodeBERT评分仍使用NPU。原程序因`from_smiles`参数错误进入官方失败分支，原始错误及评分保留，不修改生成程序凑通过。
- [x] 公共科学执行编排入口已用第2题保存的真实生成程序完成`prepare`和`run`实跑，二者退出码均为0；保留原程序错误和原始评分，未重新调用生成或辅助评分模型。
- [x] 前三题保存结果已跨端复核：原12字段、生成程序、NPU评分回执、原评分源码及各题科学输入文件身份一致，三题原始成功标记均为0。汇总见`validation/scienceagentbench-fixed3-saved-audit-20261101.json`；本次审计新增模型调用和程序执行均为0。
- [x] 当前CPU
  CI工件已在服务器隔离安装，78份包内文件与提交一致；全部102条原始记录的12字段完整往返，公共端和研究端逐条读取一致，SQLite和研究消费者均未修改。见`validation/scienceagentbench-current-ci-research-handoff-20261101.json`，新增模型调用0。
- [x] 原verified固定前三题单轮协议交付收口：[最终回执](validation/scienceagentbench-fixed3-native-delivery-20261101.json)。全102题科学执行和其他官方Agent分支另列缺口，不将全量读取作为全量科学执行；三题原程序失败保留，不声称科学任务成功。

生成客户端在执行机隔离环境安装`configs/workload-native/scienceagentbench-local-baseline-requirements.txt`；该文件不是科学执行镜像依赖。固定源码沿用上文版本，元数据必须是原`verified`
Parquet；不要用硬编码旧validation的直接脚本替换。

```bash
python scripts/run_scienceagentbench_original_baseline.py \
  --source-dir /private/ScienceAgentBench \
  --metadata /private/ScienceAgentBench-data/data/verified-00000-of-00001.parquet \
  --tokenizer /private/deployed-model-tokenizer \
  --model Qwen/Qwen3-14B \
  --api-base http://127.0.0.1:8000/v1 \
  --output-dir /private/new-science-baseline \
  --start 0 --limit 3
```

`--full`选择原102题；超出原输入预算会失败，不裁剪。输出目录必须全新，已有程序拒绝覆盖；每次请求前记录在途原位置，响应先私有保存，未知请求不得盲目重放。容器内评分入口`scienceagentbench_isolated_scoring.py`拒绝宿主机或放宽的容器环境。外层公共入口`scripts/run_scienceagentbench_scientific.py`提供`--action prepare`生成原作者镜像配方及私有任务目录，再用`--action run`连接明确指定并核对身份的专用Docker执行；镜像构建仍由使用者按生成配方完成。第2题已验证这两个公共动作，不能据此代替来源总验收。

在专用Linux CPU执行机上准备任务，所需原始资产均保存在仓库外；`--program-root`指向已保存的生成程序目录，`--npu-receipt`指向上述真实程序的NPU
CodeBERT评分回执：

```bash
python scripts/run_scienceagentbench_scientific.py --action prepare \
  --source-dir <固定作者源码目录> --metadata <原verified.parquet> \
  --asset-root <固定原科学资产根目录> --program-root <保存的生成程序目录> \
  --npu-receipt <真实NPU评分回执.json> --base-image <专用基础镜像名> \
  --work-dir <外部全新私有工作目录> --start 0 --limit 3 --xlsx-reader
```

源码身份由`scienceagentbench-author-software-pins.json`核对。准备后先构建工作目录内的原基础镜像配方，再按各`<instance_id>/recipe/Dockerfile`构建任务镜像，镜像映射JSON使用原`instance_id`作为键、完整`sha256:`镜像ID作为值。`--xlsx-reader`明确补充原Excel任务缺失的`openpyxl==3.1.5`读取依赖，不改原题或计分。默认ARM64，x86执行机显式使用`--arch x86_64`，不得混用不同架构镜像。

```bash
python scripts/run_scienceagentbench_scientific.py --action run \
  --work-dir <上述私有工作目录> --image-map <镜像映射.json> \
  --docker-host <专用Docker套接字URI> --docker-id <已核验的Docker实例ID> \
  --daemon-work-dir <Docker守护进程所见的同一工作目录>
```

执行客户端与Docker守护进程位于不同容器时，二者路径必须对应同一持久卷中的同一目录；不能直接假定客户端绝对路径可供守护进程挂载。入口核对专用Docker身份，使用无网络、只读根文件系统、受限权限的容器；原生成程序以非root用户运行。已有科学结果拒绝覆盖。程序报错仍保留官方结果四元组及日志，不改程序或换题；基础设施失败则明确失败，不算完成。

#### OVO-Bench原生在线视频路径准备（未完成来源交付）

- [x] 官方Qwen2-VL-7B-Instruct固定版本的14个必需模型文件已在NPU共享盘取得，合计16594373111字节，逐份与官方Git/LFS身份一致；未下载权重到本地，未更改卡6现有14B服务。

- [x] 官方offline协议的12类原前三题、36个完整case及全部103个原响应位置已在物理卡4真实NPU运行，103/103返回，无推理错误。BF16、原最多64帧及128-token预算保持；短片段仍按原`min(64, len(video)-2)`规则，不强行补足64帧。NPU采用明确披露的eager注意力绑定，不冒称CUDA/FlashAttention2硬件复现。

- [x] 保存结果完成原评分及独立复核：原Qwen2-VL包装返回单元素文本列表，官方REC评分器期望字符串，首次评分TypeError保留；只在评分副本中将`[text]`无损解包为完全相同的`text`，不提取答案、不改任何文字、标签或计分函数。未修改官方评分器，原生成输出不变，原任务及全部阶段顺序逐项一致；原宏平均47.04%，独立复算47.0417853751187%，按作者打印精度一致。评分和审计均0次新模型调用。回执位于共享任务根`ovobench-lossless-scalar-original-score-32068-001/qualification-public.json`及`ovobench-original103-independent-audit-32068-001/qualification-public.json`。该数值只是固定抽样可用性结果，不是全量质量或引擎性能结论。

- [x] 公开NPU入口支持固定抽样及`--full`；服务器核验当前入口的103份原媒体身份快照、1640题/3035片段全量计划，模型调用0。回执`ovobench-current-public-media-32068-001/qualification-public.json`。模型修订、隔离视频环境、原始媒体、抽样/全量与保存输出评分命令已写入使用说明。

- [x] 新CPU
  smoke运行41202187413成功；实际工件13042688731独立安装，78份载荷与Git/包内字节一致。复用1640条原索引，12类各前三题共36条公共/未修改研究消费端完整回读通过，数据库与研究代码不变；回执`/root/ovobench-actual-ci-wheel-20261031-002/qualification-public.json`。0新模型调用，输入消费与上文原生推理分别举证。

- [x] 通用CI41202187376的测试、lint均通过，CPU
  smoke41202187413通过；来源总勾仅覆盖明确的offline原生协议。[完整交付证据](validation/ovobench-offline-native-delivery-20261031.json)。103次有效生成不重跑，online路径与全量生成单列缺口。

- [x] NPU共享盘原软件与注释准备：复用专用CPU已取得的45份固定官方软件，逐份与仓库固定SHA256/字节数一致；原注释918443字节、1640个case原样转入`ovobench-original-inputs-32068-001`，软件和注释转移后再次逐份核验，Python仅AST解析。回执为该目录`qualification-public.json`。未执行模型、未取得原生成绩，不以输入准备计来源完成。

- [x] NPU侧原媒体准备完成：从本任务专用CPU既有103份原片段向NPU共享目录`ovobench-existing-original-media-32068-001`的流式传输全部完成，逐文件按已登记字节数和SHA256核验后才转正式文件。公共`verify_ovobench_original_media.py --require-all`在NPU侧再次核验全部103份exit0；回执为共享任务根`ovobench-public-npu-entry-32068-001/qualification-public.json`。原CPU文件保留，不经本地磁盘、不重复下载或完整解码；只计输入准备，不计原生推理/评分完成。
  首份`0.mp4`第一次接收短读，保留150962176字节的partial，与CPU原文件同长度前缀SHA256一致后仅补2346712字节；完整153308888字节及登记SHA256复核通过，已转正式文件。早期6/103检查点已由上项最终103/103核验覆盖；真实观察到的末尾短读和SSH断连保留，不发布失败partial，复核原前缀后只续传剩余字节。队列对已确认短读最多续传3次，身份不符即停；`1210.mp4`已实际经此路径完整核验，不隐藏成首轮成功。

- [x] 隔离`qwen-vl-utils==0.0.8`工具目录已准备并在NPU服务器导入通过：固定PyPI原wheel字节数/SHA256校验，未改全局包、未加载模型、未解码视频。此版本是显式候选兼容配置，非作者指定的依赖锁定；回执`ovobench-video-utils-overlay-32068-001/qualification-public.json`位于共享任务根。不计模型运行或合格完整环境。

- [x] 官方Decord 0.6.0隔离构建和导入通过：服务器直连GitHub clone失败保留；通过有界RAM中转取得固定提交及其两个固定子模块的构建代码。FFmpeg
  4.4.8官方源码、独立前缀构建和Decord
  CPU构建均已完成，导入exit0；每个构建最多2个低优先级CPU任务，不占NPU、不全局安装。回执为共享任务根`ovobench-ffmpeg-isolated-32068-001/qualification-public.json`及`ovobench-decord-build-32068-001/qualification-public.json`。这些不是模型环境或原生题目完成证据。

- [x] 原视频64帧CPU前处理控制通过：复用已校验的原`0.mp4`，核验后执行未修改的官方`get_max_frames`函数及隔离`qwen-vl-utils`视频路径，保持64帧、`360×420`最大像素设置，实际得到CPU
  float32张量`[64,3,336,448]`。原媒体及作者代码哈希未变，不落原帧文件；回执`ovobench-original-frame-control-32068-001/qualification-public.json`。控制文本不是数据集题目，不计原生case、模型processor加载、NPU推理或评分；这些仍待验证。

- [x] 固定12类各原前三题的36个完整case所需103份原片段已全部取得并公开登记身份，共3520157133字节；全部文件重算SHA256，通过旧完整解码回执核验489216视频帧，未重下载或重复解码。公共CLI以`--require-all`核验全部103份exit0；完整回执`/root/ovobench-fixed-original-media-all-20261031-001/complete-fixed103-registration-public.json`及`/root/ovobench-public-registered103-20261031-001/qualification-public.json`。重复分块bin缓存余量为0，仅清理已核验重复缓存，完整原文件和全部证据保留；0模型调用，不打来源总勾。

- [x] 未修改的官方offline评分器完成参考正确/全部None两项控制；12类36-case全部103响应位置均计入，None计0不丢弃，原宏平均不变。回执`/root/ovobench-original-score-reference-controls-20261031-001/qualification-public.json`及同目录`slot-count-public.json`。仅评分器控制，模型未加载，不是Qwen模型成绩。

- [x] 专用CPU核验当前官方固定版本45份原软件、79170字节，20份Python只做AST解析；复用公共作者获取器`--source ovobench --action verify`退出0。回执`/root/ovobench-original-software-20261031-001/preparation-public.json`。该软件准备阶段不导入执行原代码，模型调用0，不取得视频或发布方模型结果；后续媒体及评分控制另项记录。

- [x] README补充官方软件准备、推理及评分入口，明确三种时间模式、12种任务、向前任务全部`test_info`阶段、原64帧上限/128-token预算及CUDA到NPU环境缺口；代码与数据许可分别核对。

- [x] 合格NPU模型、原帧处理、36-case/103调用流程及真实输出原评分已核验，见本节原生执行与独立审计回执；消费交接另列，不重复生成。

- [x] 原注释918443字节整文件取得并校验；1640个case/12类任务/644个视频名、全部1567个向前阶段经公共候选适配器及`iter_workload`逐字段完整还原。`count/type`等参考不入模型输入，不改题、不丢阶段。回执`/root/ovobench-original-public-adapter-20261031-002/qualification-public.json`，0模型调用。

- [x] 固定抽样原生NPU推理及真实输出评分闭环已完成；早期元数据工件/读取证据保留其原范围，新公开入口交接仍单独核验。

- [x] 成功CPU
  CI运行41151526268的实际工件13022928107独立安装，75个Git/包内/安装载荷字节一致；复用1640条原索引，12类原前三条共36条公共/未修改研究端完整回读通过。回执`/root/ovobench-actual-ci-wheel-20261031-001/qualification-public.json`；0模型调用，不恢复研究任务，不计视频原生完成。

- [x] 官方固定媒体版本20分卷大小/LFS身份已登记并由计划CLI展示；原视频总46341744640字节，预切分总153277153280字节，专用CPU剩余5040644096字节，未启动装不下的整包下载。仅身份登记，不冒称视频取得。

- [x] 原预切分分卷仅取64KB，精确HTTP206/Content-Range及原TAR目录头核验通过。首个文件是非固定抽样的`35.mp4`，57464390字节，未因其容易取得而换题或下载；该首头探测阶段完整视频取得数为0。回执`/root/ovobench-original-media-ranges-20261031-001/first-tar-header-public.json`。此为早期探测记录，后续完整目录和103片段核验另见本节。

- [x] 公共`plan_ovobench_original_media_ranges.py`在专用CPU对实际捕获的368个原头部快照实跑exit0，逐头校验捕获哈希、TAR校验和/连续偏移/原分卷/观察ETag，已定位固定103片段中的16个、553531286字节。仅结构与范围计划，不独立证明发布方整分卷哈希；未定位项明确保留，不换片段，未下载完整视频。回执`/root/ovobench-public-media-range-plan-20261031-001/qualification-public.json`。此为中间快照；最终完整头索引及媒体核验另见本节，不以本快照冒称完成。

- [ ] 澄清代码README数据CC BY-NC-SA 4.0与官方Hub数据卡CC BY-SA 4.0差异；代码MIT单独登记，不以更宽松数据卡推定再分发权限。

- [x] ASI固定原前三题483/484/485输入媒体已齐：完整原文件共37469641字节，公共`--decode`核验全部8556帧。新取得483/484，485不重取、以硬链接复用原文件不额外占空间；各自原帧率保留，不重编码、不更换题目。回执`/root/ovobench-public-media-verifier-20261031-002/qualification-public.json`，0模型调用。该早期子集阶段为3/103；最终全量见本节首项，ASI原生模型执行/官方评分仍未完成。

- [x] 公共`prepare_ovobench_original_media.py`完整成员网络读取在Windows
  RAM中验证：原分卷ETag/精确范围/完整SHA核验通过，8800190字节与服务器保留原文件逐字节一致；不在本地落数据、不新增服务器副本。服务器CLI复用已有原文件也exit0。回执分别为`/root/ovobench-public-media-network-20261031-001/qualification-public.json`和`/root/ovobench-public-media-prepare-reuse-20261031-001/qualification-public.json`。不据此宣称服务器直连网络可用。

- [x] 公共`capture_ovobench_original_tar_headers.py`已实现，每次只取512字节原头部、跳过成员载荷、检查点可继续；RAM真实网络推进2048→2050验证通过。纠正验证脚本输出目录后，服务器CLI真实尝试失败于`Network is unreachable`，最后有效检查点成功保留；首次路径边界拒绝单独保留，不冒充网络故障。回执`/root/ovobench-public-header-capture-20261031-002/qualification-public.json`，未改动/重启主目录任务，0模型调用。

- [ ] 公共目录捕获和片段获取的服务器直连网络环境仍未通过；原主目录已沿用原控制完成（见下项），不重复扫描；本任务通过有界RAM中转完成103片段取得，不等于服务器直连公共下载链路通过。

- [x] 原主目录任务已自然完成：3036个原TAR头、3035个视频成员及两个结束标记，公共规划器`--require-complete`实跑exit0；36个原case所需103片段全部定位，总3520157133字节。仅目录/范围闭环，不重算整分卷哈希、不冒充媒体取得或模型运行。回执`/root/ovobench-complete-original-index-20261031-001/qualification-public.json`。

- [x] 全部固定103原媒体队列已退出0：复用既有3份、补齐100份，原问题和阶段未改变；逐文件保留512MiB空间，验证后只清理重复分块缓存。进度回执`/root/ovobench-fixed-original-media-all-20261031-001/progress-public.json`，完整核验见本节首项；不计原生模型或评分完成。

- [x] 首轮在5份原媒体验证后遇到发布方TLS握手EOF，已核实控制进程退出1后恢复；5份已完成媒体及有效分块缓存复用，恢复时只剩98份，不重新取已通过内容、不换源/换题。公共传输补充最多3次同一原范围重试，仅覆盖临时连接/TLS
  EOF/超时及部分5xx；权限、限流、ETag/身份差异不重试，成功结果仍记录传输失败次数，非模型质量验证。

- [x] 固定原ASI
  case485完整原片段`485.mp4`取得，8800190字节；固定URL/精确原范围和原捕获ETag一致，文件身份登记到既有profile。复用服务器PyAV18.1.0逐帧解码全部2128帧，公共`verify_ovobench_original_media.py --decode`实跑exit0，未重编码/截断/安装依赖。回执`/root/ovobench-public-media-verifier-20261031-001/qualification-public.json`。该最初阶段仅1/103片段准备完成；最终全量见本节首项，官方帧采样、NPU推理、评分均未完成，0模型调用。

- [x] 公共`prepare_ovobench_native_selection.py`在服务器成功生成12类各原前三题的36-case原字段子集，供官方`--anno_path`读取，列出全部103个必需原视频片段，不删向前阶段。实跑复用已安装CI提供侧，原文件不变；回执`/root/ovobench-native-selection-20261031-002/qualification-public.json`。首次输出边界拒绝记录保留，纠正为独立数据目录后通过；未下载视频、0模型调用。

- [x] 候选源码建立1640条引用索引；每种任务原前三条，共36条，经公共和未修改研究消费者完整回读一致。当前研究工作树消费者SHA与服务器副本一致，索引读取前后不变；未恢复研究任务。回执`/root/ovobench-original-index-20261031-001/qualification-public.json`。该候选阶段本身不是实际CI证明，后续实际CI闭环见前项，0模型调用。

#### WavCaps原生音频路径准备（未完成来源交付）

- [x] 四组件各固定原前三条，共12个真实FLAC已取得并通过同一公共CLI完整解码；原CRC、PCM MD5和原时长均匹配。AudioSet保留原`.wav`
  ID，另绑定发布包FLAC；FreeSound保留原`file_name`。四个CLI均exit0，回执：`/root/wavcaps-original-media-public-verifier-20261031-003/qualification-public.json`。仅CPU信号解码，模型调用0。
- [x] BBC固定原前三条`NHU05051026/07066104/07039376`取得并完整解码，原PCM
  MD5与时长核验通过；原字符串ID保留前导零。按原分卷ZIP64目录和盘内偏移取8段共16538564字节，不下载130GB整包，不把不同分卷的ETag误当一个值。旧两组件回执保留于`/root/wavcaps-original-media-public-verifier-20261031-002/qualification-public.json`；四组件现状见上项，不打来源总勾。
- [x] 固定SoundBible原前三条`2219/2218/2217`的真实FLAC已取得：原Hub固定版本ZIP仅取8段共4869230字节，原成员CRC及同一ETag核验；3个文件完整解码、PCM
  MD5匹配原STREAMINFO，均为32kHz单声道，时长与原字段完全一致。公共`verify_wavcaps_original_media.py --decode`实跑exit0。原JSON的`audio`实际是`wav_path`占位符，单独按原ID/原ZIP成员绑定，不改JSON或音频。不冒称完整580MB归档哈希已重算；此项仅说明SoundBible样本。
- [x] 复用公共原始数据接口，四份JSON的403050条逐字段完整还原通过；音频定位信息与caption/原描述/类别/标签分离，12条固定样本不能伪装为聊天请求。使用者缺文件时仍显式报错，不静默跳过。源码回执：`/root/wavcaps-public-adapter-20261031-003/qualification-public.json`；实际安装另见下项。
- [x] 复用既有获取器的原文件校验CLI，四份元数据实际执行通过；源码候选的403050条索引与12条固定公共/未修改研究端完整读取核验通过，旧三文件索引保留且未恢复研究任务。服务器消费端源码与研究工作树版本一致。回执：同CPU目录下`getter-verify-public.json`、`research-handoff-public.json`。原生音频任务仍未完成。
- [x] 实际CI工件13018794405独立安装通过：74份包内软件/资源逐项匹配已提交Git字节和安装文件，403050条原索引只读复用，四份各原前三条共12条公共/研究端完整读取一致，无模型调用。见[元数据CI交接回执](validation/wavcaps-original-metadata-ci-handoff-20261031.json)。这只证明原始元数据的获取与消费链路，不计原生音频交付。
- [x] 取得并校验SoundBible、BBC、AudioSet、FreeSound四份原始JSON，共403050条、174782992字节，保留各组件原前三条完整记录。FreeSound通过现有公共获取器直接下载并验证发布方LFS哈希，以硬链接复用原文件、不重复占用数据空间。完整回执：`/root/wavcaps-original-metadata-20261031-002/complete-metadata-public.json`；原三文件回执、CPU直连TLS失败记录保留，不因网络失败换数据。
- [x] 在专用CPU取得发布方固定版本的6份说明、音频captioning入口/评分器及检索入口，逐份核验Git
  blob和SHA256；3份Python仅做AST解析，未导入执行。服务端回执：`/root/wavcaps-original-source-20261031-001/preparation-public.json`。
- [x] 公共`prepare_wavcaps_original_media.py --cache-root ... --decode`在专用CPU使用保留的原压缩范围完成四组件12条重新解压、身份及PCM校验。回执`/root/wavcaps-public-media-preparation-20261031-001/qualification-public.json`，模型调用0；不重复下载原媒体。
- [ ] 公共有界媒体获取的`--download`网络路径真实核验；已实现精确206/Content-Range、长度和成员身份限制。专用CPU对FreeSound原第一成员的真实下载exit1：`urllib.error.URLError: [Errno 101] Network is unreachable`，尚未取得首成员，未换来源或换题。缓存实跑及HTTP夹具不能代替网络获取成功，不宣称完整自动获取已完成。
- [ ] 补齐专用checkpoint和完整原生依赖，完成原生NPU推理、适用评分和输出消费闭环。四组件媒体准备不等于模型运行；未下载模型，也未调用文本模型替代。

WavCaps是弱标注音频—文本资产；作者captioning示例使用专用音频编码器和BART
checkpoint，不能直接填入Qwen文本服务。原评分器要求五条参考的接口不能靠复制弱标注caption凑足；须按原任务选用真实评价资产和协议。其音频仅限学术研究且保留各源条件，不作默认企业再分发。以上准备不增加来源完成数，机器清单状态为`IN_PROGRESS`。

#### MultiAgentBench原Research续跑

MultiAgentBench续跑状态：原Research固定前三题、5/5/3个Agent及各5轮均保持。最后核查原模型与关键词各4个请求/响应已返回；本地转发器随后因620秒RPC超时退出，原CPU父/子进程仍存活，三题完整结果及评分均未完成。NPU有1个已提交/claimed但未保存响应的模型请求，不能盲目重发；CPU额外排队项不算新完成任务。保留全部原响应和失败，不将环境/排队问题当作原题质量失败。原运行目录`/root/marble-original-research-native-runtime-20261031-003`和精确PID/birth见机器清单`MULTIAGENTBENCH.native_research_execution`；不得因超时自动重启Actor或替换题目。

同一CPU/原客户端的独立网络对照显示：`http://arxiv.org/html/2401.00123`连接超时约9.02秒，HTTPS同路径约2.04秒返回200；原论文工具将检索结果URL直接用于HTML抓取，超时为60秒。此对照不是原题URL或所有慢请求的因果证明。现有任务、题目、工具及模型没有改动；已提出仅为后续新执行制作明确披露的同路径HTTPS兼容副本，等待确认，不热修改运行中的任务。

- [x] 原Research前三题真实启动，首个原生模型与关键词请求已在NPU返回。
- [ ] 三题完整原Agent流程及原评分结束，并完成保存结果核验；当前转发中断、原Actor仍存活，不能将启动、HTTP成功或论文检索当成任务完成。

#### SpokenWOZ原文本DST协议交付

- [x] 从[发布方项目页](https://spokenwoz.github.io/SpokenWOZ-github.io/)当前链接的`ssz1111/SpokenWOZ-Test-Text-Fixed`取得固定版本原`data.json`：52233254字节，发布方LFS
  SHA256匹配，1000完整对话、35564话轮。原文件保存在服务器，后续NPU共享盘副本同样完整校验；传输内存块上限8MiB，不在Windows落盘。
- [x] 固定原JSON首次出现的前三个完整对话`MUL0001/MUL0002/MUL0004`，分别50/44/42话轮；不截取单轮冒充完整对话，不换成功样本。回执：`/root/spokenwoz-original-test-text-20261031-001/acquisition-public.json`。
- [x] 公共适配对全部1000对话/35564话轮逐字段无损还原；原goal、metadata、dialog_act和span_info留在参考侧，原text、tag、words及词时间戳保持不变。公共获取器只读校验、固定前三个完整对话的准备入口均exit0。回执：`/root/spokenwoz-public-adapter-20261031-002/qualification-public.json`。
- [x] 未修改的研究消费端从固定三条索引读回完整136话轮，与公共消费端和原JSON一致；索引不变、研究任务未恢复。回执：同目录`research-handoff-public.json`。这是候选源码验证，不是CI安装工件或原生模型结果验证。
- [x] 成功CI
  `41172187985`产物`13032060104`已在专用CPU独立环境实际安装，77份Git/wheel/安装文件逐份相同；公共与未修改研究消费端复用原索引读回三完整对话136话轮，索引未变。回执：`/root/spokenwoz-actual-ci-wheel-20261031-001/qualification-public.json`。没有重新下载、准备或模型调用，不把该数据接入核验冒称原生推理。
- [ ] 尚缺的扩展模式：原音频和语音交互；另保留新鲜公共下载的服务器网络实测缺口。原文件已通过有界中转取得并全包校验，不把“此服务器直接下载未实测”写成“原数据未取得”。首次文件角色注册错误及stage001失败保留。
- [x] 固定三个完整对话已完成官方文本DST流程：Qwen3-14B作为明确披露的NPU替代模型，68个原始用户话轮请求全部`stop`、0请求失败；原提示、完整可见历史、参考及实际响应独立核验通过。未改变原客户端省略采样/预算参数的方式，仅显式声明非思考模型绑定。官方JGA与独立累计状态核验一致：4/68，约5.88%；保留低分，不换题，不与原GPT3.5或隐藏榜单比较。
- [x] 原预处理在共享盘遇到写后未刷新的中间JSON读取失败；保留原文件和stage001失败，兼容副本仅增加两处`f2.flush()`，题目、提示、顺序、参考和评分不变。原始68次生成没有重跑。回执：共享任务根`spokenwoz-original-DST-fixed3-32068-002/{launch-public.json,saved-audit-public.json}`。
- [x] 公共原生准备`prepare_spokenwoz_original_dst.py`与运行/保存核验`run_spokenwoz_original_dst.py`已提供；专用CPU实际执行原准备与`--verify-saved`，68份原提示/参考与既有NPU响应全部一致，评分4/68再次核对，无新模型调用。回执：`/root/spokenwoz-native-public-entry-20261031-002/qualification-public.json`。原准备成功回执复用，首次公共入口把输出放入源码目录被路径保护拒绝的记录保留，未降低保护。
- [x] 公共新生成循环已在不重叠的原第3至5号对话真实完成：120话轮/60请求全部`stop`、0请求失败，原官方评分和独立核验均6/60。当前公共`--verify-saved`再次通过，不新增模型调用；原前三例68次生成及4/68结果保持不变。真实DST调用累计128，不冒充128个数据集。回执：共享任务根`spokenwoz-public-native-saved-audit-32068-001/qualification-public.json`。
- [x] 公开区分生成调用和只读核验调用：初版运行报告继承了核验的`new_model_calls=0`，但已有正确的`actual_model_calls=60`。已修复计数字段，单独保留更正报告，原报告、请求、响应和评分未改写、不重新生成。修复后的保存核验新增调用数仍为0。
- [x] 原文本DST协议已按正式范围收口，见[交付记录](validation/spokenwoz-text-dst-native-delivery-20261031.json)。当前公开保存核验在服务器重查60请求通过，原运行目录136个文件哈希未变，新增模型调用0；实际CI提供方和原研究输入消费证据也已复查。
- [ ] 可选全量评测与扩展模式尚未完成：1000对话/17782请求全量生成、原音频/语音及其他响应协议。它们不阻塞已声明的文本DST协议交付，但不计对应模态覆盖，不声称复现隐藏榜单或全部来源能力。

当前公开测试文件与旧Git说明中的隐藏测试集必须区分，不能声称复现隐藏榜单。官方既有文本DST/响应基线，也有语音—文本模型；文本基线运行不能证明音频或双工流程跑通。原数据遵循发布方CC
BY-NC4.0，不在公共PR再分发。

在仓库根目录使用统一入口；所有原数据与索引保存在服务器外部目录，`--limit 3`选择前三个完整对话，不是前三个话轮。读取完整对话仅获得原生任务包，每次模型调用的历史可见范围仍由官方runner负责。

原生DST入口现支持完整对话范围`--start N --count M`及`--full`，不把默认前三个验证对话当成全部可运行范围。专用CPU已实测全量计划：1000完整对话、35564话轮、17782个原DST请求；另按原顺序准备了不重叠的第3至5号对话`MUL0012/MUL0015/MUL0019`，120话轮/60请求。回执：`/root/spokenwoz-range-full-entry-20261031-001/qualification-public.json`。全量计划不等于全量模型执行；原前三对话68份有效生成仍复用。

```bash
python scripts/prepare_reused_research_assets.py SPOKENWOZ
python scripts/prepare_reused_research_assets.py SPOKENWOZ --mode download --data-root /srv/workloads/spokenwoz
python scripts/prepare_reused_research_assets.py SPOKENWOZ --mode verify --data-root /srv/workloads/spokenwoz
PYTHONPATH=workload-preparation/src python scripts/prepare_reused_source_records.py SPOKENWOZ --data-root /srv/workloads/spokenwoz --output /srv/workloads/spokenwoz-first3 --limit 3
```

#### Full-Duplex-Bench原生准备

- [x] 固定发布方版本`3e799c45a045256f47d5f1c9cda90157e2d2ec9e`的`v1_v1.5`离线协议，专用CPU取得16份原说明、评分软件和官方样例JSON，逐份校验Git
  blob与SHA256；未在Windows落盘原始资产。
- [x] 未修改的`eval_pause_handling.py`和`eval_smooth_turn_taking.py`分别对官方自带的1份输出样例实际运行，均exit0。回执：`/root/full-duplex-original-scorer-controls-20261031-001/qualification-public.json`。这是发布方输出的评分器控制，不是本项目模型结果，也不是固定原前三题的验证。
- [x] 公共评分器准备入口`prepare_full_duplex_original_scorers.py --verify-only`已在专用CPU核验现有原11文件子集，exit0，无新下载、无重复评分。回执：`/root/full-duplex-public-source-entry-20261031-001/qualification-public.json`。新鲜`--download`的服务器网络路径未通过实测，不冒称整仓或原音频已取得。
- [ ] 原数据与音频取得、原始时序与媒体关系的provider接入。
- [ ] 合适的NPU双工语音模型、原生输入/输出音频、ASR和适用完整评分、输出消费闭环。模型调用仍为0，总项不打勾。

公共入口在仓库根目录运行，原资产仅写服务器外部目录；评分脚本依赖发布方要求中的`tqdm`，不需要模型、ASR或外部judge。官方样例输出不代表本项目模型输出。

```bash
# 默认离线计划，不下载；显式下载只取11份固定原文件，不取整仓/node_modules
python scripts/prepare_full_duplex_original_scorers.py
python scripts/prepare_full_duplex_original_scorers.py --download --output /srv/workloads/fdb-scorer-subset
python scripts/prepare_full_duplex_original_scorers.py --verify-only --output /srv/workloads/fdb-scorer-subset
python /srv/workloads/fdb-scorer-subset/v1_v1.5/evaluation/eval_pause_handling.py --root_dir /srv/workloads/fdb-scorer-subset/v1_v1.5/evaluation/example_data/pause_handling_example
python /srv/workloads/fdb-scorer-subset/v1_v1.5/evaluation/eval_smooth_turn_taking.py --root_dir /srv/workloads/fdb-scorer-subset/v1_v1.5/evaluation/example_data/smooth_turn_taking_example
```

官方的v1/v1.5离线、v2实时考官、v3语音工具调用是不同协议，不得把上述两个样例评分推及其他版本；也不得以文本问答、拼造音频或CPU神经推理替代原生流程。官方协议入口：[Full-Duplex-Bench](https://github.com/DanielLin94144/Full-Duplex-Bench)。

#### TOOLACE原始语料使用入口

来源固定为`Team-ACE/ToolACE@6bda777c88d21e5a204703c1ee45597a8fa4f734`，原`data.json`为37154735字节、11300条，SHA256为`ba12c083fca7e8da48c67ad5b895e495447da7c66e39a2e19742c082e6cb537e`。使用数据组件的`workload-data catalog download-plan TOOLACE`查看原始URL；将自行取得并校验的文件放在私有目录，不放入Git。JSON读取/准备不需要模型、工具服务或科研观测环境，数据包安装及固定工件核对见[安装说明](INSTALLATION.zh-CN.md)。

```python
from pathlib import Path
from vllm_hust_workload_data import iter_workload

for record in iter_workload("TOOLACE", data_root=Path("/private/ToolACE")):
    original = record.task["input"]  # 原system和完整conversations，非待生成请求
```

全量准备配置为`{"sources":[{"source_id":"TOOLACE","data_root":"/private/ToolACE"}]}`，使用现有pipeline的plan/run及resolve_input入口即可。`system`中的工具描述和所有历史assistant/tool消息保留原样；`outcome_record()`为None符合资产契约，不从历史对话中虚构独立任务参考。`request_body()`明确拒绝将其当作原始API请求。后续若需预测某一轮、执行工具或评估模型，须另行声明并验证实验协议，不能把本次数据准备称作已完成这些原生流程。

#### RULER 子配置执行进度（不增加来源完成数）

RULER仍是113项中的一个来源，以下子配置的勾不计作新增数据集，也不把部分子配置完成称为原13任务全部完成。模型沿用既有双NPU Qwen3.5-27B配置，原题、完整输入及原输出预算不改。

原首个500条的保存请求/响应、原字段和原评分已完成[终态审计](validation/ruler-niah4k-full500-native-audit-20261025.json)，公共入口仅对这500条做无模型CPU回放，没有重跑。其余5组使用公共入口的15条真实请求及原评分也已[逐条审计](validation/ruler-public-first5-native-audit-20261025.json)，不是读取证明。

| 子配置               | 固定验证范围    | 原生执行与原评分                                               |
| -------------------- | --------------- | -------------------------------------------------------------- |
| niah_single_1 / 4K   | 全500条         | [x] 500条已审计，原128 token预算                               |
| niah_multikey_2 / 4K | 原物理行0、1、2 | [x] 3条已审计，原128 token预算                                 |
| niah_multikey_3 / 4K | 原物理行0、1、2 | [x] 3条已审计，原128 token预算                                 |
| niah_single_1 / 32K  | 原物理行0、1、2 | [x] 3条已审计，完整输入未裁剪                                  |
| niah_single_1 / 128K | 原物理行0、1、2 | [x] 3条已审计，实际输入130930至130939 tokens，原128 token预算  |
| cwe / 4K             | 原物理行0、1、2 | [x] 3条已审计，原120 token预算                                 |
| vt / 4K              | 全500条         | [x] 500条请求、响应及原评分已审计，原30 token预算，原评分100   |
| fwe / 4K             | 全500条         | [x] 500条请求、响应及原评分已审计，原50 token预算，原评分57.33 |
| qa_1 / 4K            | 全500条         | [x] 500条请求、响应及原评分已审计，原32 token预算，原评分24.8  |

上述三组新增1500条的[原生执行审计](validation/ruler-short-output-native-audit-20261025.json)不调用模型重跑；累计与前述515条合计2015条已审计。低分如实保留，不替换失败回答以凑分，也不据此声称引擎性能提升。

字段完整性复核发现旧适配未保留原生`length_w_model_temp`与部分NIAH任务的`token_position_answer`；同时`index`可能是答案字符位置，不能作为输入侧元数据暴露。正在修复为完整保留长度字段、答案位置归参考答案侧，并对未映射字段拒绝绑定。旧绑定/消费证据仅证明当时声明的字段投影，不证明所有原生字段完整保留；现有原生模型运行不受此问题影响，不重复执行。

后续8组生成文件合计4000条均已按原生成器和字段完成[CPU准备审计](validation/ruler-eight-profiles-preparation-20261025.json)，包括128K原500条，不把准备当执行。另已完成[显式绑定及消费链路](validation/ruler-eight-profile-binding-chain-20261025.json)：8组全4000条绑定/索引、各首尾3条共48条公共回查、研究侧索引导入和直接prepare/resolve各24条。绑定CLI当时是候选软件，数据组件是固定CI50474ec0工件，研究默认锁未改；不迁观测或研究分析。

剩余六项已完成[原生执行审计](validation/ruler-remaining-six-native-audit-20261025.json)：

| 子配置               | 固定验证范围    | 原生执行与原评分                                      |
| -------------------- | --------------- | ----------------------------------------------------- |
| niah_single_2 / 4K   | 原物理行0、1、2 | [x] 3条审计，原预算128，评分100                       |
| niah_single_3 / 4K   | 原物理行0、1、2 | [x] 3条审计，原预算128，评分100                       |
| niah_multikey_1 / 4K | 原物理行0、1、2 | [x] 3条审计，原预算128，评分100                       |
| niah_multivalue / 4K | 原物理行0、1、2 | [x] 3条审计，原预算128，评分100                       |
| niah_multiquery / 4K | 原物理行0、1、2 | [x] 3条审计，原预算128，评分33.33                     |
| qa_2 / 4K            | 全500条         | [x] 500条审计，原预算32，评分45.2；显式生成器兼容修复 |

完整[218篇essay资产](validation/ruler-original-essay-corpus-20261025.json)与[原HotpotQA对象](validation/ruler-original-hotpot-object-20261025.json)已取得，均只存服务器。QA2原生成器存在可复现的控制流死循环；[独立兼容副本及准备核验](validation/ruler-qa2-generator-compatibility-20261025.json)保留前251条逐字节等价记录，并核验全部500题的必要材料、答案、题目和预算未改。不得称为官方生成器完全未修改。

累计2530条原生请求已审计，覆盖13种任务及额外32K/128K配置，不声称完整长度网格。[最终CI工件消费核验](validation/ruler-fixed-ci-research-handoff-20261025.json)完成15组7500条原字段重建、公共回查90条、研究索引导入及直接prepare/resolve各45条；CPU
smoke、tests和lint全部通过。[交付回执](validation/ruler-native-delivery-20261025.json)记录原生执行、固定工件与显式兼容边界，RULER来源主勾已勾；研究默认依赖锁未改，观测与分析未迁入。

#### GPQA 子流程进度（不增加来源完成数）

| 子流程                  | 固定范围                             | 证据状态                                                                        |
| ----------------------- | ------------------------------------ | ------------------------------------------------------------------------------- |
| 原CSV获取/准备          | main448、diamond198、extended546     | [x] 原压缩包及三个成员摘要核对，真实公共CLI准备验证                             |
| 原字段及seed0顺序       | 三split全部1192行                    | [x] 与原CSV及作者load_examples逐行核对；split重叠不去重计独立题                 |
| 候选公共/研究消费       | 公共回查首尾各3；研究每split原前三行 | [x] 公共18条、研究索引导入9条及直接prepare/resolve9条                           |
| Diamond原zero-shot      | 原全部198位置                        | [x] 197次真实生成，原索引69跳过计错，原评分103/198；4条length结果保留           |
| Main原zero-shot         | 原全部448位置                        | [x] 447个生成结果已审计，其中435次新增推理、12次等价保存响应复用；原评分222/448 |
| Extended原zero-shot     | 原全部546位置                        | [x] 545个结果逐条审计；522次新增推理、23次等价复用，原评分290/546               |
| 最终CI工件消费          | 固定软件包、同一原数据和两侧消费     | [x] 固定CI5e76949a包34个payload逐字节核对；1192行、公共18条及研究9+9通过        |
| 发布方Python3.9环境复核 | 原CSV加载/提示/解析与保存响应        | [x] Python3.9.25/pandas1.5.2，全部1192行及main/Diamond保存结果一致；不重跑模型  |

证据见[交付回执](validation/gpqa-native-delivery-20261025.json)、[全原位置兼容审计](validation/gpqa-allrows-compatibility-audit-20261025.json)、[main审计](validation/gpqa-main-full-native-audit-20261025.json)、[extended审计](validation/gpqa-extended-full-native-audit-20261025.json)、[固定CI消费](validation/gpqa-fixed-ci-research-handoff-20261025.json)及[发布方Python版本复核](validation/gpqa-publisher-py39-verification-20261025.json)。全部约定环节已核验，GPQA来源主勾已勾；不宣称GPT-4基线、CoT或检索协议复现。

原循环固定跳过物理行索引69（从0计数，即第70条），因此还需显式兼容模式补验该原位置、复用其余保存请求，才可声称全原题生成能力核验完成：`--full --include-index69 --reuse ...`不修改题目、提示或预算。原循环分数与兼容验证结果分别保留，不因补验覆盖历史结果。

| 补充执行能力核验 | 范围                                   | 状态                                                                         |
| ---------------- | -------------------------------------- | ---------------------------------------------------------------------------- |
| 原索引69补验     | 三个split各1个原位置，其余保存请求复用 | [x] Python3.9原版本客户端补3次推理，1189个保存响应复用；全部1192位置审计通过 |

## 历史抽样记录（非本轮完成总数）

更新时间：2026-10-23。清单覆盖现有 **89 个公开候选来源、8 份企业数据**；“候选”不等于已取得、已适配或已跑通。新推荐的 17 个来源另列，尚不计入已接入范围。

### DailyTalk：原Guo完整对话批量合成

- [x] [合并交付回执](validation/dailytalk-fixed3-native-delivery-20261101.json)：原作者Guo900000权重、原universal
  HiFi-GAN、FP32
  NPU、seed1234、batch8、10轮历史及原控制值，前三段完整原对话29条合成完成；声学/vocoder各4批、29个WAV完整PCM与原图头部核验通过，历史失败和原结果保留。

- [x] 全23676行原输入/参考关系保真，实际成功CI工件79文件逐份匹配；29条公共与未修改研究消费者回读及原生输入行哈希一致。通用对话准备代码两类CI通过，真实原对话23的12条准备验证通过；既有合成不重跑。

- 已交付的是上述明确原生批量合成协议，不是全量合成或音质评分。其他原对话需原ZIP/解压音频，完整ZIP分支只有夹具验证；公共`execute`包装没有为整理代码重复生成，真实原CLI合成及公共保存结果核验分别留证。以下阶段记录的单项回执不单独代表源级交付状态。

- [x] 已定位原生失败：原Guo模型进入声学前向后，当前NPU的`DynamicGRUV2Hidden`拒绝FP32隐藏权重；尚未执行vocoder，不算合成完成。原失败、权重、精度和输入均保留。

- [x] `scripts/dailytalk_gru_npu.py`提供显式、单实例、仅推理的原GRU公式分解候选，保留原参数对象和state-dict键，不替换checkpoint、不降精度、不改变10轮历史。使用PyTorch的reset-after-hidden-affine公式，不能换成原论文的另一种GRU公式。依据：[PyTorch 2.10 GRU](https://docs.pytorch.org/docs/2.10/generated/torch.nn.GRU.html)。

- [x] 20组CPU合成夹具通过，环境Python3.12.10/PyTorch2.7.1+cpu；覆盖方向、层数、bias、布局、非零/默认零hidden和原Guo的128维双层双向、1/10步输入。输出及最终hidden均按`atol=1e-6, rtol=1e-5`对照原PyTorch
  GRU，原参数不变；这是合成单元测试，不是原任务或NPU数值结果。

- [x] 实际服务器PyTorch2.10.0+cpu/Python3.12.13隔离环境完成18组CPU数值对照，最大绝对误差`3.5762786865234375e-7`；没有初始化NPU、加载原checkpoint或执行原任务。[服务器夹具回执](validation/dailytalk-gru-cpu-equations-20261101.json)。

- [x] 确认两服务在途/排队均0、仅有原受保护引擎后，在互斥窗口完成18组NPU
  FP32数值对照，最大绝对误差`2.682209014892578e-7`；包含原Guo维度，关闭HF32，不加载原checkpoint。[NPU夹具回执](validation/dailytalk-gru-npu-equations-20261101.json)。控制进程已退出，服务身份未变，互斥租约释放；不把合成算子夹具计为原TTS任务。

- [x] 原29条输入在全新输出目录尝试原checkpoint合成，声学前向1次，已越过GRU错误；在原绘图函数遇到Matplotlib3.10.9移除`tostring_rgb`的兼容失败，vocoder尚未执行、WAV文件0。失败和旧环境保留，控制/子进程均已退出，服务身份未变。[原任务失败回执](validation/dailytalk-original-gru-compat-render-blocker-20261101.json)。这不是原任务质量失败，也不算完整合成；后续先在独立绘图环境核验兼容版本，不改原请求或模型。

使用时只对原声学模型中明确定位的Guo
`gru`实例调用`install_npu_gru_forward`，并先沿用原`model.eval()`；不修改全局`torch.nn.GRU`。正式forward拒绝CPU/CUDA、非FP32、训练及非原稠密批输入。`require_npu=False`仅供合成数值夹具，不可作为实际任务的CPU回退。原失败工作区未修改；候选已在新目录完成原29条合成，尚待公开入口和最终交接，DailyTalk来源总勾仍为空。

- [x] 独立绘图overlay固定Matplotlib3.9.4原wheel并校验SHA256，原绘图函数按原调用者前置条件验证通过；原环境不变。[绘图回执](validation/dailytalk-plotting-overlay-20261101.json)。
- [x] 原29条完整合成及独立保存结果审计通过：声学/vocoder各4批，29个WAV全部完整PCM解码、均有非零采样，29份PNG签名及尺寸核验；重核2份原模型、52份既有兼容源码和29份原Guo特征，新增模型调用0。[合成审计](validation/dailytalk-original-fixed29-synthesis-audit-20261101.json)。原失败保留，无主观音质或引擎性能结论。
- [x] 公开工作区准备和完整合成执行入口代码已补齐；服务器实测准备/复核及已有29条结果复用，未为包装重复生成。公共`execute`包装未重新实跑，与既有原流程NPU证据分开记录。
- [ ] 原协议评价边界已固定，CI工件研究消费已通过，其他原完整对话的自动准备已补，待新增代码CI闭环，来源总勾为空。

#### DailyTalk 公共 Guo 文本嵌入入口

入口为
`scripts/run_dailytalk_guo_encoder_npu.py`。这是原TTS模型必需的文本嵌入预处理，不是研究特征挖掘；它不替代声学/vocoder执行或音质评价。公开入口已在服务器实测
`plan` 和已有29份原结果的
`verify`，没有导入Torch、改动旧回执或新增模型调用。[入口核验回执](validation/dailytalk-public-encoder-entry-20261101.json)。公共
`execute` 路径本次未为包装重复运行，既有NPU生成证据与该项验证分别保留。

使用固定原模型身份配置中的distiluse发布文件，先准备原元数据及完整对话输入；环境版本见
[85项观测约束](../../configs/workload-native/dailytalk-npu-py312-observed.txt)。这是ARM64/Python3.12.13实际运行环境，不是作者锁文件或平台安装脚本；厂商Torch2.10/Torch_npu2.10和CANN须由执行平台提供，不要用普通CPU包替换。原基础Matplotlib3.10.9与合成所需的独立3.9.4绘图overlay分开配置，不合并为一次全局降级。

```bash
python scripts/run_dailytalk_guo_encoder_npu.py --action plan --prepared <外部准备目录> --metadata-root <原元数据目录> --encoder-root <固定原encoder目录> --output <外部新回执目录>
python scripts/run_dailytalk_guo_encoder_npu.py --action execute --prepared <外部准备目录> --metadata-root <原元数据目录> --encoder-root <固定原encoder目录> --output <外部新回执目录> --device npu:0 --resource-window-confirmed
python scripts/run_dailytalk_guo_encoder_npu.py --action verify --prepared <上述准备目录> --metadata-root <原元数据目录> --encoder-root <固定原encoder目录> --output <已有execution-private.json所在目录>
```

执行前先核查研究占用并设置平台允许的设备映射；`npu:0`相对于当前可见设备，不代表主机物理卡号。默认计划不读取输入或加载模型；`execute`要求显式空闲窗口确认，只允许NPU
FP32，禁止覆盖已有嵌入。`verify`仅按原文件身份及结果哈希核验，不调用模型。

入口不硬编码29条：按原输入逐句使用 `encode([原文])[0]`，校验完整对话历史、512维及原128-token
encoder策略。`--source-file`和`--split train/val`可绑定其他原行集合；原行字节、原顺序及整对话关系必须匹配固定元数据，不能提交重写的prompt。其他原集合仍需先按原协议准备；现有自动音频/对话准备配置只验证了登记的三段，不能把本入口参数化等同全量准备已通过。声学/vocoder公开入口已补，实际CI工件研究消费已通过，其他完整对话的自动准备仍按下方未完成项推进。

#### DailyTalk 公共保存结果核验

- [x] `scripts/verify_dailytalk_synthesis_outputs.py`已在服务器对已有29条原合成结果实测`plan/verify`，均退出0。重核原元数据、原请求和历史关系、2份原权重、声学/vocoder各4批、29个WAV全部PCM及29份PNG头部；旧原生回执不变，未导入Torch，新增模型调用0。[公共核验回执](validation/dailytalk-public-synthesis-audit-20261101.json)。
- [ ] 本入口只验证已保存的原合成工件，不替代完整公开执行入口、最终研究消费或音质评价；来源总勾仍为空。

```bash
python scripts/verify_dailytalk_synthesis_outputs.py --action verify --prepared <外部原输入准备目录> --metadata-root <固定原元数据目录> --models-root <原900000及universal权重目录> --stage <已有合成stage目录>
```

`--stage`应包含原`native-execution/state-private.json`和`results/900000/<对话ID>/`；按原文件名逐一匹配，不通过改名、删题或重新生成来修补失败。结果中的非零采样、格式与文件完整性不是音质分数。默认`plan`不读取数据或加载模型。

#### DailyTalk 公共原生合成链路

`scripts/prepare_dailytalk_synthesis_workspace.py`从固定原作者源码创建外部新工作区，逐份校验52份文件及原权重，复核完整对话输入和已有Guo嵌入；原源码和旧输出不覆盖。仅披露两类兼容处理：未使用的DeepSpeaker/G2P分支延迟导入，以及显式NPU设备和原seed绑定。两份YAML只改路径，原模型配置、batch8、900000步权重和10轮历史不变。

`scripts/run_dailytalk_synthesis_npu.py`调用原`batch`合成CLI，使用原FP32声学模型和universal
HiFi-GAN；显式启用已验证的单实例GRU公式分解，安全权重加载，无CPU/CUDA神经网络回退。执行要求空闲窗口确认、共享互斥文件和最多6GiB的进程分配器预算；这些条件不替代平台调度或研究侧优先检查。默认`plan`不加载模型，已有结果可通过`verify --result-stage`复用。

```bash
common=(--author-root <固定原作者源码目录> --prepared <外部输入准备目录> --metadata-root <固定原元数据目录> --models-root <原权重目录> --audio-root <原音频根目录> --encoder-output <已有Guo编码回执目录> --stage <外部新stage目录>)
python scripts/prepare_dailytalk_synthesis_workspace.py "${common[@]}" --action prepare
python scripts/prepare_dailytalk_synthesis_workspace.py "${common[@]}" --action verify
PYTHONPATH="<独立Matplotlib3.9.4-overlay目录>${PYTHONPATH:+:$PYTHONPATH}" TMPDIR=/tmp MPLBACKEND=Agg python scripts/run_dailytalk_synthesis_npu.py "${common[@]}" --action execute --device npu:0 --allocator-gib 6 --lease-file <平台共享互斥文件> --resource-window-confirmed
python scripts/run_dailytalk_synthesis_npu.py "${common[@]}" --action verify --result-stage <已有原合成stage目录>
```

以上为Bash参数数组示例，尖括号替换为实际外部路径；准备前先按前述入口取得原源码、元数据、原音频、权重和Guo嵌入，激活平台已有CANN环境及独立绘图overlay。当前服务器原安装记录为CANN9.1.0、aarch64；85项Python观测约束不是CANN安装器。保留平台的`ASCEND_HOME_PATH`、`ASCEND_OPP_PATH`、`LD_LIBRARY_PATH`及CANN
Python路径，绘图overlay只前置追加，不覆盖SDK路径；使用短实例临时目录，避免共享内存管理器的Unix套接字路径过长。不要将示例直接作为平台调度器。执行前检查研究占用和设备映射；失败保留新stage，已有结果不重新生成凑通过。`verify`会重新核验公开工作区和原资产，再读取指定已保存结果，不调用模型。

- [x] 服务器公共准备入口`plan/prepare/verify`均退出0，52文件及42份Python语法检查通过，原作者文件未改，原vocoder符号链接核验通过。
- [x] 公共合成入口`plan/verify`均退出0，完整PCM核验原29条已保存WAV，无Torch导入、无新增模型调用。[实际入口回执](validation/dailytalk-public-workspace-entry-20261101.json)。
- [ ] 公共`execute`包装未为整理代码重新运行；既有原流程NPU合成成功回执单列，不把此入口的读取/核验称作第二次真实生成。原协议评价边界已固定，CI工件研究消费已通过，其他原完整对话的自动准备已补，待新增代码CI闭环，来源不打总勾。

#### DailyTalk 实际CI安装包与剩余使用链路

- [x] 提交`70335959`的全量CI `41309439575`和CPU smoke
  `41309442204`均成功；实际安装CPU工件`13084712746`后，79份提供者文件逐份匹配。复用既有23676条索引，两个原split全部原行字节还原通过，固定29条公共/未修改研究消费者回读相等，原生输入行哈希及参考媒体对应一致；无Torch导入、无新增模型调用。[实际工件回执](validation/dailytalk-actual-ci-research-handoff-20261101.json)。
- [ ] 默认原音频/Guo输入准备保留前三段，新入口已支持其他原完整对话和全split，条件和实际验证范围见下节。新增代码CI闭环和总交付审计尚未完成，不打来源总勾。

#### DailyTalk 任意原完整对话准备

`prepare_dailytalk_original_guo_inputs.py`保留原固定前三段默认行为，新增`--dialogue-ids`选择任意原完整对话，或`--all-dialogues`选择指定原split的全部对话。`--split train/val`绑定对应原元数据，未知/重复ID、缺失历史、缺失或多余音频turn均报错，不重构缺失请求、音频或目录。原行、原顺序、原10轮历史和原映射文件不变。

```bash
# 已有发布方固定哈希覆盖的原对话23，不需要重新下载或调用模型。
python scripts/prepare_dailytalk_original_guo_inputs.py --action prepare --metadata-root <原元数据根目录> --audio-root <包含dailytalk目录的原音频根目录> --output <外部新准备目录> --split val --dialogue-ids 23
# 其他原完整对话：先从原作者入口取得原ZIP，在服务器外部数据目录原样解压并保留ZIP。
python scripts/prepare_dailytalk_original_guo_inputs.py --action prepare --metadata-root <原元数据根目录> --audio-root <原解压根目录> --output <外部新准备目录> --split val --all-dialogues --original-audio-archive <原作者ZIP路径>
```

非固定选择输出`<split>_phone.selected.private.txt`；后续编码器、工作区准备和合成入口应同时传入对应`--split`与`--source-file <上述文件>`。原音频根目录是ZIP解压后`dailytalk/`的父目录，不能指向合成结果目录。其他对话的原WAV会与给定ZIP的同名原成员逐字节比对，并检查CRC/EOF、目录turn集合及原归档大小；未得到发布方完整ZIP摘要，不把大小/成员校验称为完整归档官方SHA认证。此准备入口不下载全量归档，也不创建原WAV占位文件。

- [x] 服务器实测显式选择原对话23的`plan/prepare/verify`，全部12个原turn、真实音频SHA/CRC和原历史与既有原生验证集合的对应子集一致，无Torch/模型调用；完整train22479行与val1197行的原行还原及历史构造通过。[实际准备回执](validation/dailytalk-general-cohort-preparation-20261101.json)。首次把数据输出放入软件目录被安全检查拒绝，已改用独立外部数据目录，未放宽检查。
- [ ] 非固定原音频的完整ZIP比对分支当前只有明确标注的夹具测试，未取得和验证全量原ZIP；不称全量音频已取得、全split推理已跑完。新增入口的CI闭环及总交付审计仍待完成，来源总勾不变。

#### DailyTalk 原协议及结果含义

本来源采用原作者README的`history_guo`完整对话`batch`语音合成协议，定义见[固定协议](../../configs/workload-native/dailytalk-guo-batch-protocol.json)，不是新增一个自动音质评分benchmark。原作者说明生成某一turn可能依赖对话历史，因此固定抽样按前三段完整对话取29条，保留原音素、原顺序和10轮历史，不按单句拆题。原`evaluate.py`是训练/验证损失循环，与README的合成命令分别登记，不用格式或非零采样冒充MOS、论文评价指标或质量分数。

- [x] 原协议、原模型绑定、完整对话抽样、实际输出核验及兼容变更已明确，原合成NPU证据与公开入口源码/保存结果核验分别保留。
- [x] 实际CI安装包研究消费已完成，见上方工件回执；全量音频/合成、论文音质评价和流式时间关系未验证，不计为本次已取得能力。

#### DailyTalk 原数据公共适配与研究读取

- [x] 固定原`train_phone.txt`的22479条和`val_phone.txt`的1197条，共23676条；公共适配器逐行保留原5列及原行字节，完整重建两个原文件，SQLite索引全量建立。文本、音素、说话人和情绪作为原TTS输入；对话/turn和相邻原turn关系单列，原音频只作为参考侧定位符，不包装成聊天请求。
- [x] 固定原生验证的29条经公共`resolve_input`及未修改的研究`resolve_input`回读，完整记录相等；与既有原生输入的行哈希全部匹配，29份原参考音频存在。未启动观测/分析，索引不变，未导入Torch，新增模型调用0。[候选源码联调回执](validation/dailytalk-original-provider-research-20261101.json)。
- [x] 上述候选源码联调后，实际CI安装包的同等核验也已通过；来源总勾仍需完整准备链路收口。全量参考音频未取得，只有固定三段29份音频实证；其他定位符不表示本地已有文件。原录音和turn顺序也不表示具有流式到达、重叠或打断时间戳。

```python
from pathlib import Path
from vllm_hust_workload_data import iter_workload

records = iter_workload("DAILYTALK", data_root=Path("<已按原入口取得的外部元数据根目录>"), files=["preprocessed_data/DailyTalk/val_phone.txt"])
for record in records:
    original_tts_inputs = record.task["input"]
    original_reference = record.outcome_record()["reference_audio"]
    # 交给原生TTS入口；此处没有模型调用、改写prompt或取得参考音频。
```

### MLAgentBench：原任务环境准备

- [x] 按作者`get_task_info`两条原生分支补齐读取：1个registry别名和15个真实目录任务定义，服务器逐项核对全部16个入口及原描述字节；未修改的研究消费入口16条读回一致。[原生定义接通回执](validation/mlagentbench-native-definitions-readback-20261102.json)。16是入口数，不是16个独立数据集或论文任务数；`bibtex-generation`、`literature-review-tool`、`llama-inference`缺少发布的`eval.py`，只保留原任务定义，不标为已评分benchmark。

- [ ] 正式固定样本为`cifar10`、`imdb`、`ogbn-arxiv`，见[冻结配置](../../configs/workload-native/mlagentbench-original-directory-fixed3.json)。作者Quick
  Start使用`--task cifar10`；原默认`debug`虽然共用数据但描述不同，不能代替正式样本。执行入口新增显式`--task`选择，原默认值、步骤及预算不变。

- [x] 原作者版本 `5d71205cc20a8e95d43aa7cb7120e89ca3323e31`
  的138份软件／任务定义、670271字节已校验，79份Python静态解析通过。15个发布目录含示例，不能据此称论文有15个评测任务；原许可、Agent、准备和评价脚本保留。[源码准备回执](validation/mlagentbench-original-source-preparation-20261101.json)。

- [x] 原CIFAR10完整归档170498071字节通过官方MD5校验；50000训练图、10000测试图只在专用隔离CPU容器加载。公共资产及原任务目录准备入口均已实测。[原任务准备回执](validation/mlagentbench-cifar-original-preparation-20261101.json)。软件及数据均在服务器取得，没有在Windows保存数据或权重。

- [x] 独立CPU任务镜像使用已核验的Python3.10.22／aarch64、Torch2.3.0及TorchVision0.18.0；25份真实Agent
  SDK轮子逐个对照官方PyPI校验SHA256，离线扩展镜像构建和真实SDK／原runner导入通过。原LLM及runner文件、两处适配差异保留；没有用假SDK或GPT／Claude别名。[任务镜像回执](validation/mlagentbench-cifar-cpu-image-20261101.json)、[Agent环境回执](validation/mlagentbench-local-agent-preparation-20261101.json)。

- [x] 原五轮科学基线和原任务评分器已完成，评分0.5152，仅用于科学环境核验，不替代Agent。[基线回执](validation/mlagentbench-cifar-original-baseline-20261101.json)。

- [x] 原默认 `debug`／CIFAR10 ResearchAgent 与原评分入口完成：实际NPU模型请求13个，原工具动作4步，容器退出0；原生格式重试耗尽，未提交Final
  Answer，原评分0.3103。失败不换题、不补造答案、不放大原预算，也不称任务效果成功或性能提升。[原执行／评分审计](validation/mlagentbench-default-agent-native-audit-20261102.json)。保留原提示、工具、50步／5小时、编辑4000-token上限、fast0.01参数和原OpenAI-chat分支默认500-token／0.2；本地Qwen3-14B不是作者默认Claude配置。

- [x] 公共读取适配保留原registry别名与目录原描述字节、真实原生ID及科学环境文件关系；评价软件独立放在outcome，不拼成聊天请求，底层科学数据是否取得单独核验。

- [x] 本地构建的真实wheel在服务器安装后，81份包文件逐字节校验；全部138原软件资产校验、原任务索引及未修改研究消费入口读回通过，输入／评价引用一致，数据库未变，未运行研究观测／特征或模型。[本地wheel接通回执](validation/mlagentbench-local-wheel-research-readback-20261102.json)。这是本地源码构建包，不冒称GitHub
  CI工件。

- [x] 实际绿色CPU CI `41334800512`
  的wheel已在专用服务器安装：81份包文件逐字节核验、138原软件资产核验、16原生入口与作者API及未修改研究消费入口读回一致。见[同一回执的CI补验](validation/mlagentbench-native-definitions-readback-20261102.json)。不重跑模型，也不把读取成功当作原生任务完成。

- [ ] 补齐三个冻结正式样本的原生协议。`cifar10`、`ogbn-arxiv`执行／原评分已审计，不能解读为任务效果成功；`imdb`原题要求微调DistilBERT，隔离NPU训练环境仍未具备。原默认`debug`单独保留，不替代正式样本，不换题补勾。

- [x] 原工作区／私有Docker及持久卷已恢复。作者原生`--resume`／`--resume-step`入口接通，检查任务、实际模型、原预算、轨迹与科学快照；旧日志只读挂载，输出另存。正式CIFAR10从第3步检查点续跑，保留4步原历史；恢复后的第一个请求与原第9个请求完全一致，直接复用原响应，未重复推理／重做4步。

- [x] 正式`cifar10`原生续跑和原评分已核验：共13个真实NPU请求（原段9＋续跑新增4），另有1个准确边界缓存回放；原4步工具历史和10000×10预测文件未变。原Agent格式重试耗尽、未提交Final
  Answer，原评分0.4803，不能称效果成功。首次评分因原标签数据挂载缺失返回空评分／-1却退出0，已保留；补原数据只读挂载后原评分正常，未改评分器或重跑模型。[正式原题回执](validation/mlagentbench-formal-cifar10-native-audit-20261102.json)。其余冻结任务情况见下，不打来源总勾。

长任务启动前检查执行工作区剩余寿命，至少覆盖原任务上限和收尾窗口。平台中断不记为题目效果失败；作者原生续跑重新建立环境计时，保留各段用时，不冒称无中断运行或性能测量。

- [x] `ogbn-arxiv`完整原始归档83058288字节已在服务器取得，19个原ZIP成员CRC全部通过，未抽子图。公开入口`prepare_mlagentbench_arxiv_assets.py`的verify及已有缓存acquire均通过；摘要为本次官方源下载实测固定值，不冒称发布方提供的校验和。[准备回执](validation/mlagentbench-ogbn-arxiv-preparation-20261102.json)。图算子构建、原任务准备、Agent及评分另行核验，尚不代表原生完成。

- [x] 图采样算子官方源码编译及隔离采样检查通过，公开`prepare_mlagentbench_arxiv_offline.py`执行原`scripts/prepare.py`通过：169343节点、1166243边、40类，train/valid/test分别90941/29799/48603，原划分不变。仅把官方URL下载接到已验证离线原归档；原解包、图处理及任务软件不改。两次准备失败已记录并修复，没有模型调用。

- [x] 公开图环境配方已断网构建并完成采样检查；用其默认线程设置运行原评分器，对保留预测得到完全一致的0.2571857704评分记录，未重跑Agent／训练。默认OMP／OpenBLAS／MKL／NumExpr各2线程，避免受限容器线程耗尽。原生任务此前使用同版本已验证图环境；公开镜像本次核验范围是构建、采样及原评分，不冒称在新镜像重复完成Agent。

- [x] `ogbn-arxiv`原生Agent／原评分／原题消费关系已核验：15个真实NPU请求、4个原工具动作，48603×1预测；原评分0.2571857704，未提交Final
  Answer且格式重试耗尽。原源码／适配后软件后验、实际CI提供侧与未修改研究入口的完整输入／评价关系均通过。首次评分线程创建失败的空分／-1已保留；只限制库线程数后原评分正常，不改原评分器、不重跑模型。[原生图任务回执](validation/mlagentbench-formal-ogbn-arxiv-native-audit-20261102.json)。IMDb及公开环境终验仍待补，不打来源总勾。

- [x] 公开图环境上下文入口已实测：7份官方软件按固定摘要复用，prepare／verify一致，实际构建上下文与仓库Dockerfile摘要相同，不含数据、权重或研究代码。首次采样回执因转发stdin缺失存为空文件，已保留；修复后只重做CPU小图夹具并确认完整落盘。清理未被使用的失败准备缓存191278556字节，保留原归档、当前缓存及全部结果，可重建。

准备入口：

```bash
python scripts/prepare_jsonschemabench_author_source.py --source mlagentbench --author-root <外部原源码目录> --action prepare --max-bytes 670271
python scripts/prepare_mlagentbench_cifar_assets.py --action prepare --parts software --output <外部软件目录> --max-bytes 13995854
python scripts/prepare_mlagentbench_cifar_assets.py --action prepare --parts data --output <外部原数据目录> --max-bytes 170498071
python scripts/prepare_mlagentbench_cifar_cpu.py --action prepare --software-root <已验证软件目录> --output <新的外部构建上下文>
python scripts/prepare_mlagentbench_cifar_workspace.py --action prepare --author-root <完整原源码目录> --data-root <完整原归档目录> --output <新的外部任务目录>
python scripts/prepare_mlagentbench_local_agent.py --action prepare --author-root <完整原源码目录> --output <新的外部适配目录>
python scripts/prepare_mlagentbench_arxiv_assets.py --action acquire --data-root <外部原图归档目录> --max-bytes 134217728
python scripts/prepare_mlagentbench_graph_cpu.py --action acquire --software-root <外部图软件目录> --max-bytes 16777216
python scripts/prepare_mlagentbench_graph_cpu.py --action prepare --software-root <已验证图软件目录> --output <新的外部图构建上下文>
# 显式指定前述已验证的Python3.10/aarch64科学CPU Agent基础镜像，不修改原模型服务环境：
docker build --network none --build-arg BASE_IMAGE=<已验证Agent基础镜像> -t <独立图CPU镜像> <图构建上下文>
# 仅在具备上述已验证图依赖的专用隔离CPU容器内，使用原任务的全新副本：
python scripts/prepare_mlagentbench_arxiv_offline.py --task-root <原始ogbn-arxiv任务目录副本> --archive <已验证arxiv.zip> --execute
```

各准备入口支持显式verify；已有完整资产直接校验复用，不重复下载。下载resume只接受同一公开入口建立的归属记录及原HTTP字节范围；原数据不在宿主机解包／加载pickle。纯准备目录执行原脚本后产生的额外文件不再当作未运行准备目录校验。

执行与评分：

```bash
# 默认只显示计划；仅在下面规定的专用隔离容器中添加 --execute。
python scripts/run_mlagentbench_local_agent.py --port-root <已验证适配目录> --queue <独立私有队列> --work-root <新的任务工作目录> --log-root <新的任务日志目录> --model <NPU服务实际模型ID>
# 正式CIFAR10任务在相同参数后添加 --task cifar10；其他冻结任务同理。
# 原Agent终止后，在同样隔离的CPU容器中，从 <任务工作目录>/cifar10 运行：
PYTHONPATH=<已验证适配目录> python -m MLAgentBench.eval --log-folder <原任务日志目录> --task debug --output-file <独立评分目录>/results.json
```

执行容器必须断网、只读根、移除全部capability并启用no-new-privileges；原科学数据只读挂载，适配软件／任务输出／私有队列分别限定到本任务目录，不挂Docker套接字或凭据。所有生成程序只在容器内执行。原
`debug` 描述来自原 `tasks.json`，不能替换成目录默认 `research_problem.txt`。包装入口的早期主模块加载错误在创建Agent前退出，已用
`alter_sys=True` 保留标准 `python -m`
行为；失败容器／日志仍保留。[启动快照](validation/mlagentbench-original-agent-start-20261102.json)不当最终结果。

可信队列worker只转发原请求给已有或SSH回环转发的NPU服务，可复用
`scripts/appworld_model_transport.py`；本次跨服务器使用私有缓存HTTP作业和仅内存中转，原请求／响应不落Windows。队列服务覆盖原任务全时段，不能因科学训练暂时无模型请求提前退出。每次请求核验研究侧空闲及共享租约；观测超时检查原句柄和已存结果，不重提相同作业。评分前保留实际runner
stdout／stderr到原日志目录的 `log` 文件，不补造日志；原最终文件及原科学数据保持一致，不重跑模型凑成绩。Kaggle等其他任务访问条件单独登记，不替使用者接受规则或填写凭据。

## 打勾规则

- **一个来源一个勾**：先固定至少 3 条原始样本及其 ID，再在指定服务器使用 NPU 完成该来源适用的执行、终止及评分/完整性核验，保存脱敏验证记录后才改成 `[x]`。
- 同一来源涉及不同环境、模态或执行协议时，记录实际验证子集；一个子集跑通不代表整个来源全覆盖。
- 执行成功与模型答对分开记录。模型答错须保留分数，不伪装成运行故障，也不能隐藏后换题。运行故障、未完成评分不能勾选。

对原生任务使用原始任务及适用评分器；对会话语料/历史轨迹/企业请求，不虚构官方正确率：记录原始输入完整性、实际回放协议及响应有效性，明确“无官方任务评分”。需要闭环环境的任务不得用单次聊天替代。企业原始数据、答案和含敏感载荷的日志不公开。

“待复核”表示此前可能有历史读取记录，但**不是当前版本重新验证完成**。GSM8K
的去重审计记录原始test:0–200共201个唯一任务，另补测尾部test:1316–1318，累计204/1319；运行配置不完全相同，不合并准确率。ARC-Challenge完成原始test:0–2三条样本的读取、NPU
执行及固定版本评分；ARC-Easy另完成原test前三条评分；GSM8K另有安装包三条回查；另已完成MMLU-Pro
biology与chemistry各前三条、LoCoMo原始前六题的回查、NPU执行及适用评分；MMLU-Pro其余学科/全量及LoCoMo其他类别未跑。ARC-Easy
原test前三题已抽测，其余公开来源不预先勾选。该清单是接入进度，不是统计代表性、全量正确率或性能验收结论。

企业八份来源已各完成三条 NPU 回放与响应完整性核验，不具备独立语义 gold，不能解读成任务正确率通过。三份 v1 保留原请求体；四份 v2 与超长文本保留提供的 messages/API
字段投影，不冒充纯生产请求。metadata 的输出目标没有变成 max_tokens。既往 16K 配置失败仍保留，新 128K/非思考配置单列。

## 2026-10-23阶段统计（非当前进度）

| 范围         | 已登记 | 当前版本已抽样验证闭环 |
| ------------ | -----: | ---------------------: |
| 公开候选来源 |     92 |                     11 |
| 企业私有来源 |      8 |                      8 |

每次验证更新本文件和 [机器可读台账](dataset-validation-status.json)；记录样本
ID、数据版本、模型、执行器、评分器、终态、失败原因和脱敏证据链接。不能仅凭目录适配器或 HTTP 200 修改完成状态。

## 2026-10-23来源阶段记录快照（非当前完成清单）

以下宽表保留机器台账`sources`的旧阶段记录，不随当前交付打勾更新。当前唯一完成标记以本文件前面的118项三列表及机器台账`delivery_inventory`为准；禁止拿本快照的勾数统计当前进度。

| 完成 | 数据集 / 来源        | 对应 workload 分类 ID                                                                                                                                | 已取得数据                                       | 读取适配                                   | NPU 抽样完整执行                                              | 评分/完整性核验                                               | 当前阻塞 / 记录                                                                                                                                                                                                                                                                     |
| ---- | -------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------ | ------------------------------------------ | ------------------------------------------------------------- | ------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [x]  | AGENTBENCH           | stateful-environment-interaction<br>database-query-update<br>knowledge-graph-traversal<br>web-navigation-interaction<br>os-command-execution         | FC源码/ALFWorld原归档已核验                      | 原生任务引用/原文件一致                    | ALFWorld FC 3项正常结束                                       | 官方评分1/1/1                                                 | [接入与验证](AGENTBENCH_ALFWORLD_ORIGINAL_PREPARATION.zh-CN.md)，仅pick_and_place_simple 3/109，其他环境未验证                                                                                                                                                                      |
| [x]  | AI2_ARC              | standalone-reasoning                                                                                                                                 | Challenge/Easy SHA已核验                         | 两子集各回查 3 条                          | Challenge 3 + Easy 3；6/3548                                  | 分别2/3、1/3；不合并准确率                                    | [Challenge回执](validation/arc-challenge-npu-20261022.json)、[Easy回执](validation/arc-easy-npu-20261023.json)；各子集仅抽样，非全量                                                                                                                                                |
| [ ]  | APPWORLD             | application-api-workflow                                                                                                                             | 待复核                                           | 待复核                                     | 未跑                                                          | 未跑                                                          | 待重新核验                                                                                                                                                                                                                                                                          |
| [ ]  | AZURE_LLM            | 到达/尺寸等跨任务运行维度                                                                                                                            | 固定两文件已取得                                 | 全字段读取/建索引，每文件3条回查           | 不适用：无原prompt                                            | 不适用：非任务评分                                            | 最终工件/消费端待核；仅2023两文件28185条                                                                                                                                                                                                                                            |
| [ ]  | BBH                  | standalone-reasoning                                                                                                                                 | 服务器安装候选核验27任务及27提示原始文件         | 全6511条原字段一致；81条索引回查通过       | Boolean三题复用；余下26项78条正常stop、固定评分及离线复核完成 | 全27变体固定抽样已覆盖，非6511题全跑；最终工件/文档回归待收口 | [新增原生回执](validation/bbh-all-task-variants-original-npu-20261024.json)；来源级交付仍未勾                                                                                                                                                                                       |
| [x]  | BFCL                 | structured-generation<br>application-api-workflow<br>tool-call-generation<br>evidence-search-synthesis                                               | 官方原始归档及任务核验                           | 原JSONL和索引链路验证                      | simple_python原3题完成                                        | 官方AST valid=3/3                                             | 仅静态调用3/400，不执行工具；其他类别未验                                                                                                                                                                                                                                           |
| [ ]  | BIGCODEBENCH         | code-generation-with-tests<br>code-evaluation                                                                                                        | 服务器已核验选定原文件                           | 原0/1/2三条输入/参考往返                   | NPU生成3条stop，非功能执行                                    | 未评分：无已验证沙箱                                          | [生成记录](BIGCODEBENCH_ORIGINAL_GENERATION.zh-CN.md)，功能评分/全量/最终交接未完                                                                                                                                                                                                   |
| [ ]  | BILLSUM              | text-summarization                                                                                                                                   | 待复核                                           | 待复核                                     | 未跑                                                          | 未跑                                                          | 待重新核验                                                                                                                                                                                                                                                                          |
| [ ]  | BIXBENCH             | simulation-analysis-verification<br>scientific-coding-workflow                                                                                       | 待复核                                           | 待复核                                     | 未跑                                                          | 未跑                                                          | 待重新核验                                                                                                                                                                                                                                                                          |
| [ ]  | BURSTGPT             | 到达/尺寸等跨任务运行维度                                                                                                                            | 官方首文件已取得并传服务器                       | 当前安装前三条prepare/resolve通过          | 不适用：无原prompt                                            | 不适用：非任务评分                                            | 仅首文件抽样；其他文件/最终工件/消费端待核；不是对话原文                                                                                                                                                                                                                            |
| [ ]  | CHARTQA              | visual-document-chart                                                                                                                                | 服务器已核验选定原文件                           | 原0/1/2三条输入/参考往返                   | 阻塞：无Qwen3-VL权重，且SSH拒绝重连                           | 未评分                                                        | [服务器读取记录](PUBLIC_ASSET_SERVER_READINESS.zh-CN.md)，仅读回执；未发视觉模型请求，等待可用视觉模型与服务器连接                                                                                                                                                                  |
| [ ]  | CROSSCODEEVAL        | repository-completion                                                                                                                                | 待复核                                           | 待复核                                     | 未跑                                                          | 未跑                                                          | 待重新核验                                                                                                                                                                                                                                                                          |
| [x]  | DAILYTALK            | conversation-continuation<br>audio-streaming-interaction                                                                                             | 原23676行全读/索引通过                           | 原Guo完整对话批量合成，非在线流式          | 固定原29条NPU合成，声学/vocoder各4批                          | 实际CI79文件与研究29条回读一致                                | [原生协议回执](validation/dailytalk-fixed3-native-delivery-20261101.json)；全量原ZIP仅夹具分支、非全量音频/合成/MOS，无流式时序证明                                                                                                                                                 |
| [ ]  | DISCOVERYBENCH       | scientific-experiment-planning<br>simulation-analysis-verification                                                                                   | 待复核                                           | 待复核                                     | 未跑                                                          | 未跑                                                          | 待重新核验                                                                                                                                                                                                                                                                          |
| [x]  | DOLLY                | general-single-turn-assistant<br>document-grounded-analysis<br>text-summarization<br>text-extraction-classification<br>text-rewriting                | 完整文件SHA已核验                                | 原前三条字段回查                           | NPU生成3条stop                                                | 响应完整性，未评语义正确性                                    | [来源与接入边界](DOLLY_ORIGINAL_INSTRUCTIONS.zh-CN.md)，3/15011；未覆盖其他类别与全量                                                                                                                                                                                               |
| [ ]  | DS1000               | code-generation-with-tests<br>scientific-coding-workflow                                                                                             | 服务器已核验选定原文件                           | 原0/1/2三条输入/参考往返                   | 原补全3题，1条原预算length                                    | 未功能评分：隔离/依赖待核                                     | [原始生成记录](DS1000_ORIGINAL_GENERATION.zh-CN.md)，原预算1024与后处理顺序不变；非完整任务/全量                                                                                                                                                                                    |
| [ ]  | DUREADER             | general-single-turn-assistant<br>rag-followup<br>document-grounded-analysis                                                                          | 待复核                                           | 待复核                                     | 未跑                                                          | 未跑                                                          | 待重新核验                                                                                                                                                                                                                                                                          |
| [ ]  | FLEURS               | offline-audio-understanding<br>audio-streaming-interaction                                                                                           | 待复核                                           | 待复核                                     | 未跑                                                          | 未跑                                                          | 待重新核验                                                                                                                                                                                                                                                                          |
| [ ]  | FLORES200            | text-translation                                                                                                                                     | 待复核                                           | 待复核                                     | 未跑                                                          | 未跑                                                          | 待重新核验                                                                                                                                                                                                                                                                          |
| [ ]  | GAIA                 | evidence-search-synthesis                                                                                                                            | 待复核                                           | 待复核                                     | 未跑                                                          | 未跑                                                          | 待重新核验                                                                                                                                                                                                                                                                          |
| [ ]  | GPQA                 | standalone-reasoning                                                                                                                                 | 待复核                                           | 待复核                                     | 未跑                                                          | 未跑                                                          | 待重新核验                                                                                                                                                                                                                                                                          |
| [x]  | GSM8K                | standalone-reasoning                                                                                                                                 | SHA 已核验                                       | 安装包回查 3 条                            | 唯一题号累计204/1319                                          | 跨配置；不汇总准确率                                          | [累计覆盖回执](validation/gsm8k-cumulative-coverage-20261023.json)；[尾部补测](validation/gsm8k-tail-npu-20261023.json)三题strict/flexible均正确；非全量、非性能结果                                                                                                                |
| [ ]  | HUMANEVAL            | code-generation-with-tests                                                                                                                           | 原文件及164题校验                                | 原字段隔离及安装验证                       | 原3题生成完成，非完整任务                                     | 受阻：无可用安全沙箱                                          | 官方功能测试未执行，保持未通过                                                                                                                                                                                                                                                      |
| [ ]  | INFINITEBENCH        | document-grounded-analysis                                                                                                                           | 待复核                                           | 待复核                                     | 未跑                                                          | 未跑                                                          | 待重新核验                                                                                                                                                                                                                                                                          |
| [ ]  | JSONSCHEMABENCH      | structured-generation                                                                                                                                | 原始 Github_easy 三文件已取得                    | 候选 wheel 三条往返通过                    | 原始三个任务均 stop                                           | 原校验 1/3，保留 maxLength 失败                               | 最终 CI 工件与消费端待核；非全库/参考引擎复现                                                                                                                                                                                                                                       |
| [ ]  | LABBENCH             | standalone-reasoning<br>scientific-experiment-planning                                                                                               | 待复核                                           | 待复核                                     | 未跑                                                          | 未跑                                                          | 待重新核验                                                                                                                                                                                                                                                                          |
| [ ]  | LIBRISPEECH          | offline-audio-understanding<br>audio-streaming-interaction                                                                                           | 待复核                                           | 待复核                                     | 未跑                                                          | 未跑                                                          | 待重新核验                                                                                                                                                                                                                                                                          |
| [ ]  | LIVECODEBENCH        | code-generation-with-tests<br>code-evaluation                                                                                                        | 待复核                                           | 待复核                                     | 未跑                                                          | 未跑                                                          | 待重新核验                                                                                                                                                                                                                                                                          |
| [ ]  | LMSYS_CHAT           | general-single-turn-assistant<br>conversation-continuation                                                                                           | 待复核                                           | 待复核                                     | 未跑                                                          | 未跑                                                          | 待重新核验                                                                                                                                                                                                                                                                          |
| [x]  | LOCOMO               | conversation-memory-recall                                                                                                                           | 官方原文件及revision/SHA已核验                   | 固定CI1986题字段/3条回查及消费联调         | 原索引0–5完成，含完整原始对话上下文                           | 发布方F1：0.6667/0/0.2857/1/0.5/0.75，均值0.5337              | 仅6/1986，Qwen替代GPT-4-Turbo；恢复环境将TMPDIR置实例本地/tmp，NFS故障根因未证；第5类444题字段兼容及其余1980题未验证；保留首次HTTP500记录；CC-BY-NC-4.0                                                                                                                             |
| [ ]  | LONGALPACA           | document-grounded-analysis                                                                                                                           | 待复核                                           | 待复核                                     | 未跑                                                          | 未跑                                                          | 待重新核验                                                                                                                                                                                                                                                                          |
| [x]  | LONGBENCH            | rag-followup<br>document-grounded-analysis                                                                                                           | 官方归档及Qasper校验                             | 安装及原字段隔离验证                       | qasper原3题完成                                               | 官方QA-F1 22.52%                                              | 仅3/200；模型chat传输披露，其他子集未验                                                                                                                                                                                                                                             |
| [ ]  | LONGBENCH_V2         | document-grounded-analysis                                                                                                                           | 官方文件SHA已核验                                | 服务器同三题完整输入回查                   | 公共入口同三题复验，1条原预算耗尽                             | 官方评分0/3，错误保留                                         | [尝试与范围说明](LONGBENCH_V2_ORIGINAL_ATTEMPT.zh-CN.md)，入口已复验；原预算终态及更长范围未完成                                                                                                                                                                                    |
| [x]  | LONGMEMEVAL          | rag-followup<br>conversation-memory-recall                                                                                                           | 原资产已取得                                     | 原读者／BM25及QA入口已接                   | 原前三题各3条，合计9条已验证                                  | 原QA流程，本地9B judge明确披露                                | 实际CI工件与未修改研究消费者9条完整回读通过                                                                                                                                                                                                                                         |
| [ ]  | LONGMEMEVAL_V2       | evolving-corpus-rag<br>experience-memory-recall                                                                                                      | 待复核                                           | 待复核                                     | 未跑                                                          | 未跑                                                          | 待重新核验                                                                                                                                                                                                                                                                          |
| [ ]  | LONGVIDEOBENCH       | video-understanding                                                                                                                                  | 待复核                                           | 待复核                                     | 未跑                                                          | 未跑                                                          | 待重新核验                                                                                                                                                                                                                                                                          |
| [x]  | LONGWRITER           | general-single-turn-assistant                                                                                                                        | 完整文件SHA已核验                                | 原消息/参考隔离回查                        | NPU生成3条stop                                                | 响应完整性，未评质量/篇幅达标                                 | [LongWriter-6k说明](LONGWRITER_ORIGINAL_MESSAGES.zh-CN.md)，训练语料3/6000，不是LongBench-Write评估                                                                                                                                                                                 |
| [ ]  | MATH500              | standalone-reasoning                                                                                                                                 | OpenAI PRM800K固定版本；500行/500唯一ID及SHA已核 | 当前适配器完整读500行；答案/解答与输入隔离 | H4 Inspect固定前三题入口已提供，尚未NPU执行                   | H4官方4轮/pass_at_1/model_graded_fact待执行                   | [来源与协议及运行命令](MATH500_ORIGINAL_TASK.zh-CN.md)、[本地读取回执](validation/math500-source-read-20261023.json)；PRM800K确定性评分脚本仅作数据包诊断，不冒充H4官方分数；仓库已有200请求性能材料，run绑定评级D，单独保留，不重复做性能                                          |
| [ ]  | MBPP                 | code-generation-with-tests                                                                                                                           | 待复核                                           | 待复核                                     | 未跑                                                          | 未跑                                                          | 待重新核验                                                                                                                                                                                                                                                                          |
| [ ]  | MIND2WEB             | web-navigation-interaction                                                                                                                           | 待复核                                           | 待复核                                     | 未跑                                                          | 未跑                                                          | 待重新核验                                                                                                                                                                                                                                                                          |
| [ ]  | MLAGENTBENCH         | scientific-experiment-planning<br>scientific-coding-workflow                                                                                         | 待复核                                           | 待复核                                     | 未跑                                                          | 未跑                                                          | 待重新核验                                                                                                                                                                                                                                                                          |
| [ ]  | MLEBENCH             | scientific-coding-workflow                                                                                                                           | 待复核                                           | 待复核                                     | 未跑                                                          | 未跑                                                          | 待重新核验                                                                                                                                                                                                                                                                          |
| [x]  | MMLU_PRO             | standalone-reasoning                                                                                                                                 | SHA已核验                                        | biology组件回查3题；chemistry原文件选题3条 | 两学科各3/3完成                                               | 各3/3答对                                                     | [biology回执](validation/mmlu-pro-biology-npu-20261022.json)、[chemistry回执](validation/mmlu-pro-chemistry-npu-20261023.json)；合计6/12032，其余学科/全量未跑                                                                                                                      |
| [ ]  | MMMU                 | image-understanding<br>visual-document-chart                                                                                                         | 待复核                                           | 待复核                                     | 未跑                                                          | 未跑                                                          | 待重新核验                                                                                                                                                                                                                                                                          |
| [ ]  | MOONCAKE             | 到达/尺寸等跨任务运行维度                                                                                                                            | 固定两文件已取得                                 | 全字段读取/建索引，每文件3条回查           | 不适用：无原prompt                                            | 不适用：非任务评分                                            | 最终工件/消费端待核；仅FAST25两文件35639条                                                                                                                                                                                                                                          |
| [ ]  | MULTINEWS            | text-summarization                                                                                                                                   | 待复核                                           | 待复核                                     | 未跑                                                          | 未跑                                                          | 待重新核验                                                                                                                                                                                                                                                                          |
| [x]  | MVBENCH              | video-understanding                                                                                                                                  | 完整3895文件校验/公共获取通过                    | 统一4000题/索引与原媒体关系通过            | 固定60题原生完成                                              | 官方exact_matching，34/60                                     | 实际CI62载荷及60条提供侧/当前研究回读通过；完整4000题模型评测未跑。                                                                                                                                                                                                                 |
| [ ]  | NARRATIVEQA          | rag-followup<br>document-grounded-analysis                                                                                                           | 待复核                                           | 待复核                                     | 未跑                                                          | 未跑                                                          | 待重新核验                                                                                                                                                                                                                                                                          |
| [ ]  | NATURAL_INSTRUCTIONS | general-single-turn-assistant<br>structured-generation<br>text-translation<br>text-extraction-classification<br>text-rewriting<br>text-summarization | 固定五个NI指令任务已取得                         | 各任务安装候选前三条回查通过               | 分类102、翻译250、摘要3；新增改写3/实体抽取3                  | 原版函数评分，分任务记录                                      | 原三任务[355实例回执](validation/natural-instructions-original-npu-20261023.json)；[新增6实例回执](validation/ni-editing-extraction-original-npu-20261023.json)，新增批研究消费/CI工件未验；JSON等未测子类不升级                                                                    |
| [ ]  | NATURALQUESTIONS     | general-single-turn-assistant<br>rag-followup<br>document-grounded-analysis                                                                          | 待复核                                           | 待复核                                     | 未跑                                                          | 未跑                                                          | 待重新核验                                                                                                                                                                                                                                                                          |
| [ ]  | NATUREBENCH          | simulation-analysis-verification<br>scientific-coding-workflow                                                                                       | 待复核                                           | 待复核                                     | 未跑                                                          | 未跑                                                          | 待重新核验                                                                                                                                                                                                                                                                          |
| [ ]  | OASST1               | general-single-turn-assistant<br>conversation-continuation                                                                                           | 固定validation已取得并核SHA                      | 三个历史前缀安装包索引回查通过             | 三个续接点/两树生成均stop                                     | 无独立质量oracle；首批固定wheel研究消费通过                   | [历史续接证据](validation/oasst1-original-history-npu-20261023.json)；仅历史续接，非实时会话或全量；[固定消费回执](https://github.com/intellistream/agent-workload-suite/blob/86e664471bd191703847195033d186291caf2f17/docs/evidence/oasst-original-history-consumer-20261023.json) |
| [ ]  | OPENCODE             | code-generation-with-tests<br>code-evaluation                                                                                                        | 待复核                                           | 待复核                                     | 未跑                                                          | 未跑                                                          | 待重新核验                                                                                                                                                                                                                                                                          |
| [ ]  | OPENMATH2            | standalone-reasoning                                                                                                                                 | 待复核                                           | 待复核                                     | 未跑                                                          | 未跑                                                          | 待重新核验                                                                                                                                                                                                                                                                          |
| [ ]  | OPENTHOUGHTS3        | standalone-reasoning<br>code-generation-with-tests                                                                                                   | 待复核                                           | 待复核                                     | 未跑                                                          | 未跑                                                          | 待重新核验                                                                                                                                                                                                                                                                          |
| [ ]  | OSWORLD              | desktop-gui-interaction                                                                                                                              | 待复核                                           | 待复核                                     | 未跑                                                          | 未跑                                                          | 待重新核验                                                                                                                                                                                                                                                                          |
| [ ]  | PAPERBENCH           | scientific-experiment-planning<br>scientific-coding-workflow                                                                                         | 待复核                                           | 待复核                                     | 未跑                                                          | 未跑                                                          | 待重新核验                                                                                                                                                                                                                                                                          |
| [ ]  | PAWS                 | 待分类核对                                                                                                                                           | 待复核                                           | 待复核                                     | 未跑                                                          | 未跑                                                          | 待重新核验                                                                                                                                                                                                                                                                          |
| [ ]  | QASPER               | rag-followup<br>document-grounded-analysis                                                                                                           | 待复核                                           | 待复核                                     | 未跑                                                          | 未跑                                                          | 待重新核验                                                                                                                                                                                                                                                                          |
| [ ]  | QRECC                | rag-followup<br>evolving-corpus-rag                                                                                                                  | 固定原归档与test16451条完整核验                  | 安装CLI准备/绑定/索引3条；候选研究消费通过 | 未跑，0模型任务                                               | 未跑原生检索评分                                              | [原版准备说明](QRECC_ORIGINAL_PREPARATION.zh-CN.md)；缺原检索库/执行/评分，发布换锁待CI；会话历史不是检索证据或动态语料                                                                                                                                                             |
| [ ]  | QUALITY              | document-grounded-analysis                                                                                                                           | 原HTML-stripped dev核SHA：230题组/115文章        | 候选包2086问题读取/索引、三篇首题回查通过  | 原三题MC似然12请求完成；同题交接复验                          | 固定harness acc1/3、acc_norm2/3；非发布方基线                 | [读取回执](validation/quality-original-reader-20261023.json)、[执行回执](validation/quality-original-mc-npu-20261023.json)；CI工件/研究消费/全量未验，不是自由生成                                                                                                                  |
| [ ]  | REPOBENCH            | repository-completion                                                                                                                                | 待复核                                           | 待复核                                     | 未跑                                                          | 未跑                                                          | 待重新核验                                                                                                                                                                                                                                                                          |
| [ ]  | RULER                | 待分类核对                                                                                                                                           | 待复核                                           | 待复核                                     | 未跑                                                          | 未跑                                                          | 待重新核验                                                                                                                                                                                                                                                                          |
| [ ]  | SCHEMASTORE          | structured-generation                                                                                                                                | 待复核                                           | 待复核                                     | 未跑                                                          | 未跑                                                          | 待重新核验                                                                                                                                                                                                                                                                          |
| [x]  | SCIENCEAGENTBENCH    | scientific-experiment-planning<br>simulation-analysis-verification<br>scientific-coding-workflow                                                     | 原verified102条元数据，原前三题完整科学资产      | 102条原12字段与公共/研究端回查一致         | 原前三题单轮生成与隔离科学执行终态完成                        | 原CodeBERT NPU评分，三题程序失败保留                          | [最终回执及未跑模式](validation/scienceagentbench-fixed3-native-delivery-20261101.json)，不代表全102题执行或其他Agent分支                                                                                                                                                           |
| [x]  | SCIENCEWORLD         | scientific-experiment-planning<br>simulation-analysis-verification                                                                                   | 原资产7207条、30类已取得                         | 已适配；参考路径与输入分离                 | 固定9原生episode已跑完                                        | 保存作者评分已核验；严格新环境重放未通过                      | 实际CI51安装载荷、7207原字段/索引复用及9研究回读闭环通过；不代表全量模型执行或确定性重放                                                                                                                                                                                            |
| [ ]  | SPIDER               | structured-generation                                                                                                                                | 待复核                                           | 待复核                                     | 未跑                                                          | 未跑                                                          | 待重新核验                                                                                                                                                                                                                                                                          |
| [x]  | SPOKENWOZ            | conversation-continuation<br>audio-streaming-interaction                                                                                             | 原JSON1000对话/35564话轮已核验                   | 固定3+3完整对话，136+120话轮               | 原文本DST 68+60请求全部stop                                   | 官方/独立评分4/68、6/60，低分保留                             | 文本DST抽样协议已交付；音频与其他响应协议另列缺口，未声称全量推理。                                                                                                                                                                                                                 |
| [x]  | STRUCTEVAL           | structured-generation                                                                                                                                | 官方文件SHA已核验                                | 修正后组件回查50原题                       | Text→JSON原50条全部完成                                       | 官方语法/路径均分0.9616，3条未满分                            | [完整子集回执](validation/structeval-text-json-full-npu-20261022.json)，其他格式/转换未跑                                                                                                                                                                                           |
| [ ]  | SWE_BENCH            | repository-agent-loop                                                                                                                                | 待复核                                           | 待复核                                     | 未跑                                                          | 未跑                                                          | 待重新核验                                                                                                                                                                                                                                                                          |
| [ ]  | SWE_VERIFIED         | repository-agent-loop                                                                                                                                | 服务器已核验选定原文件                           | 原0/1/2三条输入/参考往返                   | 原生任务未跑                                                  | 未评分                                                        | [服务器读取记录](PUBLIC_ASSET_SERVER_READINESS.zh-CN.md)，非任务执行；原生环境/全量/最终交接未完                                                                                                                                                                                    |
| [ ]  | SWESMITH             | repository-agent-loop                                                                                                                                | 11份原始任务分片已取得                           | 全59,136条读取通过，原七字段保留           | 未跑原生Agent                                                 | 未跑官方评分                                                  | [准备与接通证据](validation/swesmith-original-first3-preparation-20261102.json)；amd64执行环境在ARM64上未验证，不打完成勾                                                                                                                                                           |
| [x]  | TAUBENCH             | policy-constrained-service                                                                                                                           | 官方归档SHA已核验                                | 原生任务引用同三条                         | telecom_full 3/3正常结束                                      | 官方环境reward 1、1、0                                        | [完整引用交接回执](validation/tau2-reference-handoff-npu-20261022.json)，其他domain/全量未跑                                                                                                                                                                                        |
| [ ]  | TOOLACE              | structured-generation<br>tool-call-generation                                                                                                        | 服务器已核验选定原文件                           | 原0/1/2三条输入/参考往返                   | 不适用：历史对话/元数据                                       | 不适用：无原生任务评分                                        | [服务器读取记录](PUBLIC_ASSET_SERVER_READINESS.zh-CN.md)，非任务执行；原生环境/全量/最终交接未完                                                                                                                                                                                    |
| [ ]  | TOOLBENCH            | application-api-workflow                                                                                                                             | 待复核                                           | 待复核                                     | 未跑                                                          | 未跑                                                          | 待重新核验                                                                                                                                                                                                                                                                          |
| [ ]  | TRACELAB             | interactive-code-editing<br>evolving-corpus-rag<br>repository-agent-loop                                                                             | 服务器已核验选定原文件                           | 原0/1/2三条输入/参考往返                   | 不适用：历史对话/元数据                                       | 不适用：无原生任务评分                                        | [服务器读取记录](PUBLIC_ASSET_SERVER_READINESS.zh-CN.md)，非任务执行；原生环境/全量/最终交接未完                                                                                                                                                                                    |
| [ ]  | TULU3                | general-single-turn-assistant<br>text-extraction-classification<br>text-rewriting                                                                    | 待复核                                           | 待复核                                     | 未跑                                                          | 未跑                                                          | 待重新核验                                                                                                                                                                                                                                                                          |
| [ ]  | ULTRACHAT            | general-single-turn-assistant<br>conversation-continuation                                                                                           | 待复核                                           | 待复核                                     | 未跑                                                          | 未跑                                                          | 待重新核验                                                                                                                                                                                                                                                                          |
| [ ]  | VIDEOMME             | video-understanding                                                                                                                                  | 原2700题注释校验通过，完整媒体准备中             | 字段/原视频关系已检查，公共适配待补        | 未跑，计划原始8帧及字幕两profile                              | 未跑                                                          | 学术研究限定、原媒体/原生环境/评分及公共消费待完成；不打勾。                                                                                                                                                                                                                        |
| [ ]  | VIDEOMME_V2          | video-understanding                                                                                                                                  | 待复核                                           | 待复核                                     | 未跑                                                          | 未跑                                                          | 待重新核验                                                                                                                                                                                                                                                                          |
| [ ]  | VISIONARENA          | image-understanding                                                                                                                                  | 待复核                                           | 待复核                                     | 未跑                                                          | 未跑                                                          | 待重新核验                                                                                                                                                                                                                                                                          |
| [ ]  | VISUALWEBARENA       | web-navigation-interaction                                                                                                                           | 待复核                                           | 待复核                                     | 未跑                                                          | 未跑                                                          | 待重新核验                                                                                                                                                                                                                                                                          |
| [ ]  | WAVCAPS              | offline-audio-understanding<br>audio-streaming-interaction                                                                                           | 四组件403050条及12原音频已核验                   | 原前三条/组件；模型未取得                  | 缓存准备入口12条通过；原生模型未跑                            | 未跑                                                          | 公共直连下载未通过；已有12原音频不重下。弱标注资产非原生流式任务；原生音频模型、适用评分及输出消费待完成。                                                                                                                                                                          |
| [ ]  | WEBARENA             | web-navigation-interaction                                                                                                                           | 待复核                                           | 待复核                                     | 未跑                                                          | 未跑                                                          | 待重新核验                                                                                                                                                                                                                                                                          |
| [ ]  | WILDCHAT             | general-single-turn-assistant<br>conversation-continuation                                                                                           | 待复核                                           | 待复核                                     | 未跑                                                          | 未跑                                                          | 待重新核验                                                                                                                                                                                                                                                                          |
| [ ]  | WORKARENA            | web-navigation-interaction                                                                                                                           | 待复核                                           | 待复核                                     | 未跑                                                          | 未跑                                                          | 待重新核验                                                                                                                                                                                                                                                                          |
| [ ]  | XLAM_TOOL            | structured-generation<br>tool-call-generation                                                                                                        | 待复核                                           | 待复核                                     | 未跑                                                          | 未跑                                                          | 待重新核验                                                                                                                                                                                                                                                                          |
| [ ]  | XLSUM                | text-summarization                                                                                                                                   | 待复核                                           | 待复核                                     | 未跑                                                          | 未跑                                                          | 待重新核验                                                                                                                                                                                                                                                                          |
| [ ]  | XSUM                 | text-summarization                                                                                                                                   | 待复核                                           | 待复核                                     | 未跑                                                          | 未跑                                                          | 待重新核验                                                                                                                                                                                                                                                                          |

上述历史89来源的旧状态原样保留。现新增登记以下3个既定交付来源，当前公开登记92来源；不是增加113项交付范围。下表保留候选登记时状态，最终交付状态以顶部当前表为准，不用历史勾选推算当前完成数。

| 完成 | 来源          | 分类                           | 当前核验与缺口                                                                                             |
| ---- | ------------- | ------------------------------ | ---------------------------------------------------------------------------------------------------------- |
| [ ]  | BEIR_SCIFACT  | embedding-retrieval、reranking | 原5183文档、300测试查询全量检索和原top-100重排及评分完成；公共执行入口、固定安装工件与研究接通审计未闭环。 |
| [ ]  | BEIR_NFCORPUS | embedding-retrieval、reranking | 原3633文档、323测试查询全量检索和原top-100重排及评分完成；公共执行入口、固定安装工件与研究接通审计未闭环。 |
| [ ]  | STSB          | embedding-retrieval            | 原始句对及连续标签，模型与原生评价仍待验证；不是生产检索轨迹或二分类任务。                                 |

## 企业来源：只公开接入元数据

| 完成 | 来源                       | 对应 workload 分类 ID | 数据与读取                   | NPU 抽样完整执行 | 响应/完整性核验               | 当前阻塞 / 记录                                                                                 |
| ---- | -------------------------- | --------------------- | ---------------------------- | ---------------- | ----------------------------- | ----------------------------------------------------------------------------------------------- |
| [x]  | PAIO-CHAT-1000             | 待分类核对            | 服务器 SHA 与 3 条原文已核验 | NPU 3/3 完成     | 响应/声明格式通过，语义未评估 | [128K配置记录](validation/paio-chat-1000-npu-20261022.json)，保留原始预算与既往失败             |
| [x]  | PAIO-CODE-EVAL-1000        | 待分类核对            | 服务器 SHA 与 3 条原文已核验 | NPU 3/3 完成     | 响应/声明格式通过，语义未评估 | [128K配置记录](validation/paio-code-eval-1000-npu-20261022.json)，保留原始预算与既往失败        |
| [x]  | PAIO-JSON-500              | 待分类核对            | 服务器 SHA 与 3 条原文已核验 | NPU 3/3 完成     | 响应/声明格式通过，语义未评估 | [128K配置记录](validation/paio-json-500-npu-20261022.json)，保留原始预算与既往失败              |
| [x]  | PAIO-LONG-PREFILL-5000     | 待分类核对            | 服务器 SHA 与 3 条原文已核验 | NPU 3/3 完成     | 响应/声明格式通过，语义未评估 | [128K配置记录](validation/paio-long-prefill-5000-npu-20261022.json)，保留原始预算与既往失败     |
| [x]  | PAIO-LONGTEXT-4000         | 待分类核对            | 服务器 SHA 与 3 条原文已核验 | NPU 3/3 完成     | 响应/声明格式通过，语义未评估 | [128K配置记录](validation/paio-longtext-4000-npu-20261022.json)，保留原始预算与既往失败         |
| [x]  | PAIO-PREFIX-SHARED-5000    | 待分类核对            | 服务器 SHA 与 3 条原文已核验 | NPU 3/3 完成     | 响应/声明格式通过，语义未评估 | [128K配置记录](validation/paio-prefix-shared-5000-npu-20261022.json)，保留原始预算与既往失败    |
| [x]  | PAIO-REUSE-CONV-5000       | 待分类核对            | 服务器 SHA 与 3 条原文已核验 | NPU 3/3 完成     | 响应/声明格式通过，语义未评估 | [128K配置记录](validation/paio-reuse-conv-5000-npu-20261022.json)，保留原始预算与既往失败       |
| [x]  | PAIO-SEMANTIC-SIMILAR-5000 | 待分类核对            | 服务器 SHA 与 3 条原文已核验 | NPU 3/3 完成     | 响应/声明格式通过，语义未评估 | [128K配置记录](validation/paio-semantic-similar-5000-npu-20261022.json)，保留原始预算与既往失败 |

## workload 覆盖索引（36 类历史分类，非当前40类）

下列是候选来源关系，不是已验证覆盖；需按原生子类型选择记录，不能把整个数据集自动归为某一类。

| workload               | 分类 ID                          | 候选来源                                                                                                            |
| ---------------------- | -------------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| 通用单次助手请求       | general-single-turn-assistant    | WILDCHAT、LMSYS_CHAT、DOLLY、NATURALQUESTIONS、DUREADER、NATURAL_INSTRUCTIONS、OASST1、TULU3、ULTRACHAT、LONGWRITER |
| 对话续接               | conversation-continuation        | WILDCHAT、LMSYS_CHAT、OASST1、BURSTGPT、ULTRACHAT、SPOKENWOZ、DAILYTALK                                             |
| 检索条件下的问答与跟进 | rag-followup                     | QRECC、QASPER、NARRATIVEQA、DUREADER、NATURALQUESTIONS、LONGBENCH、LONGMEMEVAL                                      |
| 文档条件下的分析       | document-grounded-analysis       | LONGBENCH、LONGBENCH_V2、INFINITEBENCH、QUALITY、QASPER、NARRATIVEQA、NATURALQUESTIONS、DUREADER、LONGALPACA、DOLLY |
| 独立推理               | standalone-reasoning             | GSM8K、MATH500、BBH、GPQA、MMLU_PRO、AI2_ARC、OPENMATH2、OPENTHOUGHTS3、LABBENCH                                    |
| 结构化生成与转换       | structured-generation            | JSONSCHEMABENCH、STRUCTEVAL、SPIDER、SCHEMASTORE、BFCL、TOOLACE、XLAM_TOOL、NATURAL_INSTRUCTIONS                    |
| 单题代码生成与测试     | code-generation-with-tests       | HUMANEVAL、MBPP、BIGCODEBENCH、DS1000、LIVECODEBENCH、OPENCODE、OPENTHOUGHTS3                                       |
| 代码评判请求           | code-evaluation                  | LIVECODEBENCH、OPENCODE、BIGCODEBENCH                                                                               |
| 仓库上下文代码补全     | repository-completion            | REPOBENCH、CROSSCODEEVAL                                                                                            |
| 交互式代码编辑助手     | interactive-code-editing         | TRACELAB                                                                                                            |
| 状态化环境操作         | stateful-environment-interaction | AGENTBENCH                                                                                                          |
| 数据库查询与更新执行   | database-query-update            | AGENTBENCH                                                                                                          |
| 知识图谱查询与遍历     | knowledge-graph-traversal        | AGENTBENCH                                                                                                          |
| 网页导航与交互         | web-navigation-interaction       | AGENTBENCH、MIND2WEB、WEBARENA、VISUALWEBARENA、WORKARENA                                                           |
| 操作系统命令执行       | os-command-execution             | AGENTBENCH                                                                                                          |
| 桌面GUI操作            | desktop-gui-interaction          | OSWORLD                                                                                                             |
| 应用API多步工作流      | application-api-workflow         | APPWORLD、BFCL、TOOLBENCH                                                                                           |
| 规则约束业务办理与排障 | policy-constrained-service       | TAUBENCH                                                                                                            |
| 工具选择与参数生成     | tool-call-generation             | BFCL、XLAM_TOOL、TOOLACE                                                                                            |
| 证据检索与综合         | evidence-search-synthesis        | GAIA、BFCL                                                                                                          |
| 科学实验规划           | scientific-experiment-planning   | DISCOVERYBENCH、PAPERBENCH、MLAGENTBENCH、SCIENCEWORLD、SCIENCEAGENTBENCH、LABBENCH                                 |
| 仿真、分析与验证流水线 | simulation-analysis-verification | SCIENCEWORLD、DISCOVERYBENCH、BIXBENCH、NATUREBENCH、SCIENCEAGENTBENCH                                              |
| 科学代码工作流         | scientific-coding-workflow       | SCIENCEAGENTBENCH、NATUREBENCH、MLEBENCH、MLAGENTBENCH、BIXBENCH、PAPERBENCH、DS1000                                |
| 演化语料检索           | evolving-corpus-rag              | QRECC、LONGMEMEVAL_V2、TRACELAB                                                                                     |
| 跨会话对话记忆         | conversation-memory-recall       | LONGMEMEVAL、LOCOMO                                                                                                 |
| 任务经历与环境轨迹记忆 | experience-memory-recall         | LONGMEMEVAL_V2                                                                                                      |
| 长程仓库 Agent         | repository-agent-loop            | SWE_BENCH、SWE_VERIFIED、TRACELAB、SWESMITH                                                                         |
| 一般图像理解           | image-understanding              | VISIONARENA、MMMU                                                                                                   |
| 视觉文档与图表理解     | visual-document-chart            | CHARTQA、MMMU                                                                                                       |
| 视频理解               | video-understanding              | VIDEOMME、VIDEOMME_V2、MVBENCH、LONGVIDEOBENCH                                                                      |
| 离线音频识别与理解     | offline-audio-understanding      | LIBRISPEECH、FLEURS、WAVCAPS                                                                                        |
| 文本摘要               | text-summarization               | XSUM、MULTINEWS、BILLSUM、XLSUM、DOLLY、NATURAL_INSTRUCTIONS                                                        |
| 文本翻译               | text-translation                 | FLORES200、NATURAL_INSTRUCTIONS                                                                                     |
| 文本抽取与分类         | text-extraction-classification   | DOLLY、NATURAL_INSTRUCTIONS、TULU3                                                                                  |
| 文本改写与编辑         | text-rewriting                   | NATURAL_INSTRUCTIONS、TULU3、DOLLY                                                                                  |
| 音频流交互             | audio-streaming-interaction      | SPOKENWOZ、DAILYTALK、LIBRISPEECH、FLEURS、WAVCAPS                                                                  |

## 新推荐，待后续补接入

17 个新增候选及适用边界见
[缺口来源目录](gap-sources-20261021.json)。97项是历史盘点口径；当前交付统一按机器清单的113项统计，另5项仅分类，不用历史数字重新计算完成率。

## 后续批次：CodeJudge-Eval 原生固定抽样交付

固定来源为[COLING2025论文](https://aclanthology.org/2025.coling-main.7.pdf)及[作者仓库](https://github.com/CodeLLM-Research/CodeJudge-Eval)，不是其他同名CodeJudge框架。沿用发布数据的完整`input`和独立`answer`，不把代码生成题改写成自制评判题；0/1-shot与easy/middle/hard配置分别保留。该任务是判断给定代码，不是生成并执行代码，不需要为验证判断请求而在宿主运行不可信代码。

论文设temperature0，报告Accuracy与Macro-F1并推荐后者。论文说明用多个正则提取选项，提取失败保留为失败；当前作者仓库只发现说明、许可和图片，未取得该提取器实现，也未核定作者max-token参数。后续若依据论文实现提取/评分，应明确为独立协议实现及声明模型/服务配置，不能冒称作者原脚本或分数复现。先取得原始数据并核对原提示与选项，再接入运行。

服务器已取得六份原JSON，每份1860条、共11160条配置记录，全部大小/LFS校验通过。首次HTTP429失败保留，第二次请求在至少10分钟退避后成功，未绕过限流。安装后的候选数据包完整核验/索引11160行，六配置各前三题共18条经公开及固定研究消费者回读一致；复用公开准备接口对六原文件只读核验通过。使用原始完整input、独立answer和原问题/候选代码/生成模型关系，不重构评判提示。六配置各前三题共18条已在现有Qwen3.5-9B
BF16 TP1
NPU上执行，均正常stop。原始请求、参考、响应、独立strict-v1提取与各配置Accuracy/Macro-F1通过零模型调用离线审计。运行明确声明8192输出预算及非思考回退配置，不冒称作者未给出的预算/模型/提取器复现；低分保留。实际CI安装工件及最终交付回执尚未完成，仍不打勾。

公开准备复用`python scripts/prepare_reused_research_assets.py CODEJUDGE_EVAL --mode plan`；获取用`--mode download --data-root <外部新目录>`，必要时显式选择同固定版本`--hf-endpoint https://hf-mirror.com`；已有资产用`--mode verify --data-root <原目录>`只读核验。该接口新增按已审计发布方profile显式选择data角色，旧来源默认raw_source不变，不重复下载已经取得的原文件。

原生入口为`python scripts/run_codejudge_original.py --data-root <原目录> --output <独立新目录> --endpoint http://127.0.0.1:19466/v1 --model <真实模型ID> --api-server-pid <PID> --npu-engine-pid <PID> --max-output-tokens <明确声明的回退预算>`；默认只展示计划，加`--execute`执行，`--full`不抽样，默认六配置各原顺序前三题。验证使用8192输出上限和显式`--disable-thinking`，不把它们标为作者默认。`--resume`只续缺失条目，保留旧状态快照，不重复已保存请求，也不自动重试失败题或换题。各配置分别按原3/6/9选项空间评分，不能混合其不同标签语义。

离线核验使用`python scripts/verify_codejudge_saved.py --data-root <原目录> --native-root <原运行目录> --output <新的外部审计JSON>`，无网络/模型依赖。后台任务使用独立进程会话及日志，避免SSH输出管道关闭中断任务；原生回答、完整请求/响应JSONL及控制台日志仅保留服务器外部目录，不进入Git。

## DeepPlanning：原资产、消费与五类原生抽样交付完成

公开准备：`python scripts/prepare_deepplanning_original_assets.py --output <仓库外新目录>`默认仅展示计划；加`--execute`取得固定作者版本，或加`--reuse-root <已核验资产目录>`复用原始文件。`--verify-only`只读核验68份原文件；服务器下载、离线核验和独立目录复用均已验证。原始任务和数据库不进入Git或本地客户端。

独立API客户端使用Python3.10，实际验证的依赖记录见`configs/workload-native/deepplanning-client-requirements.txt`。这是作者未锁版本依赖的实测解析与导入兼容配置，不冒称作者官方锁文件；初次缺soundfile/tomli/tqdm的失败保留。

购物入口为`python scripts/run_deepplanning_original_shopping.py --assets <原资产目录> --output <独立新运行目录> --endpoint <本地模型API/v1> --model <真实模型ID> --api-server-pid <PID> --npu-engine-pid <PID>`，默认只展示计划，加`--execute`执行。默认每个难度原顺序前三题，可选`--levels 1 2 3`或`--full`；使用作者rerun_ids选择原题，保留原Agent两阶段工具流程、每阶段400调用参数和购物车评分，不缩减输出预算或换题。

旅行入口`scripts/run_deepplanning_original_travel.py`使用相同基础参数并要求`--language en`或`zh`，保留原Agent、报告转换和约束评分三个阶段。报告转换的作者qwen-plus配置显式路由到已声明本地模型；这是模型回退，不是作者转换器分数复现，不启用外部付费API。英文、中文原前三题均已完整运行、转换和评分，保存记录及原逐题/聚合分数已独立重算核验；原聚合分母仍为每语言全部120题，不把抽样聚合当全量质量或样本均分。

原生旅行结束后，使用同一作者客户端运行`python scripts/verify_deepplanning_travel_saved.py --assets <原始资产目录> --native-root <原运行目录> --output <独立新审计目录>`。入口先拒绝未终结任务，再核验作者原源码、原抽样、提示/输入、工具调用关系、报告及原数据库字节，重算作者评分并与保存结果逐项对照；不调用模型，不覆盖旧评分。作者转换器未保存原始转换请求/响应，因此该审计不声称独立重放转换模型，也不是质量或性能复现。入口夹具通过不能替代真实输出核验。

候选及早期实际CI消费证据保留。当前实际CI工件12763808461（a83dfd58）已在服务器独立安装，52份包内载荷与工件一致；全部360条原记录重建/索引、五个子集各前三条公共及当前研究消费者回读通过。研究当前main与已测试consumer/pipeline/init的字节身份一致，未修改其环境或默认依赖。旅行meta_info/query_with_constraints保留为参考，不送入原生Agent输入；工具与原ID关系保留。

购物三级、英文/中文旅行共五个profile各原前三题、合计15题已执行并完成作者评分及保存结果核验。购物一级原BrokenPipeError仍保留，完整两阶段历史及原评分0/0/1在独立副本恢复，未重跑模型；其他零分、转换重试及中文首次零请求占用拒绝均保留。实际436次guarded
SDK create尝试与414条保存Agent
assistant消息分开记录，不虚构差额归因或设备成本。可用全量入口不代表已对全部360题运行模型。[完整交付证据](validation/deepplanning-original-native-delivery-20261028.json)

## 第二批：ScienceWorld 固定原生抽样交付完成

执行入口已开放全部原始task index 0–29（30份作者原提示已确认存在）。默认`--set test_mini`只用于固定首3个test
variation的可用性抽样；实际实验可使用`--set test --task-nums 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29`运行全部原test，保持作者`cut_off=False`，不把验证抽样限制成公共能力。原数据、任务、提示、动作处理和原评分不改。

原始任务资产与作者ReAct代码可统一准备在仓库外：`python scripts/prepare_scienceworld_original_assets.py --profile author-react --execute --output /external/swiftsage-source`。入口固定作者提交，核验12份原文件；`goldpaths-all.zip`保留原字节，原仓库路径`data_utils/goldpaths-all.zip`明确记录。已有原资产可同时传入`--reuse-root`与`--gold-archive`逐文件核验复用，不重复下载。该复用路径已在服务器实测通过。默认不传`--execute`仍只打印计划。

数据组件将作者公开的30类、7207个原生task/variation映射为环境任务：train3592、dev1796、test1819；保留原任务描述、划分、索引关系。作者gold
path只放参考结果，不进入模型输入，也不作为本次模型观测。用外部配置`{"sources":[{"source_id":"SCIENCEWORLD","data_root":"/external/swiftsage-source"}]}`执行`python -m vllm_hust_workload_data.pipeline run --config /external/input.json --output /external/fresh-prepared`即可建立原生引用索引，研究侧沿既有resolve接口消费。候选包先完成全7207条核验；后续实际CI包51安装载荷核验、9条固定研究回读通过，并以读取代码逐字节相同的证据复用原全量核验，不重跑模型。四个公共入口已纳入分类索引。

当前固定9个原生episode已全部结束，464次模型调用；保存作者评分依次为任务3的100/10/0、任务4的0/3/3、任务5的25/83/25，不据此宣称全量质量或性能。`verify_scienceworld_original_saved.py --saved-scores-only`可无模型、无Java核验保存身份/协议/评分一致性；默认严格CPU环境重放仍保留为额外检查，已有观察文本/freelook差异，未通过、不改写或豁免。评分核验器曾错误计入reset初始分，已按作者last_score初始化为0的原逻辑修正；原输出不变，失败过程保留。

原生运行入口为 `scripts/run_scienceworld_original.py`，客户端依赖快照为
`configs/workload-native/scienceworld-react-client-requirements.txt`（ARM64
Python3.10；不是作者完整CUDA训练环境）。从固定SwiftSage提交获取原文件放在仓库外，使用该隔离环境及Java17执行；以下目录均为使用者指定的外部路径，不把数据或模型放进仓库：

```bash
python -m pip install -r configs/workload-native/scienceworld-react-client-requirements.txt
python scripts/run_scienceworld_original.py \
  --author-source-root /external/swiftsage-source \
  --output /external/fresh-native-output \
  --endpoint http://127.0.0.1:19466/v1 --model Qwen/Qwen3.5-9B \
  --api-server-pid <existing-api-pid> --npu-engine-pid <existing-engine-pid> \
  --tokenizer-cache /external/tokenizer-cache --task-nums 3,4,5 --disable-thinking
```

`--disable-thinking` 是明确声明的Qwen非思考部署配置，不改变作者提示、采样题目、动作预算或评分。作者GPT-4 tokenizer/context
profile保留；因此是替代模型的接入验证，不是GPT-4质量复现。入口每次模型调用前核查服务无排队、NPU仅有指定引擎；不启动或停止模型。原生输出保留在外部目录，返回后仍需评分和保存记录独立复核，不能仅凭退出码打勾。当前固定9个episode已结束，保存评分与候选研究回读通过，实际CI与消费闭环已通过；原预检完成27类任务配置核验，但退出清理调用了旧版本不存在的close方法，异常保留，已确认没有遗留Java进程。

公共准备入口：`python scripts/prepare_scienceworld_original_assets.py --output /outside/repository/scienceworld`
默认只打印计划；增加 `--execute` 获取固定官方版本的15份原文件，或使用 `--verify-only`
核验已有目录。下载失败保留现场，不覆盖原目录；该入口不运行模型，不代表原生交付完成。

已验证的1.3.0环境只用于作者环境可用性检查：30类任务各variation 0完成加载、reset和step，无简化。原文件、JAR、Python客户端均放服务器，不纳入Git。回执见
`validation/scienceworld-original-native-delivery-20261027.json`。

原生Agent基线拟复用作者[SwiftSage论文分支](https://github.com/SwiftSage/SwiftSage/tree/b8ea624231a6a8d83d78c1c12bea1b44b8ea758b)，而非其已转为通用推理的main。该分支要求
`scienceworld==1.1.3`，已在服务器另建隔离环境安装，不能与1.3.0检查混算。原ReAct流程使用作者提示、test_mini原test首3个variation、100环境步数、最多200次Agent循环、每次50生成token，以及原动作处理和环境评分。作者脚本的easy配置须明确标为作者基线profile，不宣称无简化ScienceWorld或全量论文复现。原生模型执行、保存评分核验及候选研究侧消费已完成；严格新环境重放差异保留，实际CI及原生抽样交付闭环已完成；未通过的严格重放检查不改写成成功。
