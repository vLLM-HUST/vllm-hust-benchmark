# 数据集逐项验证清单

当前交付
**61/113项**，待完成52项，另5项仅分类。首批#432的60项已合入main；早期五个草稿已关闭并链接汇总，历史和分支保留。第二批#538新增ScienceWorld固定原生抽样交付，按最新main重新验证后合入。打勾不代表全量模型评测或高分保证。

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
- [x] 现有950条非渲染结果保留不重跑；当前PATH没有Docker/Podman/bwrap，完整隔离namespace探测失败（挂载传播Permission
  denied）。私有Node可复用但不代表沙箱可用；二进制存在本身也不算环境合格。
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

### InfiniteBench分协议进展（2026-10-26；来源总勾仍未完成）

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
- [ ] `longbook_sum_eng`、`longbook_choice_eng`、`longbook_qa_chn`：旧27B服务原前三题至少一题超131072上下文，最高402028输入token；新9B服务尚未逐协议验证，不截断或改选题。
- [ ] `math_calc`：原后置system轮被当前Qwen原生模板拒绝400；不改消息角色，尚未完成原生验证。

旧27B六协议18次任务与新9B两个协议6次任务分开保留，不能当作统一模型的总体质量或性能结果；旧11次length及新增3次length终态和低分不删除，不增预算重跑凑分。阶段证据统一在[原复用审计](validation/research-acquisition-reuse-audit-20261025.json)。

### StructEval分步骤核验（2026-10-26；不增加来源完成数）

- [x] 公共审计入口在服务器重评950条历史原题输出，19个非渲染协议各50条；官方分数一致，新增模型调用0次。
- [x] 完整2035条记录及索引、每条10个原字段和参考隔离核验通过。
- [x] 44种原协议组合各取原始前3条，132处公共入口与固定研究消费者回查通过；研究默认依赖未改变。
- [ ] 剩余1085条渲染/VQA协议的原生执行、视觉评分及完整来源交付。

以上使用[原阶段回执](validation/structeval-nonrender-checkpoint3-20261025.json)记录；前三项是证据复用与数据链路核验，不是整套原生benchmark完成。

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

| 来源 ID                    | 本轮勾选                     | 复用证据或待补事项                                                                                                                                                                                                                                                                                                                                  |
| -------------------------- | ---------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| WILDCHAT                   | [x] 有界原始历史续写可用     | 原首分片59857条绑定/索引；前三会话全部4处NPU续写，3 stop/1声明2048预算length保留；前三条完整字段与固定工件公共/研究回查一致。[证据](validation/wildchat-original-continuation-20261026.json)；无质量评分/在线Agent承诺，其他13分片和全量生成未验证。                                                                                                |
| LMSYS_CHAT                 | [ ] 历史资产已复用           | 已复用研究原6文件的版本/URL/校验和，支持离线计划与原文件只读校验；服务器路径、准备映射、原生执行/评分和固定消费待补。不是重新取数，不提前勾选。                                                                                                                                                                                                     |
| OASST1                     | [x] 原始会话语料可用         | 两split原资产、88838行字段核对及各三点历史续接/安装包prepare→resolve通过。[交付回执](validation/oasst-corpus-delivery-20261024.json)；无官方质量评分、非实时会话、非全量生成                                                                                                                                                                        |
| ULTRACHAT                  | [x] 原始历史续写可用         | 复用原适配/准备入口；原test_gen28304条完整字段/索引核验，前三会话全部7处NPU续写正常stop，首尾6处固定工件公共/研究回查一致；[证据](validation/ultrachat-original-continuation-20261026.json)。发布方合成历史语料，无质量评分；不是在线Agent，其他7分片/全量模型推理未验证。                                                                          |
| DOLLY                      | [x] 原请求语料可用           | 八原类别各三条共24条生成/完整性及固定版本CI、安装回核通过。[交付回执](validation/dolly-request-corpus-delivery-20261024.json)；非15011条全量，无官方语义评分                                                                                                                                                                                        |
| GSM8K                      | [x] 原生抽样交付             | 复用204/1319唯一原题执行评分；固定CI组件核对完整原资产及输入/参考隔离、保存原生结果回查。[交付回执](validation/gsm8k-native-delivery-20261024.json)；支持原生全量，未全跑、跨配置不汇总分数                                                                                                                                                         |
| MATH500                    | [x] 原生抽样交付             | H4原前三题完整4轮及评分完成；固定CI wheel全500条字段、3条往返及三项CI通过。[交付回执](validation/math500-native-delivery-20261024.json)；非全500题，不冒充独立裁判质量认证                                                                                                                                                                          |
| GPQA                       | [x] 可用                     | 原三个CSV全部1192位置均有响应及评分审计；原skip69循环分数保留，显式兼容只补3次推理、复用1189结果，原题/提示/预算不改。固定CI、作者Python3.9及研究消费通过，不称GPT-4或其他协议复现。                                                                                                                                                                |
| MMLU_PRO                   | — 仅分类                     | 复用历史回执，核对完整使用链路                                                                                                                                                                                                                                                                                                                      |
| AI2_ARC                    | [x] 原生抽样交付             | 两子集各原前三题及末三题共12题执行、原版似然评分与回查通过；固定CI组件核对六文件。[交付回执](validation/arc-native-delivery-20261024.json)；支持分别全量，未全跑、不合并分数                                                                                                                                                                        |
| LONGBENCH                  | [x] 登记Qasper交付           | 原Qasper200题原生执行、官方评分及完整输入/参考均复核；研究端固定工件三条双路径接通。不宣称其他LongBench子集或原Qasper全库。[交付回执](validation/longbench-qasper-native-delivery-20261024.json)                                                                                                                                                    |
| LONGBENCH_V2               | [ ] 原上下文容量阻塞         | 原Short三题复用；事前固定Medium原3/5/8行中两题超131072窗口，Long原0/2/7行全部超限。未截断、换题或新增推理，不计完成。[容量回执](validation/longbench-v2-length-group-capacity-20261025.json)                                                                                                                                                        |
| LONGMEMEVAL                | [ ] 评分/容量阻塞            | 三份原资产已取得；oracle/S共1000题字段关系已验，原前三题各3条真实生成及请求/响应审计完成（S完整输入113452–116259 tokens，原CoT预算800）。原judge客户端环境已验，但缺官方支持judge；M原首题历史1201716 tokens超预算，未截断/换题/发生成请求。见原生checkpoint，复用已有6条，不重跑。                                                                 |
| LONGMEMEVAL_V2             | [ ] 待开展                   | 查已有入口，缺失再补原生协议                                                                                                                                                                                                                                                                                                                        |
| QASPER                     | [x] 原版v0.3抽样原生交付完成 | 复用原5049字段/索引与图像准备；原dev0..2完成3次NPU答案/证据预测和作者原CLI评分，Answer F1约0.4996、Evidence F1=0保留。当前A9精确工件/研究dd三split9处旧索引回查通过。[证据](validation/qasper-native-delivery-20261026.json)。非LongBench200、全dev/test、LED或视觉推理，不筛图表参考，不称质量通过。                                               |
| NARRATIVEQA                | [x] 原summary-QA已完成       | 原test前三题生成/双参考评分/保存审计完成；全46765问答三表字段关系与索引、精确CI44文件及固定研究端18处回查通过。[回执](validation/narrativeqa-native-delivery-20261027.json)。不是full-story、全test推理或作者模型复现。                                                                                                                             |
| XLAM_TOOL                  | [ ] 待开展                   | 查已有入口，缺失再补原生协议                                                                                                                                                                                                                                                                                                                        |
| BIGCODEBENCH               | [ ] 隔离运行环境阻塞         | 当前实例复核：Docker/Podman不存在，可信namespace空操作因挂载传播权限失败。保留1140原任务、前三条已有生成和32作者文件；不降低作者30GiB限制、不在宿主执行生成代码，待合格隔离评分环境。                                                                                                                                                               |
| SWE_BENCH                  | [ ] 待开展                   | 查已有入口，缺失再补原生协议                                                                                                                                                                                                                                                                                                                        |
| SWE_VERIFIED               | [ ] 缺口已列待补             | 复用历史证据；原生缺口见37项复核表及方案8.5.2                                                                                                                                                                                                                                                                                                       |
| VISIONARENA                | [ ] 待开展                   | 查已有入口，缺失再补原生协议                                                                                                                                                                                                                                                                                                                        |
| MMMU                       | [x] 原validation抽样交付完成 | 全92文件/11550行/16字段与索引核验，30学科固定90题NPU推理和原评分（含4多图），152处公共/固定研究消费往返通过。[回执](validation/mmmu-native-delivery-20261027.json)。59次length保留，非完整validation质量或LLaVA复现。                                                                                                                               |
| CHARTQA                    | [x] 原测试抽样原生交付完成   | 原测试2500条全字段/图像字节/索引核验；双类别固定6题NPU生成、原评分与公共离线复核，CI产物和固定研究6条往返通过。[回执](validation/chartqa-native-delivery-20261026.json)。非全量质量/性能；修复构建器全新重编译未验证。                                                                                                                              |
| VIDEOMME                   | [ ] 待开展                   | 查已有入口，缺失再补原生协议                                                                                                                                                                                                                                                                                                                        |
| VIDEOMME_V2                | [ ] 待开展                   | 查已有入口，缺失再补原生协议                                                                                                                                                                                                                                                                                                                        |
| MVBENCH                    | [ ] 待开展                   | 查已有入口，缺失再补原生协议                                                                                                                                                                                                                                                                                                                        |
| LONGVIDEOBENCH             | [ ] 待开展                   | 查已有入口，缺失再补原生协议                                                                                                                                                                                                                                                                                                                        |
| LIBRISPEECH                | [x] 已完成                   | 原两split前三+最长共8条完整音频NPU转写和原规范WER通过；固定公开入口另跑未见原位置，5559条保真、CI38文件工件及两split公共/研究消费已核验。不是全5559推理或流式ASR。                                                                                                                                                                                  |
| FLEURS                     | [x] 原生交付可用             | 原102语言test/模型全校验，各原始前3条共306条完整波形NPU转写/CER评分和保存结果独立重评通过；最新成功CI44包3语言9处公共/研究消费一致。非全test或Conformer复现。                                                                                                                                                                                       |
| WAVCAPS                    | [ ] 待开展                   | 查已有入口，缺失再补原生协议                                                                                                                                                                                                                                                                                                                        |
| SCIENCEAGENTBENCH          | [ ] 缺口已列待补             | 复用历史证据；原生缺口见37项复核表及方案8.5.2                                                                                                                                                                                                                                                                                                       |
| DS1000                     | [x] 已交付                   | 通用原题ID／范围／full入口已接通，七库24道原题生成及作者评分完成；1000原字段公共／研究消费一致。ARM移植差异及告警保留，非全1000题成绩。见[回执](validation/ds1000-native-delivery-20261025.json)。                                                                                                                                                  |
| MIND2WEB                   | [ ] 待开展                   | 查已有入口，缺失再补原生协议                                                                                                                                                                                                                                                                                                                        |
| GAIA                       | [ ] 待开展                   | 查已有入口，缺失再补原生协议                                                                                                                                                                                                                                                                                                                        |
| STRUCTEVAL                 | [ ] 非渲染部分完成           | 已审计950条唯一原题，19个非渲染协议各全50题；339→480→621→950增量证据不重复计算。其余1085条渲染/VQA及完整来源交接待验证，不勾选整个来源。[阶段审计](validation/structeval-nonrender-checkpoint3-20261025.json)                                                                                                                                       |
| PAWS                       | [x] 已完成                   | 全8000条人工test＋15条辅助分类、保存输入/响应/评分复核；全751450条4字段/索引、精确CI44文件和固定研究消费21处回查通过。[交付回执](validation/paws-native-delivery-20261027.json)。声明式模型适配非作者聊天运行器；noisy辅助标签不作人工test。                                                                                                        |
| XLSUM                      | [x] 已完成                   | 原45变体6173个固定位置均核验原输入/编码/解码及作者ROUGE；10变体全6068、其余35各3。固定CI/研究消费与独立环境通过，不代表全114198条NPU推理；限非商业研究。                                                                                                                                                                                            |
| FLORES200                  | [x] 原生抽样协议已交付       | 原Meta归档204代码、2009公开平行位置保留；复用原NLLB-600M三语向9次NPU翻译/6个唯一平行位置，chrF++与原SPM spBLEU评分、固定CI安装及公共/研究消费通过；[交付证据](validation/flores200-native-delivery-20261026.json)。不重生成、不替换3个缺失模型语言标记、不称全部语向已执行，模型限非商业研究。                                                      |
| BURSTGPT                   | [x] 已交付                   | 三份各原20条官方模拟已核；固定CI提供者及研究消费端三份各原3行消费相等。无新模型调用；第2/3份仅授权官方缩放，不作原时序性能结论。见[回执](validation/burstgpt-native-delivery-20261025.json)。                                                                                                                                                       |
| TRACELAB                   | [ ] 缺口已列待补             | 复用历史证据；原生缺口见37项复核表及方案8.5.2                                                                                                                                                                                                                                                                                                       |
| AZURE_LLM                  | [x] 数据资产可用             | Azure 本轮脱敏回执；非模型 benchmark                                                                                                                                                                                                                                                                                                                |
| MOONCAKE                   | [x] 固定官方存储回放可用     | 两原轨迹35639条读取/索引及12实际CI研究回读；两个官方场景各首3条原生I/O通过并独立核验，测试页已精确清理。无prompt合成、无模型推理或性能结论。[证据](validation/mooncake-official-storage-delivery-20261027.json)                                                                                                                                     |
| QRECC                      | [ ] 阻塞                     | 官方原始collection规模超过当前约1.42TB可用空间；未缩减或替换                                                                                                                                                                                                                                                                                        |
| INFINITEBENCH              | [ ] 历史资产已复用           | 已复用研究原12文件的版本/URL/校验和，支持离线计划与原文件只读校验；服务器路径、准备映射、原生执行/评分和固定消费待补。不是重新取数，不提前勾选。                                                                                                                                                                                                    |
| RULER                      | [x] 可用                     | 原13种任务15组配置2530条真实请求及原评分审计完成，完整输入/预算保留；固定CI工件完成7500条全字段重建、公共回查90条及研究两路径各45条。QA2显式兼容修复已披露；不声称完整长度网格或复现发布方模型。                                                                                                                                                    |
| QUALITY                    | [x] 原NYU多选集成可用        | 原6题/24候选似然及固定harness评分、完整文章token与安装包2086题索引/6条回查通过。[交付回执](validation/quality-native-delivery-20261024.json)；非SCROLLS数据、非发布方训练基线，非全2086题执行                                                                                                                                                       |
| BBH                        | [x] 原生抽样交付             | 全27变体原任务准备、执行及固定评分可用；81个唯一原题已验证，固定CI工件安装回查及三项CI通过。[交付回执](validation/bbh-native-delivery-20261024.json)。不是6511题全量或高分保证                                                                                                                                                                      |
| BFCL                       | [x] 登记原生类别可用         | simple_python原400题生成、官方AST判定及原文/响应/安装包400条prepare→resolve审计通过。[交付回执](validation/bfcl-native-delivery-20261024.json)；静态函数/参数匹配，不代表其他BFCL类别或真实工具执行                                                                                                                                                 |
| TAUBENCH                   | [x] 登记telecom交付          | 复用ENV原三题，新跑ENV＋ACTION原三题并官方重评分一致，双角色/200步/10错误/ALL保留；干净客户端125依赖、89原源码及保存结果重评分核验通过，全部2285原生引用研究侧接通。非其他domain或全2285模型运行完成。[交付回执](validation/tau2-native-delivery-20261024.json)                                                                                     |
| TOOLBENCH                  | [ ] 待开展                   | 查已有入口，缺失再补原生协议                                                                                                                                                                                                                                                                                                                        |
| AGENTBENCH                 | [ ] 等研究窗口后续跑         | 官方FC ALFWorld前三组全41题已审计：复用0–2、新执行3–40，保留失败及原评分；剩68题待验证，不启动下一组、不冒充AgentBench整套完成。[回执](validation/agentbench-original-complete-groups-audit-20261024.json)                                                                                                                                          |
| SCHEMASTORE                | [x] 原schema资产已完成       | 原4810文件全核验，官方CLI测试976 schemas exit0；全1006 schema/3688正负例引用及索引、精确CI45文件和固定研究端16 schema/70例子字节回查通过。[回执](validation/schemastore-native-delivery-20261027.json)。非LLM请求/质量benchmark。                                                                                                                   |
| JSONSCHEMABENCH            | [x] 原生固定协议交付         | 11原配置全部核验：WashingtonPost全38，其余9具名及default各原前3，共68配置视图任务；原bench/evaluator及逐项重评通过。33原文件19100视图行/索引、99处固定提供者/研究回查通过。旧生成复用，不称全量推理、统一后端性能或模型质量达标。[交付回执](validation/jsonschemabench-official-workflow-readiness-20261024.json)                                   |
| LIVECODEBENCH              | [ ] 待开展                   | 查已有入口，缺失再补原生协议                                                                                                                                                                                                                                                                                                                        |
| HUMANEVAL                  | [x] 已交付                   | 原前三题NPU生成结果复用，公共隔离入口官方评分2/3，失败保留；164题原字段公共/研究消费相等，原3题索引消费相等。不是全164题模型成绩。见[回执](validation/humaneval-native-delivery-20261025.json)。                                                                                                                                                    |
| MBPP                       | [x] full原生抽样可用         | 原题11/12/13双NPU生成、原始三测试与setup隔离CodeEval：参考解均通过，预测2通过/1失败原样保留；974原记录完整映射/索引，14处公共/研究回查一致，精确CI安装工件已核。[证据](validation/mbpp-original-native-sample-20261026.json)；非全500题，sanitized仅取得资产。                                                                                      |
| REPOBENCH                  | [x] 固定原生抽样交付完成     | 官方默认30层各3题共90原题NPU生成/原评分及保存审计完成；49684原行与60研究回读复用，47实际CI安装载荷重新核对。固定评分哈希种子0，EM均0保留，不是质量验收。见[交付回执](validation/repobench-native-delivery-20261027.json)。                                                                                                                          |
| CROSSCODEEVAL              | [x] 原生协议已交付           | 69,494配置行全字段/索引核验复用；四语言原0..2生成/评分保留，公共入口原3..5共12条NPU生成及作者评分通过。精确成功CI44文件及12处公共/研究回查完成；非全语料或所有检索配置推理。[证据](validation/crosscodeeval-original-acquisition-20261026.json)。                                                                                                   |
| LOCOMO                     | [x] 原生抽样交付             | 原9题复用、新99题执行及原提示/官方评分复核完成，合计108条；第3类全96题，其他四类各原前三题。原数据不改，第5类显式缺键兼容非上游已合并补丁；研究端接通。不称全1986题、图像或RAG完成。[交付回执](validation/locomo-native-delivery-20261025.json)                                                                                                     |
| NATUREBENCH                | [ ] 待开展                   | 查已有入口，缺失再补原生协议                                                                                                                                                                                                                                                                                                                        |
| DISCOVERYBENCH             | [ ] 待开展                   | 查已有入口，缺失再补原生协议                                                                                                                                                                                                                                                                                                                        |
| MLAGENTBENCH               | [ ] 待开展                   | 查已有入口，缺失再补原生协议                                                                                                                                                                                                                                                                                                                        |
| MLEBENCH                   | [ ] 待开展                   | 查已有入口，缺失再补原生协议                                                                                                                                                                                                                                                                                                                        |
| WEBARENA                   | [ ] 待开展                   | 查已有入口，缺失再补原生协议                                                                                                                                                                                                                                                                                                                        |
| VISUALWEBARENA             | [ ] 待开展                   | 查已有入口，缺失再补原生协议                                                                                                                                                                                                                                                                                                                        |
| OSWORLD                    | [ ] 待开展                   | 查已有入口，缺失再补原生协议                                                                                                                                                                                                                                                                                                                        |
| XSUM                       | [x] 已完成                   | 原BART固定test抽样、官方HF ROUGE与公开入口未见原题/空正文边界验证通过；三split226711条保真、CI38文件工件和研究消费已核验。不是全11334条test推理或2018论文复现；34条原空正文保留。                                                                                                                                                                   |
| MULTINEWS                  | [x] 已完成                   | 原PEGASUS固定test抽样/原HF评分、公共入口未见原题及空正文边界通过；56216条与作者loader逐条一致，原配对/换行/分隔符保留。CI39文件工件和研究消费已核，不是全5622题test或全文长上下文验证。                                                                                                                                                             |
| BILLSUM                    | [x] 原生协议已交付           | 发布方HF原23455条三字段、两测试集冻结前三条及公共入口3/4/5的NPU推理/HF评分、固定CI39文件工件与研究消费均通过。不代表全量推理、原论文clean_text或完整原始归档。[交付回执](validation/billsum-native-delivery-20261025.json)                                                                                                                          |
| TULU3                      | [x] 原始历史续接交付         | 原6文件、全939343条字段与索引、18处固定研究回查；6分片前3条共18对话21次NPU续接，精确CPU CI工件复核通过。见原生交付回执；非官方评分/全量推理。                                                                                                                                                                                                       |
| NATURAL_INSTRUCTIONS       | [x] 登记原生任务可用         | 五文件/8284实例字段及固定CI回查通过；分类102、翻译250、抽取261、改写1178全覆盖，摘要首尾6题，合计1797唯一实例。[交付回执](validation/ni-native-delivery-20261024.json)；非整个NI，不替代底层数据集原benchmark协议                                                                                                                                   |
| OPENMATH2                  | [ ] 待开展                   | 查已有入口，缺失再补原生协议                                                                                                                                                                                                                                                                                                                        |
| OPENTHOUGHTS3              | [x] 原语料续写交付           | 固定三分片30000行/索引保真，三域各3条NPU续写及保存审计、实际CI46载荷/18研究回读通过；九条均预算耗尽无最终答案，不作质量或全120万条执行结论。[回执](validation/openthoughts3-native-delivery-20261027.json)                                                                                                                                          |
| OPENCODE                   | [ ] 待开展                   | 查已有入口，缺失再补原生协议                                                                                                                                                                                                                                                                                                                        |
| SWESMITH                   | [ ] 待开展                   | 查已有入口，缺失再补原生协议                                                                                                                                                                                                                                                                                                                        |
| WORKARENA                  | [ ] 待开展                   | 查已有入口，缺失再补原生协议                                                                                                                                                                                                                                                                                                                        |
| APPWORLD                   | [ ] 待开展                   | 查已有入口，缺失再补原生协议                                                                                                                                                                                                                                                                                                                        |
| SCIENCEWORLD               | [x] 固定原生抽样交付完成     | 原生9个episode/464次NPU调用及保存作者评分核验完成；7207原始记录全字段/索引、9实际CI研究回读、51安装载荷与CI闭环通过。入口支持全部30类及原生完整test。严格新环境文本重放差异保留，非确定性复现或质量/性能验收。[交付证据](validation/scienceworld-original-native-delivery-20261027.json)。                                                          |
| BIXBENCH                   | [ ] 待开展                   | 查已有入口，缺失再补原生协议                                                                                                                                                                                                                                                                                                                        |
| LABBENCH                   | [x] 已完成                   | 31原子集各首3条共93题原生NPU运行、原评分及保存复核通过；87 stop/6 length保留，45题被原解析器解析、14题正确。1967条索引/原图像、CI46载荷和124研究回读已核复用；不是全量推理或作者模型质量复现。                                                                                                                                                      |
| PAPERBENCH                 | [ ] 待开展                   | 查已有入口，缺失再补原生协议                                                                                                                                                                                                                                                                                                                        |
| SPOKENWOZ                  | [ ] 待开展                   | 查已有入口，缺失再补原生协议                                                                                                                                                                                                                                                                                                                        |
| DAILYTALK                  | [ ] 待开展                   | 查已有入口，缺失再补原生协议                                                                                                                                                                                                                                                                                                                        |
| LONGWRITER                 | [x] 原语料生成交付           | 6000条原字段/索引审计，复用前三条、新范围入口实跑末三条均完成响应完整性复核；研究侧双路径三条接通。无语义或长度达标结论，不是LongBench-Write。[交付回执](validation/longwriter-native-delivery-20261024.json)                                                                                                                                       |
| LONGALPACA                 | [x] 原始指令语料使用已交付   | 原12000条全部字段/索引和12处固定工件公共/研究回查，补回3231条file来源标识于非输入侧；固定原0..2与7324..7326共6题NPU执行并核验，6 stop，含2条原非空input。[证据](validation/longalpaca-native-delivery-20261026.json)。构造训练语料而非独立质量benchmark，不称全量生成、训练复现或每条均为文档分析。                                                 |
| TOOLACE                    | [x] 有界原始历史续写可用     | 全11300条原system/对话字段及索引、首尾6条固定工件公共/研究回查；原三会话全部15处NPU续写正常stop。[证据](validation/toolace-original-continuation-20261026.json)；原历史tool返回保留、不造工具/ID；无在线Agent/BFCL/ToolACE-8B复现或全量模型推理承诺。                                                                                               |
| DUREADER                   | [x] 原demo原生交付完成       | 原dev100题/453文档完整NPU生成、五种原评分分支与保存输入/响应复核；94条预测格式有效、6条格式错误保留。200题/916文档保真及12处固定工件公共/研究消费复用。[交付证据](validation/dureader-native-delivery-20261026.json)。是Qwen生成式替代配置，不是作者旧基线或完整DuReader/Zhidao；无效Depends/ROUGE-L不填0。                                         |
| SPIDER                     | [ ] 原数据库获取阻塞         | 已核验Spider及官方test-suite-sql-eval的12份原文件；两个官方GoogleDrive下载入口在当前实例网络不可达，原题/数据库尚未取得。按Test Suite Accuracy准备，不换成字符串或旧单库评分；0模型调用，不计完成。                                                                                                                                                 |
| NATURALQUESTIONS           | [x] NQ-open原任务已交付      | 原dev3610题纯文本全量NPU生成/评分，原生聊天模式另3题；91535条索引、12处固定工件公共/研究回查及干净客户端重评分通过；[交付证据](validation/nq-open-native-delivery-20261026.json)。低分与空输出保留，两profile不混分，不是完整NQ文档、RAG或质量通过。                                                                                                |
| MULTIAGENTBENCH            | [ ] 待开展                   | 查已有入口，缺失再补原生协议                                                                                                                                                                                                                                                                                                                        |
| DEEPPLANNING               | [ ] 原资产已取得             | 服务器已取得固定作者版本63份原源码/题目文件及全部5份原数据库归档（约104MB，逐份LFS SHA256核验）；公开准备入口已通过68文件只读核验及独立目录复用；独立客户端导入通过，候选包360行核验/索引及15条研究回读通过；购物一级原题运行中，其他难度/旅行、实际CI与原生评分仍待完成，0模型调用，不计完成。购物保留原购物车评分，旅行保留原报告转换及约束评分。 |
| STREAMINGBENCH             | [ ] 待开展                   | 查已有入口，缺失再补原生协议                                                                                                                                                                                                                                                                                                                        |
| OVOBENCH                   | [ ] 待开展                   | 查已有入口，缺失再补原生协议                                                                                                                                                                                                                                                                                                                        |
| FULL_DUPLEX_BENCH          | [ ] 待开展                   | 查已有入口，缺失再补原生协议                                                                                                                                                                                                                                                                                                                        |
| LIBRITTS                   | [ ] 待开展                   | 查已有入口，缺失再补原生协议                                                                                                                                                                                                                                                                                                                        |
| LJSPEECH                   | [ ] 待开展                   | 查已有入口，缺失再补原生协议                                                                                                                                                                                                                                                                                                                        |
| BEIR_SCIFACT               | [x] 已完成                   | 原test全量检索/top100重排、原评分、最终公开NPU命令及固定CI组件/研究消费均通过。范围仅为登记集合与协议，不是BM25或全部BEIR基线。                                                                                                                                                                                                                     |
| BEIR_NFCORPUS              | [x] 已完成                   | 原test全量检索/top100重排、原评分、最终公开NPU命令及固定CI组件/研究消费均通过。范围仅为登记集合与协议，不是BM25或全部BEIR基线。                                                                                                                                                                                                                     |
| MSMARCO_PASSAGE            | [ ] 原候选集合获取阻塞       | Microsoft原README/许可/评分器/脚本4文件已核验；官方静态下载入口连接重置、旧Blob入口409，公共镜像探测404。已找到IR-datasets规范校验元数据，仍缺原dev候选集合与qrels；不得造负例、改用QA或缩减候选集合。仅非商业研究。                                                                                                                                |
| STSB                       | [x] 已完成                   | 原dev1500/test1379全量NPU编码与原评分通过；8628行保真、固定CI组件、三split公共回查/研究消费及安装后缓存重算通过。不含训练复现或生产检索流量。                                                                                                                                                                                                       |
| DOCVQA                     | [x] 固定原生视觉问答可用     | 六原文件5349题全字段/原图及引用索引、36实际CI研究回读；固定首3题NPU原生模板/图像处理/ANLS及独立保存复核完成，分数0.75/1/1。非全量模型评测或作者模型质量复现。[证据](validation/docvqa-original-native-delivery-20261027.json)                                                                                                                       |
| CODEJUDGE_EVAL             | [ ] 待开展                   | 查已有入口，缺失再补原生协议                                                                                                                                                                                                                                                                                                                        |
| DRAGON_PUBLIC              | [ ] 待开展                   | 查已有入口，缺失再补原生协议                                                                                                                                                                                                                                                                                                                        |
| TEMPORALWIKI               | [ ] 待开展                   | 查已有入口，缺失再补原生协议                                                                                                                                                                                                                                                                                                                        |
| COPILOT_AGENT_2026         | [x] 原生元数据资产可用       | 七日606分片/301026原会话全量核验和索引，49实际CI载荷与候选一致，42固定实际CI研究回读及3条公共smoke完成；不造prompt、不宣称请求回放。[证据](validation/copilot-2026-metadata-delivery-20261027.json)                                                                                                                                                 |
| AZURE_LMM_2025             | [x] 服务轨迹元数据资产交付   | 全100万行四字段保真/索引、47实际CI安装载荷、首3/末3研究回读完成；无原prompt/媒体，不造请求，不称视觉benchmark。见[回执](validation/azure-lmm-2025-asset-delivery-20261027.json)。                                                                                                                                                                   |
| HLE_VERIFIED               | — 仅分类                     | 本轮不补实现、不重跑                                                                                                                                                                                                                                                                                                                                |
| SWE_BENCH_PRO              | — 仅分类                     | 本轮不补实现、不重跑                                                                                                                                                                                                                                                                                                                                |
| FRONTIERSCIENCE            | — 仅分类                     | 本轮不补实现、不重跑                                                                                                                                                                                                                                                                                                                                |
| TERMINAL_BENCH_2_1         | — 仅分类                     | 本轮不补实现、不重跑                                                                                                                                                                                                                                                                                                                                |
| PAIO-CHAT-1000             | [x] 私有原请求可用           | 复用原前三条NPU回放；固定CI读取/原响应束/原声明格式回核通过。[交付复核](validation/enterprise-fixed-ci-handoff-20261024.json)；无语义gold，正文不公开                                                                                                                                                                                               |
| PAIO-CODE-EVAL-1000        | [x] 私有原请求可用           | 复用原前三条NPU回放；固定CI读取/原响应束/原声明格式回核通过。[交付复核](validation/enterprise-fixed-ci-handoff-20261024.json)；无语义gold，正文不公开                                                                                                                                                                                               |
| PAIO-JSON-500              | [x] 私有原请求可用           | 复用原前三条NPU回放；固定CI读取/原响应束/原声明格式回核通过。[交付复核](validation/enterprise-fixed-ci-handoff-20261024.json)；无语义gold，正文不公开                                                                                                                                                                                               |
| PAIO-LONG-PREFILL-5000     | [x] 私有原请求可用           | 复用原前三条NPU回放；固定CI读取/原响应束/原声明格式回核通过。[交付复核](validation/enterprise-fixed-ci-handoff-20261024.json)；无语义gold，正文不公开                                                                                                                                                                                               |
| PAIO-LONGTEXT-4000         | [x] 私有原请求可用           | 复用原前三条NPU回放；固定CI读取/原响应束/原声明格式回核通过。[交付复核](validation/enterprise-fixed-ci-handoff-20261024.json)；无语义gold，正文不公开                                                                                                                                                                                               |
| PAIO-PREFIX-SHARED-5000    | [x] 私有原请求可用           | 复用原前三条NPU回放；固定CI读取/原响应束/原声明格式回核通过。[交付复核](validation/enterprise-fixed-ci-handoff-20261024.json)；无语义gold，正文不公开                                                                                                                                                                                               |
| PAIO-REUSE-CONV-5000       | [x] 私有原请求可用           | 复用原前三条NPU回放；固定CI读取/原响应束/原声明格式回核通过。[交付复核](validation/enterprise-fixed-ci-handoff-20261024.json)；无语义gold，正文不公开                                                                                                                                                                                               |
| PAIO-SEMANTIC-SIMILAR-5000 | [x] 私有原请求可用           | 复用原前三条NPU回放；固定CI读取/原响应束/原声明格式回核通过。[交付复核](validation/enterprise-fixed-ci-handoff-20261024.json)；无语义gold，正文不公开                                                                                                                                                                                               |

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
- [ ] SCIENCEAGENTBENCH：verified配套科学数据/参考程序/评价脚本、隔离运行环境、本地模型及辅助评分模型未就绪；发现两个上游直接入口仍读旧validation
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

数据沿用`osunlp/ScienceAgentBench@9c6e96c9e74572e979b0930ee735041cef528cb7`的原`verified`文件，102题；代码固定为`OSU-NLP-Group/ScienceAgentBench@c26e151ed601ba109dc4d35e057ff8e73fec469d`。原适配保留task_inst、domain_knowledge、dataset_folder_tree和dataset_preview；gold_program_name与eval_script_name仅作参考，其余instance/domain/源码及输出路径字段保留元数据。不要把整个envelope或参考侧直接当模型输入。

原数据URL及SHA可用`workload-data catalog download-plan SCIENCEAGENTBENCH`查看。配置`{"sources":[{"source_id":"SCIENCEAGENTBENCH","data_root":"/private/ScienceAgentBench"}]}`即可用现有pipeline全量准备/回查；数据目录须保留原`data/verified-00000-of-00001.parquet`路径。本次102条均实测回查，不仅前三题，也未修改源Parquet。

[发布方当前说明](https://github.com/OSU-NLP-Group/ScienceAgentBench/tree/c26e151ed601ba109dc4d35e057ff8e73fec469d)要求verified版使用配套`benchmark_verified.zip`，并明确不得在线重新分发解包数据。该包中科学数据、参考程序及原评价脚本尚未取得；预览文字不能替代真实数据文件，旧benchmark压缩包不能冒充verified修正版。

原生生成入口为上游run_infer.py/agent.py，评估可使用上游`evaluation.harness.run_evaluation`容器化路径，直接Conda评估为另一官方路径。当前两份直接脚本run_infer.py和run_eval.py仍硬编码`validation`
split；后续必须明确将生成和评估同时绑定原verified版本，并核验instance ID及配套评价脚本，不能静默运行不同split或回退main。原源码未修改，本次未执行任何生成程序或评分。

环境不能写成只装一个requirements就可用：上游主Agent和科学程序评估环境分开，直接路径使用Python3.10及独立sci-agent-eval环境；任务还会配置各自的科学依赖，CodeBERT和视觉judge也是评分链路的一部分。本机有Conda命令但其环境未核验，未找到Docker；既有隔离预检未证明可安全运行生成程序。本地NPU模型、辅助评分模型、原框架模型表/上下文和loopback配置尚待适配与验证；不得在全局推理环境执行上游会卸载/同步包的评估脚本，不发默认外部付费模型请求，不删指标以凑通过。原代码与依赖文件已私有保存并校验，但未安装或冒充干净环境复现。

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
| [ ]  | DAILYTALK            | conversation-continuation<br>audio-streaming-interaction                                                                                             | 待复核                                           | 待复核                                     | 未跑                                                          | 未跑                                                          | 待重新核验                                                                                                                                                                                                                                                                          |
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
| [ ]  | LONGMEMEVAL          | rag-followup<br>conversation-memory-recall                                                                                                           | 待复核                                           | 待复核                                     | 未跑                                                          | 未跑                                                          | 待重新核验                                                                                                                                                                                                                                                                          |
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
| [ ]  | MVBENCH              | video-understanding                                                                                                                                  | 待复核                                           | 待复核                                     | 未跑                                                          | 未跑                                                          | 待重新核验                                                                                                                                                                                                                                                                          |
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
| [ ]  | SCIENCEAGENTBENCH    | scientific-experiment-planning<br>simulation-analysis-verification<br>scientific-coding-workflow                                                     | 服务器已核验选定原文件                           | 原0/1/2三条输入/参考往返                   | 原生任务未跑                                                  | 未评分                                                        | [服务器读取记录](PUBLIC_ASSET_SERVER_READINESS.zh-CN.md)，非任务执行；原生环境/全量/最终交接未完                                                                                                                                                                                    |
| [x]  | SCIENCEWORLD         | scientific-experiment-planning<br>simulation-analysis-verification                                                                                   | 原资产7207条、30类已取得                         | 已适配；参考路径与输入分离                 | 固定9原生episode已跑完                                        | 保存作者评分已核验；严格新环境重放未通过                      | 实际CI51安装载荷、7207原字段/索引复用及9研究回读闭环通过；不代表全量模型执行或确定性重放                                                                                                                                                                                            |
| [ ]  | SPIDER               | structured-generation                                                                                                                                | 待复核                                           | 待复核                                     | 未跑                                                          | 未跑                                                          | 待重新核验                                                                                                                                                                                                                                                                          |
| [ ]  | SPOKENWOZ            | conversation-continuation<br>audio-streaming-interaction                                                                                             | 待复核                                           | 待复核                                     | 未跑                                                          | 未跑                                                          | 待重新核验                                                                                                                                                                                                                                                                          |
| [x]  | STRUCTEVAL           | structured-generation                                                                                                                                | 官方文件SHA已核验                                | 修正后组件回查50原题                       | Text→JSON原50条全部完成                                       | 官方语法/路径均分0.9616，3条未满分                            | [完整子集回执](validation/structeval-text-json-full-npu-20261022.json)，其他格式/转换未跑                                                                                                                                                                                           |
| [ ]  | SWE_BENCH            | repository-agent-loop                                                                                                                                | 待复核                                           | 待复核                                     | 未跑                                                          | 未跑                                                          | 待重新核验                                                                                                                                                                                                                                                                          |
| [ ]  | SWE_VERIFIED         | repository-agent-loop                                                                                                                                | 服务器已核验选定原文件                           | 原0/1/2三条输入/参考往返                   | 原生任务未跑                                                  | 未评分                                                        | [服务器读取记录](PUBLIC_ASSET_SERVER_READINESS.zh-CN.md)，非任务执行；原生环境/全量/最终交接未完                                                                                                                                                                                    |
| [ ]  | SWESMITH             | repository-agent-loop                                                                                                                                | 待复核                                           | 待复核                                     | 未跑                                                          | 未跑                                                          | 待重新核验                                                                                                                                                                                                                                                                          |
| [x]  | TAUBENCH             | policy-constrained-service                                                                                                                           | 官方归档SHA已核验                                | 原生任务引用同三条                         | telecom_full 3/3正常结束                                      | 官方环境reward 1、1、0                                        | [完整引用交接回执](validation/tau2-reference-handoff-npu-20261022.json)，其他domain/全量未跑                                                                                                                                                                                        |
| [ ]  | TOOLACE              | structured-generation<br>tool-call-generation                                                                                                        | 服务器已核验选定原文件                           | 原0/1/2三条输入/参考往返                   | 不适用：历史对话/元数据                                       | 不适用：无原生任务评分                                        | [服务器读取记录](PUBLIC_ASSET_SERVER_READINESS.zh-CN.md)，非任务执行；原生环境/全量/最终交接未完                                                                                                                                                                                    |
| [ ]  | TOOLBENCH            | application-api-workflow                                                                                                                             | 待复核                                           | 待复核                                     | 未跑                                                          | 未跑                                                          | 待重新核验                                                                                                                                                                                                                                                                          |
| [ ]  | TRACELAB             | interactive-code-editing<br>evolving-corpus-rag<br>repository-agent-loop                                                                             | 服务器已核验选定原文件                           | 原0/1/2三条输入/参考往返                   | 不适用：历史对话/元数据                                       | 不适用：无原生任务评分                                        | [服务器读取记录](PUBLIC_ASSET_SERVER_READINESS.zh-CN.md)，非任务执行；原生环境/全量/最终交接未完                                                                                                                                                                                    |
| [ ]  | TULU3                | general-single-turn-assistant<br>text-extraction-classification<br>text-rewriting                                                                    | 待复核                                           | 待复核                                     | 未跑                                                          | 未跑                                                          | 待重新核验                                                                                                                                                                                                                                                                          |
| [ ]  | ULTRACHAT            | general-single-turn-assistant<br>conversation-continuation                                                                                           | 待复核                                           | 待复核                                     | 未跑                                                          | 未跑                                                          | 待重新核验                                                                                                                                                                                                                                                                          |
| [ ]  | VIDEOMME             | video-understanding                                                                                                                                  | 待复核                                           | 待复核                                     | 未跑                                                          | 未跑                                                          | 待重新核验                                                                                                                                                                                                                                                                          |
| [ ]  | VIDEOMME_V2          | video-understanding                                                                                                                                  | 待复核                                           | 待复核                                     | 未跑                                                          | 未跑                                                          | 待重新核验                                                                                                                                                                                                                                                                          |
| [ ]  | VISIONARENA          | image-understanding                                                                                                                                  | 待复核                                           | 待复核                                     | 未跑                                                          | 未跑                                                          | 待重新核验                                                                                                                                                                                                                                                                          |
| [ ]  | VISUALWEBARENA       | web-navigation-interaction                                                                                                                           | 待复核                                           | 待复核                                     | 未跑                                                          | 未跑                                                          | 待重新核验                                                                                                                                                                                                                                                                          |
| [ ]  | WAVCAPS              | offline-audio-understanding<br>audio-streaming-interaction                                                                                           | 待复核                                           | 待复核                                     | 未跑                                                          | 未跑                                                          | 待重新核验                                                                                                                                                                                                                                                                          |
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

## 第二批：DeepPlanning 原资产、消费与原生运行（未交付）

公开准备：`python scripts/prepare_deepplanning_original_assets.py --output <仓库外新目录>`默认仅展示计划；加`--execute`取得固定作者版本，或加`--reuse-root <已核验资产目录>`复用原始文件。`--verify-only`只读核验68份原文件；服务器下载、离线核验和独立目录复用均已验证。原始任务和数据库不进入Git或本地客户端。

独立API客户端使用Python3.10，实际验证的依赖记录见`configs/workload-native/deepplanning-client-requirements.txt`。这是作者未锁版本依赖的实测解析与导入兼容配置，不冒称作者官方锁文件；初次缺soundfile/tomli/tqdm的失败保留。

购物入口为`python scripts/run_deepplanning_original_shopping.py --assets <原资产目录> --output <独立新运行目录> --endpoint <本地模型API/v1> --model <真实模型ID> --api-server-pid <PID> --npu-engine-pid <PID>`，默认只展示计划，加`--execute`执行。默认每个难度原顺序前三题，可选`--levels 1 2 3`或`--full`；使用作者rerun_ids选择原题，保留原Agent两阶段工具流程、每阶段400调用参数和购物车评分，不缩减输出预算或换题。

旅行入口`scripts/run_deepplanning_original_travel.py`使用相同基础参数并要求`--language en`或`zh`，保留原Agent、报告转换和约束评分三个阶段。报告转换的作者qwen-plus配置显式路由到已声明本地模型；这是模型回退，不是作者转换器分数复现，不启用外部付费API。当前仅完成入口夹具，原题未验证，不计完成。

安装后的候选数据包已在服务器重建全部360条原记录并建立索引，五个子集各前三题经公开入口和固定研究消费者回读，共15条逐字段一致。旅行meta_info/query_with_constraints保留为参考，不送入原生Agent输入；工具库与原ID关系保留。此证据不是实际CI工件验证，更不是模型任务闭环。购物一级固定前三题已启动，同一原生进程仍在运行；取得原评分、保存结果审计及实际CI证据前不打勾。

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
