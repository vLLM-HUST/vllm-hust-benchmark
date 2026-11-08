# 结构化输出：原始 Text→JSON 任务与企业请求

类别：`structured-generation`。采用发布方原始任务，不用自行重写的prompt替代。企业PAIO-JSON是请求数据，不称独立权威benchmark；其既有三条回放/格式核验没有独立语义gold，继续单列。

## 已验证与未验证

当前交付范围为19个完整非渲染原生协议、950题；原NPU生成、官方语法/路径评分及当前CI工件/未修改研究消费者接通均已核验，见[协议级交付回执](validation/structeval-nonrender-native-delivery-20261101.json)。原2035行全部可读取，生成入口支持显式选择原协议及全量该协议；未完成的渲染/VQA模式继续独立登记，不把非渲染完成称作整个套件完成。下列阶段记录保留原时间与范围。

2026-10-24按原2035行核查输入格式、输出格式及rendering组合：44种协议中，非渲染19种共950题、渲染25种共1085题，不能用Text→JSON一个子集代替其余协议。公开历史入口保留原默认行为及`--all-json`，新增显式`--input-type/--output-type`选择非渲染格式协议、`--start/--count`选择该协议内原始连续位置、`--all-selected`选择完整协议。选择保留原总表行号及task_id，不改query/参考或官方workflow。示例参数`--input-type CSV --output-type JSON --all-selected`运行原CSV→JSON；官方原文件、归档和模型参数仍按下文命令提供。不存在的协议或越界区间拒绝，不自动改成Text→JSON。

新增入口已完成18个其他非渲染协议各原始前三题的NPU执行与官方评分，共54条；保存的原题全部字段与固定选择一致，安装的官方Python源码与归档一致，官方评分器重算与保存分数一致，复核过程模型调用0。与历史Text→JSON全50题不重叠，合计104条唯一任务。参见[新增协议审计回执](validation/structeval-nonrender-protocols-audit-20261024.json)。渲染/VQA仍需独立隔离环境及合适模型，当前text-only服务不能据此声称支持。旧50题源码身份继续由历史回执保留，不改写成新选择功能实跑证据。

- [x] StructEval原始Text→JSON前三条：task_id
  000500/000501/000502，原文件行1085–1087。官方语法/路径评分1、1、0.68，错误保留。[回执](validation/structeval-json-original-npu-20261022.json)。
- [x] 修正后的数据包已在服务器安装，对全部50条Text→JSON原题进行准备/回查，原query一致，VQA与raw_output_metric未进入输入/注释。[组件回查](validation/structeval-provider-roundtrip-20261022.json)。此步骤模型调用0，不是额外50条推理。
- [x] 全部50条原始Text→JSON已完成NPU执行及官方评分，平均final_eval_score为0.9616，其中3条未满分。[完整子集回执](validation/structeval-text-json-full-npu-20261022.json)。它是语法/路径分数，不是语义准确率；与前三条重叠，合计唯一任务仍为50。
- [x] 其余18个非渲染协议各原始前三题，共54条，完成原生执行及官方评分复核；错误分数保留，不以满分作为链路完成条件。
- [x] CSV→JSON其余47题已完成原生执行，原题字段和官方重评分均核查一致；与此前该协议3题合计全50题。复核模型调用0，见[完整协议增量回执](validation/structeval-nonrender-complete-protocols-audit-20261024.json)。当前合计151条唯一任务；Text→JSON和CSV→JSON各自全50题，其余17个非渲染协议各3题。
- [x] CSV→XML其余47题也已完成相同原题/官方重评分审计；累计198条唯一任务，三个协议各全50题，其余16协议各3题，证据在同一完整协议增量回执中维护。
- [x] CSV→YAML、JSON→CSV、JSON→XML的剩余47题各自也已完成原题/官方重评分审计，累计339条唯一任务：6个非渲染协议各全50题、其余13协议各3题。
- [x] JSON→YAML、TOML→JSON、TOML→YAML各剩47题已完成原题和官方重评分审计；原339题不重跑，累计480条唯一原题，9个协议各全50题、其余10协议各3题。[阶段审计回执](validation/structeval-nonrender-checkpoint-20261024.json)。
- [x] Text→CSV、Text→TOML、Text→XML各剩47题完成原题及官方重评分审计，新增141题，累计621条唯一原题。12个非渲染协议各全50题，其余7协议各3题。[第二阶段审计](validation/structeval-nonrender-checkpoint2-20261024.json)。旧回执不改写，机器清单按339→480→621证据链对齐，不重复模型执行。
- [x] Text→YAML、XML→CSV/JSON/YAML、YAML→CSV/JSON/XML各剩47题均完成原题及官方重评分审计，新增329题，累计950条唯一原题，19个非渲染协议各全50题。[第三阶段审计](validation/structeval-nonrender-checkpoint3-20261025.json)。复核模型调用0，原339→480→621证据不改写、不重跑。
- [x] 19个完整非渲染协议交付闭环：原950题结果复用，111份保存文件哈希一致，公开原数据/源码获取与独立客户端环境已补齐；当前固定CI工件与未修改研究代码完成2035行字段核对及132次完整原记录回查，0新模型调用。
- [ ] 25个渲染/VQA协议整套覆盖未完成，已完成部分及原生失败见唯一清单；不能以950条非渲染完成声称整个2035题套件完成。
- [ ] PAIO-JSON的语义准确率：未提供独立gold；保持不评估，不因合法JSON或schema通过而打勾。

## 原始数据和官方实现

固定HF数据 `TIGER-Lab/StructEval` revision `936be10548688c6d417cd435dd906158f19ba14b`，原train
parquet完整SHA校验。该文件在数据卡中对应test split，不依据文件名把它当训练集重划。

官方仓库作者推荐的LiteLLM分支固定提交
`7781339a5b033965ac664133543ebd10773643dd`，完整源码归档SHA与安装Python源码一致。官方`build_prompt(query)`加原有代码标记，官方workflow执行生成、语法解析和路径评分；不是自行追加新提示/答案或自制评分器。

使用现有Qwen3.5-27B、双卡910B2、BF16/graph、本地NPU服务，text-only/nonthinking配置。原官方workflow的temperature=0、max_tokens=None，concurrency=1、render_concurrency=1，关闭重试并设置请求timeout300秒。对所选非渲染格式调用官方语法/路径检查，不调用VLM或外部judge；这一分数不等同通用语义正确率。

## 输入与评分隔离修正

原目录把`VQA`参考问答置于annotation_fields，而输入envelope包含replay.annotations，边界不严谨。本批移到outcome_fields，与raw_output_metric一起仅供评分侧。原始文件不修改，task
query不改；当前官方模型运行只读取query，因此既有Text→JSON得分不是因这次修正而改变。回归测试检查整个input_record，而非只检查task.input，避免注释旁路带入答案。

## 运行

### 获取原数据与官方源码，建立独立客户端

在服务器的外部目录准备原文件，不放进仓库。公开获取入口复用现有固定哈希下载器，数据固定2035行Parquet，官方LiteLLM分支固定完整源码ZIP；文件名中的train不改变发布方test
split定义。已有文件先用只读验证，不重复下载。原数据与源码分别遵循发布方数据卡和固定源码中的许可证。

```sh
python scripts/prepare_structeval_original_assets.py --output /private/structeval-assets
# 已有该入口创建的匹配目录时可加 --resume；只读核验用 --verify-only。
# 指定 --parts data 或 --parts scoring 可只准备缺少的一部分。

python3.12 -m venv /private/structeval-client
env -u PYTHONPATH /private/structeval-client/bin/python -m pip install \
  -r configs/workload-native/structeval-client-requirements.txt \
  '/private/structeval-assets/scoring/structeval-litellm-official.zip[render]'
env -u PYTHONPATH /private/structeval-client/bin/python -m pip check

# 完整原Text→JSON协议50题；输出目录必须是新的私有目录。
env -u PYTHONPATH TORCH_DEVICE_BACKEND_AUTOLOAD=0 \
  /private/structeval-client/bin/python scripts/run_structeval_original_json.py \
  --data-file /private/structeval-assets/data/data/train-00000-of-00001.parquet \
  --official-archive /private/structeval-assets/scoring/structeval-litellm-official.zip \
  --output /private/structeval-text-json-full \
  --endpoint http://127.0.0.1:19466/v1 \
  --model workload-qwen35-27b-128k-nonthinking --disable-thinking --all-json
```

示例模型ID对应历史双NPU27B服务；运行前须与实际本地NPU服务的`/v1/models`一致，不自动把它映射为9B或其他模型。模型服务与该CPU
HTTP客户端是两个独立环境，不向现有推理环境安装这些依赖。客户端实测Python3.12.13，直接依赖版本见requirements文件；它不是全量传递依赖锁。已有执行环境在去除继承的CANN
`PYTHONPATH`后`pip check`退出0；保留该变量时会混入未配齐的CANN工具包依赖。这只调整客户端子进程环境，不修改NPU服务、算子或模型参数。非渲染协议虽然安装原render
extra以满足上游模块导入，但不执行浏览器、生成代码或VQA；渲染模式仍必须走独立隔离执行器。

模型结果应按明确的模型工件、服务配置及输出profile分别报告。历史27B的0.9616是其原50题语法/路径分数，不是其他模型或重新配置后可直接引用的成绩。真实获取、环境、原生执行和研究读取的证据分别登记在唯一清单中，不能仅凭下载或安装成功打勾。

### 输出配置与历史结果复用

新增经确认的`--output-profile qwen35-9b-nonthinking-32768`：必须同时指定`--model Qwen/Qwen3.5-9B --disable-thinking`，显式将workflow输出上限绑定为32768。默认`publisher-null`仍保持原`max_tokens=None`。回执同时记录原上限、实际上限、profile及是否偏离发布方预算；两种配置单独报告，不能合并成绩或把新配置称为官方默认。新配置仅用于失败题及未跑题，成功生成保留复用；已取消且结果未知的请求需先核查，不自动重放。该配置已完成软件测试，真实任务结果另行登记，不因加入配置补勾。

传输参数与生成预算分别记录：`--request-timeout 300`控制单次HTTP超时，`--max-retries 0`控制官方客户端外层重试；入口显式关闭SDK内部重试，避免一次逻辑请求被隐式发送三次。这些参数不改变原题、`max_tokens=None`或评分规则。历史运行曾发生内部额外重试，原始失败记录保留，不改写为新配置实跑结果。取消且未收到响应的请求记录为结果未知，不能计作零分或未经核查自动重放。

渲染任务的最新逐协议状态以[唯一打勾清单](DATASET_CHECKLIST.zh-CN.md)及[机器清单](dataset-validation-status.json)为准。Text→LaTeX全50题已完成审计；后续Text→TikZ传输失败，保留34份有效生成及一个结果未知请求，尚未完成该协议评分。已完成的生成不能因批次失败而全部重跑。

```sh
python scripts/run_structeval_original_json.py \
  --data-file /authorized/StructEval/data/train-00000-of-00001.parquet \
  --official-archive /authorized/structeval-official.zip \
  --output /private/fresh-three \
  --endpoint http://127.0.0.1:19466/v1 --model workload-qwen35-27b-128k-nonthinking
```

添加`--all-json`执行全部50条原始Text→JSON，不包括格式转换到JSON。程序强制新建私有输出目录，校验原始数据/源码，模型传输仅许可批准的本地端点。原始选择及推理输出包含题目、评分参考与模型回答，不发布Git/CI。

任务环境独立安装官方源码及render extra（上游模块导入需要，JSON子集不会启动浏览器），固定LiteLLM1.65.0、OpenAI
SDK2.54.0。库修复后的数据wheel与原始官方执行器是两个组件；研究侧只消费数据组件，不迁入观测、性能剖析或研究分析。CI只验证临时软件夹具和CLI边界，不取真实数据或模型。
