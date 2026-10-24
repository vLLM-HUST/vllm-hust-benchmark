# AgentBench ALFWorld FC 原始任务接入与验证

原new_std索引0–108共109个任务现已全部完成原版任务/AgentRL
Session、NPU模型交互及官方评分。复用已审计41题，仅新执行其余68题；核验1625次模型交互及保存结果，官方评分函数复算为46/109。此处“完成”指原生执行链路和结果审计完成，不要求模型答对每题，也不代表其他AgentBench环境或官方Docker等价性。研究侧全部109项原生引用接通另见下方勾选，不能当作研究观测验收。不是QBI派生任务、静态回放或独立ALFWorld结果冒充AgentBench。[全量原生审计回执](validation/agentbench-alfworld-all109-native-20261024.json)。

- [x] 固定官方AgentBench提交`d1e4a10db08c87075c78972e48ecc182be03e2d5`及完整源码归档SHA。
- [x] 按该版本官方Dockerfile所指的ALFWorld源码和0.2.2发布游戏归档下载、校验并在服务器解包，不使用旧缓存中不存在的预选任务替代题。
- [x] 原`new_std`索引共109项，按原加载顺序预选0、1、2；对应游戏文件均存在并与官方来源归档字节一致。[准备回执](validation/agentbench-alfworld-original-preparation-20261022.json)。
- [x] 三条官方原生任务、AgentRL
  Session交互及原生终止/评分，分别4/7/20步、合计31次模型请求。[执行回执](validation/agentbench-alfworld-native-npu-20261022.json)。
- [x] 公开构建脚本、固定依赖的干净安装及pip
  check；最终标准OpenCV环境同三题复验仍为1/1/1。[交接复验](validation/agentbench-alfworld-public-handoff-npu-20261023.json)。两次运行合计仍为三个唯一任务，不计六题。
- [x] 此版本ALFWorld全部109项原生执行和官方结果复算。
- [x] 固定数据包与研究消费包实际安装后，通过研究侧prepare和verify两条API重验全部109项原生索引及配套原游戏资产引用，与原执行引用完全一致；没有新跑游戏或调用模型。[研究交接回执](validation/agentbench-alfworld-research-handoff-20261024.json)。这是数据和原生关系接通，不代表研究观测runtime已接入或验收。
- [ ] AgentBench的其他环境。

初始准备阶段（后续执行进展见文末）：2026-10-24使用固定CI数据组件，在服务器复核并准备了全部109个官方任务引用及其原始归档资产。原索引有六组：pick_and_place
9题、pick_clean_then_place 5题、pick_heat_then_place 27题、pick_cool_then_place 29题、look_at_obj
18题、pick_two_obj
21题。已执行的0–2仅覆盖第一组，其他106题仍未执行；本次引用准备的模型调用和新原生执行均为0，不计完成。各组保持官方连续索引，后续复用旧三题而非重跑。参见[全索引准备回执](validation/agentbench-original-groups-preparation-20261024.json)。

当前官方版本属于Function
Calling版，不能与历史v0.2、其他论文协议或QBI裁剪结果混为同一来源版本。任务类为`stateful-environment-interaction`；DB、KG、OS、Web任务不能因为同属AgentBench而一并勾选。

准备过程中保留了服务器直连GitHub超时、CMake 4不兼容、较新spaCy依赖不支持Python 3.9及依赖回溯记录。当前在任务专用Python
3.9环境内固定兼容构建/运行依赖，并排除继承的CANN
Python路径；未更改任务、提示、工具动作规则、步数或评分。此环境不是已验证的官方Docker镜像等价物。模型服务和持续验证队列不受此次依赖准备影响。

进一步原生构建已确认Fast
Downward可编译，原TextWorld安装脚本曾因缺少aarch64的Inform7编译器失败。现在以PDDL专用构建跳过未使用的Inform7/Glulx可执行文件安装钩子，保留原Python代码和CFFI扩展；安装后76个TextWorld、74个ALFWorld、37个AgentRL
Python文件逐字节匹配固定原始归档。首轮采用headless OpenCV；最终公开recipe固定标准OpenCV
4.11.0.86和官方Docker所需libGL运行库，明确拒绝两个OpenCV发行包覆盖同一命名空间。两种环境的同题结果分别保留，使用protobuf兼容codec；不覆盖Inform7编译、Glulx游戏或视觉THOR，不声称官方Docker镜像等价。

后续必须使用固定官方任务类及其原生交互/评分，模型调用仅路由到批准的本地NPU服务。环境状态反馈是完成原生任务所必需的交互，不是迁移科研观测钩子；任务日志和原始载荷只保留私有目录。准备成功或软件测试均不能替代真实任务执行证据。

## 可复用链路

数据准备使用Python>=3.10的根包和独立workload-data包；原生环境单独使用Python 3.9.25。不能为了将二者塞入同一环境而降低数据包声明要求。旧0.1.0
wheel不含新增AgentBench引用API，需固定本PR的提交和wheel SHA，不能只看版本号。

官方AgentBench和三份游戏归档的下载URL、SHA见`workload-data catalog download-plan AGENTBENCH`。另外准备以下原始runtime源码归档及worker
wheel；脚本内固定完整SHA，拒绝替换来源：

- ALFWorld：`https://api.github.com/repos/alfworld/alfworld/zipball/a5d3476e65e3effc2ac9b7f2493979ba6794e2a6`
- TextWorld：`https://api.github.com/repos/MarcCote/TextWorld/zipball/634f9f91fec732a79dd9e7623675301a53f06623`
- Fast
  Downward：`https://api.github.com/repos/MarcCote/downward/zipball/84769171b9d965bf5739eaa7cf6604b0d9697534`
- 从PyPI下载原`agentrl_worker-0.4.0-py3-none-any.whl`，固定SHA由执行器校验。原文件均单独取得，不重新分发。

将三份游戏归档按原目录解压到同一授权目录。数据准备环境中执行：

```sh
workload-data agentbench-native-tasks \
  --official-archive /authorized/agentbench.zip \
  --game-archives /authorized/json.zip /authorized/pddl.zip /authorized/tw-pddl.zip \
  --data-root /authorized/original-games --indices 0 1 2 > /private/native-references.json
```

引用只包含原始索引、路径、SHA和执行要求，不包含任务描述/目标、原状态JSON或评分载荷。它既不是模型请求也不是已执行轨迹。执行器重新核对官方索引、三份原归档和引用中的全部本题资产，不能通过自行改题并重算SHA绕过发布方原字节。

## 原生ARM环境与运行

下面是Linux aarch64专用PDDL构建路径；准备一个新的Python
3.9虚拟环境，不能修改正在提供模型服务的环境。依赖文件是实测版本快照，不包含模型权重。Torch仅供原环境库导入，推理仍通过NPU服务。

```sh
python3.9 -m venv /private/af-env
env -u PYTHONPATH -u PYTHONHOME -u LD_PRELOAD /private/af-env/bin/python -m pip install \
  -r configs/workload-native/agentbench-alfworld-py39-dependencies.txt
env -u PYTHONPATH -u PYTHONHOME -u LD_PRELOAD /private/af-env/bin/python \
  scripts/build_agentbench_pddl_runtime_wheels.py \
  --downward-archive /authorized/downward.zip \
  --textworld-archive /authorized/textworld.zip \
  --alfworld-archive /authorized/alfworld.zip --output /private/fresh-runtime-build
env -u PYTHONPATH -u PYTHONHOME -u LD_PRELOAD /private/af-env/bin/python -m pip install \
  /private/fresh-runtime-build/wheels/*.whl
env -u PYTHONPATH -u PYTHONHOME -u LD_PRELOAD /private/af-env/bin/python -m pip check
```

环境须具备`libGL.so.1`及其依赖，与官方Dockerfile的图形库要求一致。本次服务器按官方Debian bookworm
arm64包索引取得并校验共享库文件，解包到任务私有目录，只为原生任务进程设置LD_LIBRARY_PATH；未全局安装系统包或执行包维护脚本。若目标主机已有库则无需这一步；缺库须补环境，不能靠安装两个互相覆盖的OpenCV发行包绕过。

构建仅调整打包钩子和依赖元数据，不修改runtime/task
Python文件。所有输出目录必须新建，构建失败保留状态，不等同任务执行。随后（如使用私有图形库，须给该进程配置对应LD_LIBRARY_PATH）：

```sh
env -u PYTHONPATH -u PYTHONHOME -u LD_PRELOAD /private/af-env/bin/python \
  scripts/run_agentbench_alfworld_original.py \
  --official-archive /authorized/agentbench.zip \
  --game-archives /authorized/json.zip /authorized/pddl.zip /authorized/tw-pddl.zip \
  --runtime-source-archives /authorized/textworld.zip /authorized/alfworld.zip \
  --worker-wheel /authorized/agentrl_worker-0.4.0-py3-none-any.whl \
  --data-root /authorized/original-games --task-references /private/native-references.json \
  --output /private/fresh-native-run --endpoint http://127.0.0.1:19466/v1 \
  --model workload-qwen35-27b-128k-nonthinking
```

原任务类保留工具定义、20步上限、动作校验、终止和评分。框架reward记录不作为模型消息；其余原Session消息和tools直接交给服务。temperature=0、max_tokens=2048及现有Qwen3.5-27B替代模型都是显式部署配置，不冒充作者论文协议。脚本复制原选中任务到私有运行目录，不改变授权原数据；仅允许本地模型端点。

模型答错、无效动作、步数耗尽、服务错误和取消分别保留。构建/环境探测是0模型调用的独立证据，不能计入完整任务数；只有完整原生交互/终止和适用官方评分才更新来源勾选。CPU
CI只测试临时软件夹具、引用契约和来源边界，不下载原题、不构建native runtime、不跑模型或性能。

2026-10-24核查固定官方`task.py`与`SampleStatus`后，修正执行器的结果分类：原生`task limit reached`和`agent invalid action`是可由官方聚合评分的任务失败，不要求答对才算执行过；只有带实际模型调用及原生0/1奖励的受支持终态，才具备所选协议的执行资格。`task error`、`unknown`、取消，以及上下文/协议不适配仍不算完成。模型响应若以`length`结束，则保留2048预算及原输出，交给原生Session和动作校验，不追加生成、不改动作、不换题。这个执行器修正目前仅有软件测试，旧三题执行回执及其源码SHA保持历史事实；不得将旧回执称为修改后执行器的实跑证据。执行器中的`source_checkbox_eligible`只描述本次选题的执行条件，不自动代表全来源交付完成。

后续进展：已启动其余106题的固定原索引执行队列，按官方六组分别运行，复用历史0–2题，不因失败换题。复用已核验的原Python3.9/标准OpenCV/libGL环境与同一Qwen3.5-27B服务；每组串行交互，并与独立StructEval正确性队列共享服务既有两个请求槽，不改变服务或20步流程。不记录或推断吞吐、延迟等性能结论；研究线程优先。当前只是运行中，不作为修改后入口已通过或来源完成的证据，需保存并核查各组实际终态、原题资产及官方评分后再更新。

首组复核结果：pick_and_place新增原索引3–8共6题均已执行终止并经官方聚合评分复算，与保存结果一致；其中原索引7为`agent invalid action`、奖励0，原样保留。逐项核验实际模型请求/响应、原生结果SHA及运行副本/原目录中的游戏资产字节；审计调用模型0次。连同复用0–2题，该官方组全9题验证完成，这是修改后执行器的实际执行证据。其余100题尚待验证，不能称全部109题或AgentBench整套完成。[首组审计回执](validation/agentbench-original-complete-groups-audit-20261024.json)。

第二组进展：pick_clean_then_place的原索引9–13共5题也已完成相同审计；现前两组全14题验证完成（新增11题、复用3题），其余95题仍待验证。同一[增量审计回执](validation/agentbench-original-complete-groups-audit-20261024.json)保留每组的实际任务索引、终态和官方评分，运行中的其他组不计入。

第三组进展：pick_heat_then_place的原索引14–40共27题完成同一套原始资产、模型响应、结果SHA和官方聚合评分审计。累计前三组全41题（新增38题、复用3题），其余68题尚未完成验证。按研究线程优先约定，不启动下一组；模型服务已在最后一个并行验证任务自然结束、请求清零后正常关闭，数据与证据保留。此资源交接不代表来源交付完成，也不影响继续CPU审计和软件检查。
