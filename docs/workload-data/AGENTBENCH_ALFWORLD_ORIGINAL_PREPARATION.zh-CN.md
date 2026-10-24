# AgentBench ALFWorld FC 原始任务接入与验证

原new_std索引0–2已完成原版任务/AgentRL
Session、NPU模型交互及官方评分，三个任务均正常结束并得分1。它们均为pick_and_place_simple，只证明具名三题接入，不是全109项或其他AgentBench环境通过；不是QBI派生任务、静态回放或独立ALFWorld结果冒充AgentBench。

- [x] 固定官方AgentBench提交`d1e4a10db08c87075c78972e48ecc182be03e2d5`及完整源码归档SHA。
- [x] 按该版本官方Dockerfile所指的ALFWorld源码和0.2.2发布游戏归档下载、校验并在服务器解包，不使用旧缓存中不存在的预选任务替代题。
- [x] 原`new_std`索引共109项，按原加载顺序预选0、1、2；对应游戏文件均存在并与官方来源归档字节一致。[准备回执](validation/agentbench-alfworld-original-preparation-20261022.json)。
- [x] 三条官方原生任务、AgentRL
  Session交互及原生终止/评分，分别4/7/20步、合计31次模型请求。[执行回执](validation/agentbench-alfworld-native-npu-20261022.json)。
- [x] 公开构建脚本、固定依赖的干净安装及pip
  check；最终标准OpenCV环境同三题复验仍为1/1/1。[交接复验](validation/agentbench-alfworld-public-handoff-npu-20261023.json)。两次运行合计仍为三个唯一任务，不计六题。
- [ ] 其他任务/子集、全部109项，以及AgentBench的其他环境。

2026-10-24使用固定CI数据组件，在服务器复核并准备了全部109个官方任务引用及其原始归档资产。原索引有六组：pick_and_place 9题、pick_clean_then_place
5题、pick_heat_then_place 27题、pick_cool_then_place 29题、look_at_obj 18题、pick_two_obj
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
