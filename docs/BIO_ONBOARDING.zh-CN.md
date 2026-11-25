# 生信原生任务的注册与驱动接入

本接口服务于已适配生信模型的项目评测：让执行、独立参考、统计与原件复核复用同一条流水线。模型神经网络和生产 runtime 仍由 Scientist 仓库维护。当前桥接范围为 ESM-2 150M 与
ProGen2-small 的七类原生任务，采用单卡 FP16 eager；不是正式榜单成绩，也不宣称其他家族已经验收。

## 注册结构与版本

公共实现位于 `src/vllm_hust_benchmark/bio_registry.py`、`bio_interfaces.py`，JSON Schema 随包发布。模型侧示例是
Scientist 的 `docs/benchmark/native_registry.json` 与 `registered_suite.json`。

注册表定义三层身份：

- `components`：组件 ID、版本、类别、接口 ABI 版本及 `module:Class` 导入入口。
- `contracts`：任务 ID、版本、operation、输入/输出类型和严格的参数 JSON Schema。
- `profiles`：具体发布物 ID、版本、资产 ID、五类组件及各自配置、精度/模式/上下文限制、原生能力、任务绑定和拒绝探针。

同一个任务契约可以绑定多个发布物；任务绑定 ID 在每个 profile 内唯一。一个 profile 共享一个 runtime、输入适配器、参考提供者及资源采集器，任务允许独立覆盖
evaluator。若同一权重需要不同 runtime 或参考图，使用不同 profile、共享 asset_id。

注册版本描述配置内容，组件版本描述实现语义，`interface_version=1` 是调用 ABI，contract version 描述任务语义。版本或内容变化会改变注册表规范化
SHA256，旧结果继续保留原注册表和源快照。注册文件是可信代码配置，只加载已审阅的 Python 入口；组件构造函数不得分配模型或设备。

## 五类接口

| 接口                | 方法与职责                                                                                                                                                                                    |
| ------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `RuntimeDriver`     | `supports` 声明契约支持；`validate` 做分配前检查；`open` 构造真实 runtime；`execute` 调用原生任务；`diagnostics` 保存额外证据；`work` 从真实输入输出核算工作量；`rejections` 生成具名拒绝探针 |
| `InputAdapter`      | `supports`、`validate`、`prepare`；准备后的 identity 保存序列、token、样本与源窗口的对应关系                                                                                                  |
| `ReferenceProvider` | `supports`、`load`、`expected`；在独立子进程运行原作者/HF 计算图，不把被测引擎当参考                                                                                                          |
| `OutputEvaluator`   | `supports`、`compare`；返回具名指标、固定阈值和通过状态，供即时比较与离线重算共用                                                                                                             |
| `ResourceCollector` | `snapshot`；从实际 worker 读取内存、PID、进程启动身份等资源证据                                                                                                                               |

每个实现提供 `api_version` 和 `config_schema`。方法签名以 `bio_interfaces.py`
为准；数据依具体任务契约传递，不把所有输出强制转换成生成文本。`DriverSession` 管理一次 runtime
的启动、观察、执行与关闭，禁止重复启动、关闭后调用及跨注册/发布物执行。runtime 必须提供 `close()`。

## 执行路径

`CLI → suite/schema 与数据校验 → Registry.preflight → InputAdapter.prepare → DriverSession.start → RuntimeDriver.execute → 原件归档 → 独立 ReferenceProvider → OutputEvaluator → 统计与报告 → verify`。

预检查在模型构造前拒绝未知发布物/任务、错误组件类别/版本/配置、参数拼写或类型错误、契约不兼容、精度/模式/上下文越界及缺少覆盖的用例。输入坐标和生物字母表由输入适配器负责。调用执行层前还会以真实
tokenizer 核对输入身份。注册声明不代表能力验收通过；能力边界、资产及任务还必须实际执行验证。

通用循环不按 ESM-2 或 ProGen2 的发布物名字分派。模型侧桥接负责 tokenizer、特殊 token、坐标、调用参数、参考计算和拒绝行为；这部分允许保留与具体家族有关的实现。

## 新模型接入操作

1. 确认生产 runtime 已能运行目标任务，固定权重、tokenizer、参考源、运行模式及数值验收门槛。
1. 选择已存在的 contract，核对输入输出和语义。语义不一致时新增版本或契约，不能借用相近名字绕过验证。
1. 评估五类实现能否复用。只有 runtime 方法、codec、参考图及 evaluator 的语义全部兼容才可只新增 profile 配置；其余实现一个薄桥接，新的执行类型才新增 driver。
1. 在与 registry 相邻的资产清单登记原件、runtime 视图和参考来源。注册 profile 指向资产 ID；相对清单文件名必须可移植。
1. suite 使用 `schema_version=2`，声明 registry ID/版本、明确的 profile→任务 coverage、数据 SHA/窗口、用例、运行协议及数值
   gates。参数必须符合对应 contract Schema。
1. 先运行 `check`；通过后运行完整实测、独立参考及 `verify`，保存失败记录。别名或 mock 只能证明接口路由，不能作为新模型验收。

当前 ESM codec 只适用于已验证的 ESM token/坐标语义；ProGen codec 与 retained evaluator 只适用于对应官方特殊 token、停止和采样规则。其他
encoder、RITA/RNA 等家族需要先证明兼容或新增 codec/reference/evaluator；不得只改目录就宣称接入成功。

## 运行入口

使用同一 Python 环境安装本包的 bio 依赖，并安装项目固定 Torch/NPU/runtime 环境。不要升级共享环境来满足本工具的可选依赖；可创建带
system-site-packages 的隔离环境。

```sh
python -m pip install -e '/root/vllm-hust-benchmark-workspace[bio]'
python -m vllm_hust_benchmark.bio check \
  --scientist-root /root/scientist-lightweight-vllm-workspace \
  --registry /root/scientist-lightweight-vllm-workspace/docs/benchmark/native_registry.json \
  --suite /root/scientist-lightweight-vllm-workspace/docs/benchmark/registered_suite.json
python -m vllm_hust_benchmark.bio run \
  --scientist-root /root/scientist-lightweight-vllm-workspace \
  --output /root/vllm-hust-benchmark-workspace/.local/bio/my-registered-run
python -m vllm_hust_benchmark.bio verify \
  --output /root/vllm-hust-benchmark-workspace/.local/bio/my-registered-run
python -m vllm_hust_benchmark.bio report \
  --output /root/vllm-hust-benchmark-workspace/.local/bio/my-registered-run \
  --destination /root/vllm-hust-benchmark-workspace/.local/bio/my-recomputed-report
```

`run` 默认使用注册版七任务套件。`--pilot` 缩短重复次数，结果明确标记 pilot，不计作完整验收。资产/data/upstream/python 均可显式指定；单独运行模型侧
native/reference 也必须使用安装了本包的相同解释器。

## 原件、兼容与后续验收

新格式 manifest 记录 registry ID/版本/SHA、原始 registry 路径、任务 coverage、源文件摘要、资产及实际请求；输出目录另存 registry
与资产清单、输入数据、源快照、模型原始输出和独立参考。`verify` 比较归档与当前注册身份、数据、源和证据清单，再重算数值、工作量、时钟、拒绝恢复、清理及报告。不能把更换驱动后的旧数据当作新实测。

旧 schema-v1 两模型套件仍可显式传入；旧格式记录的 report
保留原计算口径。旧记录没有注册证据，不会补写或冒充新版验收。默认注册表显式要求第一批原生入口回归；其他注册表若关闭该兼容回归，结果仅属于其声明的注册任务范围，不会把未运行的回归写成通过。

本次实现解决注册、驱动与现有原型迁移。下一步是在不修改通用执行循环的前提下，为 encoder 和自回归路径分别接入第二个真实模型家族，完成跨家族复用验收；随后再扩展新的输入输出和任务类型。当前不支持
streaming、cancel 或并发服务实验，不能用离线批处理结果代替这些能力。

## 跨系列配置

Scientist 的 `docs/benchmark/registered_cross_series_registry.json`
提供四发布物绑定；`registered_cross_series_suite.json` 运行新两模型，`registered_cross_series_all_suite.json`
可运行四模型。输入、作者默认值、BPE边界、受保护公共代码及RNA contact非有限输出的分层判定见同目录
`registered_cross_series_protocol.zh-CN.md`。实际通过与科学有效性以运行证据分别判定。

## ProstT5 配对翻译与条件评分

在配套 Scientist 工作区中，`docs/benchmark/registered_prostt5_registry.json` 注册双向翻译与条件评分，
`docs/benchmark/registered_prostt5_suite.json` 选择用例。将两个路径分别传给现有 CLI 的 `--registry`、`--suite`，并将
`--data` 指向同工作区的 `docs/benchmark/data/prostt5`。 用例使用 1CRN 同一条链的氨基酸与结构字母窗口。评分区分源长度、目标长度和实际参与计分的位置，
翻译对照独立参考的 token 与解码文本。源码依赖可以在注册表的 `scientist_sources` 中按仓库相对路径声明， 运行时随注册适配器一起归档并核验。完整协议见 Scientist
工作区的 `docs/benchmark/prostt5_protocol.zh-CN.md`。
