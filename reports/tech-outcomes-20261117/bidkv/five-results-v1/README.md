# BidKV完整五项：已执行，未建立机制收益

[完整中文报告](report/REPORT.md)、[HTML](report/report.html)、[图表](report/throughput.svg)和[派生摘要](report/summary.json)保留所有正负变化、原质量、失败/截断/未决、成本与范围。

两臂各672条solver和720条共同Native judge输入均完成，14stage原SSE在受控环境独立重放2064条、四份原grader
CPU重算逐字节一致；公共仓库提供逐请求数字与原件SHA，并没有公开/重新读取私有原问题、答案或SSE。两臂full
solver策略调用/决策均0，不能把吞吐或质量差异归因BidKV。MMLU144/200→149/200；HLE32/200→33/200并保留94/92未决；Frontier
Olympiad63/100→55/100、Research1/60→2/60，未决8/4，不重判挑选。SWE吞吐负变化保留。

原exporter的[manifest](manifest.json)、[summary](summary.json)、四份原质量汇总、[受控原件库存](controlled-originals-index.json)、执行计划和源码身份均保留原字节。[closure-manifest](closure-manifest.json)原`audit-final-records81.py`仅公开改名为`audit-final-records81.py.source`，字节SHA不变，由根校验器显式映射；其余路径不变。后来补充的[真实生命周期/策略](full-owned-lifecycle-policy.json)、[实际启动身份](actual-execution-identity.json)及本报告由项目根manifest继续覆盖，不改写原exporter/closure
manifest。

正式controller的exit_code来自实际process.wait()；仅取消路径使用的actual_client_returncode保留null。Popen前真实构造的command.json不冒称独立/proc
argv或正式服务loaded-import
RPC。模型退出/双卡释放、诊断4条、初始化及内存采样分别归档。[闭合与原质量复算](closure-and-quality-recompute.json)保留actualwait；[只读数值校验](../verify_full_five.py)不需要模型/私有输入。

报告生成源和5项CPU反例在report目录，两次真实CPU重建的五件成品逐字节相同，receipt不冒充两次模型实验。连续C8资格另见[c8-results-v2](../c8-results-v2/README.md)；1GiB人工KV压力诊断也独立，不能替换保留EOS的本完整五项。
