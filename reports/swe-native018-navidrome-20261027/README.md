# SWE-bench-Pro指定0.18基线：Navidrome增量实测

指定0.18基线已执行15/731个不同任务，官方原记录2/15、未决0；1题补丁应用未获证明，2题确定补丁应用失败，均不能视为已应用候选补丁的功能验证。尚未执行716题，不是全量验收或B0/B1配对。评分未决0不代表所有预期测试名齐全。

## 配置与方法

Qwen3.5-35B-A3B，vLLM 0.18.0 / vLLM-Ascend
0.18.0；100步、1200秒、max_tokens=4096、temperature=0、seed=0、thinking=false、零自动重试。复用相同运行身份的5项资格，不增加资格请求。原镜像go.mod
fixture按文件与diff精确绑定，不删除或改写；Agent与评分各使用独立容器，原官方评分器不变。

## 结果

Navidrome仅一次Agent，53次模型调用、511.436103秒，主动提交。原9941字节补丁应用成功，应用后与测试前实际待评分文件身份一致。原Scanner
Suite执行27项spec，26通过、1项Images字段断言失败；嵌套spec不代替官方目标测试名评分：F2P为0/1、P2P无目标，预期测试名缺失0。原解析器记录TestScanner失败，官方任务判false。保留原TagLib弃用警告，未重跑Agent或评分。

模型服务、Agent及评分容器均已释放。历史14题及其应用/编译限制保留，未覆盖原件。

## 证据

summary-navidrome.json为聚合；review-navidrome.json为独立复核，release-navidrome.json及custody-navidrome.json记录释放和受限原件身份。私有题目、答案、补丁、请求标识和轨迹不公开。

离线执行 `python3 verify.py` 复核公开聚合与哈希，不声称无需受限原件即可重跑原始审计。
