# SWE-bench-Pro：第二批前三题增量结果

模型为 Qwen3.5-35B-A3B，指定 B0 为 vLLM 0.18.0 / vLLM-Ascend 0.18.0。 每题沿用原官方 Agent 与评分器，单次尝试、100 步、1200
秒、max_tokens=4096、零自动重试。 复用五项已完成模型资格，不新增资格请求。评分阶段不调用模型。

| 任务    | 模型调用 |   Agent 秒 | 提交 | 补丁应用 | F2P | P2P     | 官方结果 |
| ------- | -------: | ---------: | ---- | -------- | --- | ------- | -------- |
| nodebb  |       54 | 265.495685 | True | exit 0   | 0/3 | 252/252 | false    |
| ansible |       36 | 361.464713 | True | exit 0   | 4/4 | 26/26   | true     |
| element |       96 | 511.628888 | True | exit 0   | 0/3 | 0/0     | false    |

三题均完整保存应用前、应用后及测试前实际待评分状态；应用成功，应用后与测试前文件身份一致。 NodeBB 原 Mocha 为 262 pass / 3 fail，原解析器为 252 pass / 3
fail，二者分列；三个必要目标均为真实断言失败。 Ansible 原解析器 30 项通过，4 个 F2P 与 26 个 P2P 全部覆盖。两题 missing=0，未改补丁、评分器或原结果。
Element 原 Jest 实际为 1 pass / 2 fail，两个失败均为快照断言。原解析器仅从 PASS 文件块收集，略过本次 FAIL 文件，输出为空；官方 F2P missing=3
与原 false 保留。 Element 的 missing=3 不是三个实测断言失败，也不是测试未运行；不修补原解析器或重复评分。

累计 21/731 项尝试；原官方记录 3/21，披露同一 Proton 答案依赖恢复后 4/21；710 项未执行。恢复评分不新增 Agent 尝试。
此前应用失败、应用证据受限、编译失败及未提交部分补丁均保留；不代表全量验收或 B0/B1 配对通过率。

## 证据与复算

- [累计规范结果](catalog.json)；[原记录与恢复结果区分](current-aggregate.json)。
- [NodeBB 原聚合](summary-nodebb.json)；[独立复核](review-nodebb.json)。
- [Ansible 原聚合](summary-ansible.json)；[独立复核](review-ansible.json)。
- [Element 原聚合](summary-element.json)；[独立复核](review-element.json)。
- 三份原聚合含冻结预算、实际运行身份、评分源码及受限原件 SHA；独立复核含 CPU 与服务封存、资源释放证据。
- 私有题目、答案、补丁与请求标识不进入本目录。

在仓库根目录运行：

```sh
python reports/swe-native018-wave2-first-three-20261027/verify.py
```
