**内置与外置q40在相同源码和32题执行历史下，全部完整tokens、文本、finish/stop、评分、映射调度和四rank
collective几何相同。两者都29/32，01741仍新增答错（E→C，5551 tokens、stop）；full200未启动。**

移植对照的等价性通过不等于质量修复，也没有证明quota或某个算子是唯一根因。07338恢复H/3292与此前八题局部成功不能代表整体通过。

- [完整报告](REPORT-v2.md)、[32题机器可读表](posterior32-table-v2.json)
- [内置完整输出](builtin-run-v1.json.gz)、[外置完整输出](external-run-v1.json.gz)
- [外置主分析](external-analysis-v1.json.gz)、[独立复核](external-independent-v1.json.gz)
- [builtin源码包](builtin-source-capsule-v1.tar.gz)、[external增补源码包](external-source-addendum-v1.tar.gz)、[发布映射](publication-map-v2.json)
- [原PENDING草稿](REPORT-v1.md)保留，当前状态见[已终态状态](external-status-v2.json)。

无完整200题、性能或Goodput通过声明，无NPY或匿名HTTPS读回声明。
