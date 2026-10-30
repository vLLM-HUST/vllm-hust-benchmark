本目录提供中文合并报告、冻结源文件清单和确定性证据归档。报告中的实验PASS不等于质量PASS；07338退化仍未修复，PR27保持Draft。

先阅读
`REPORT-v1.md`。`source-manifest-v1.json`列出所有实际归档文件；`external-reference-index-v1.json`区分随包文件和仅记录路径/SHA的早期外部依赖。`package-receipt-v1.json`给出实际大小和SHA。

在原服务器、源文件仍保持清单记录的字节时，可重建到一个新文件名：

```sh
python build_evidence_v1.py build --manifest source-manifest-v1.json --archive reproduced-evidence-v1.tar.gz --output reproduced-receipt-v1.json
python build_evidence_v1.py verify --archive reproduced-evidence-v1.tar.gz --output reproduced-verification-v1.json
```

输出文件已存在即拒绝覆盖。确定性gzip/tar固定时间戳、uid/gid和顺序；相同清单及输入字节产生相同压缩文件。build会检查源目录有无新增或丢失文件，并在写入每个成员时核验SHA；失败时可能留下不完整输出，不能把它当成通过verify的包。

只核验收到的包不需要模型、NPU、网络或原目录：

```sh
python build_evidence_v1.py verify --archive muxpd-first-mixed-evidence-v1.tar.gz --output local-verification-v1.json
```

verify逐项检查包内MANIFEST、重复/缺失/额外成员、字节数与SHA，不会解压执行代码。逐项核验成功不代表本包包含所有外部历史大文件，也不代表答案质量或性能通过。

本轮源清单生成方式保留如下；已有文件不能覆盖，应在新版本目录使用新文件名：

```sh
python build_evidence_v1.py references --config source-roots-v1.json --output external-reference-index-v1.json
python build_evidence_v1.py plan --config source-roots-v1.json --output source-manifest-v1.json
```

脚本全部测试为CPU小型临时fixture，覆盖终态失败保留、未终态/残留进程拒绝、缓存排除、符号链接拒绝、确定性归档，以及源文件变更/遗漏拒绝。没有重新运行数值或设备实验。
