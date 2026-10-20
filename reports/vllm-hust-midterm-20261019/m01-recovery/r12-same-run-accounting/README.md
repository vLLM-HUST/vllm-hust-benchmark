# r12 同轮核算

本目录保存 r12 的派生分析，不替换原始请求、设备轨迹或正式 B1。

| 文件                                | 内容                                                            |
| ----------------------------------- | --------------------------------------------------------------- |
| `r12-hardware-intervals.json`       | 三个命名区间的 HBM 积分、上下界及 HCCS TX 计数                  |
| `r12-matrix-accounting.json`        | 双卡共同确认的请求区间内矩阵小计及边界、尾部小计                |
| `vocab-projection-summary-r12.json` | 同轮词表投影行数与语义运算量，包含 drain，未证明设备完成边界    |
| `cross-run-cache-policy.json`       | 后续运行的缓存隔离要求及 r12 有限核查范围                       |
| `sources.tar.gz`                    | 生成核算结果和本机验证时的原始脚本，保留文件字节及源文件哈希    |
| `hbm-db.tar.gz.part-*`              | 本轮系统采样导出的两卡 HBM SQLite 数据库，顺序拼接后为 gzip tar |

请求与设备原始证据复用相邻目录 `deferred-device-r12/archive.json` 所索引的归档。 源脚本和汇总中的绝对路径是原采集环境的来源记录，不是其他容器的部署要求。

仓库验证命令：

```bash
pytest -q tests/test_midterm_r12_accounting.py tests/test_midterm_deferred_r12.py
```

测试从已归档轨迹重建矩阵上下文与 token 总量、路由行数、词表行数和 HCCS 上下界， 并从本目录数据库重新核验 HBM 原始计数与导出带宽的一致性及三个区间积分。
它不证明全部模型算子覆盖、各算子设备完成边界或全部历史 B0/B1 缓存策略一致性。 完整 F 和 M01 达成率保留为空值。
