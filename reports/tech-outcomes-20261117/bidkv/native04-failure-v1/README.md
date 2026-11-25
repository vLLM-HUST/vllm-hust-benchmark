# Native04：共同运行包零生成失败

默认True CPU稀疏映射修复使用的首个copied
Ascend包漏了原安装中的CANN算子元数据/内核资源。模型初始化实际报`aclnnAddRmsNormBias`、`EZ1009`和缺失`binary_info_config.json`，未通过API健康检查，**0生成、0
C8、0五项正式请求**。这不是MOD吞吐结果。

[原字节failure capsule](failure-capsule.json)保留5条fatal摘录、9份受控原件大小/SHA、804个原安装CANN资产的库存与失败copy存在标记，以及真实owned
wait状态：预检子进程−2，API清理等待120秒后由owned cleanup结束，实际−9。完整日志与设备释放记录在受控目录；公共清单不冒充完整日志重新读取。

对应失败源见[driver-mod-v3](../driver-mod-v3/README.md)。后续v4仅补完整原安装资源并保留相对symlink/原字节清单，CPU补丁本身不变；不改原安装树、不重建算子、不更换SDK。后续模型健康、实际绑定、C8和完整五项必须另行实测。
