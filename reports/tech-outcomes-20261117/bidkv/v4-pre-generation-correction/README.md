# v4正式生成前的路径修正

最终CPU路径审计发现初版campaign的动态目录仍含native03和C8-03；当时尚未执行C8或完整五项。原startup、Native05模型进程、启动命令和carrier身份未变，不因此重新加载模型。

[更正receipt](correction-receipt.json)、[初版manifest](source-manifest.json)和[原campaign字节](campaign81.py.source)保留。该manifest只引用初版的全部成员，公共本目录仅归档变更前campaign；不声称其余初版成员在此重复公开。

最终源见[v4](../driver-mod-v4/README.md)。修正后的实际Linux
CPU检查执行三路solver/judge路径构造、生命周期子路由和C8输出/读取路径相等检查，使用mocked模型与health，0NPU、0生成；它证明操作路径合同，不证明模型或性能通过。该CPU检查首个fixture因临时父目录缺失失败的记录受控保留，修正后的第二次真实返回码0。
