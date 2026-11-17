# 原v2审计guard失败保留：仅补独立Candidate诊断

原Native8条HTTP/SSE均完整成功、16407输入/4096输出，但9d3客户端在完成后误把scheduler内部ID当作wireID而断言失败。真实clientRC1、controller/ownedlauncherRC2，原attempt保持failed，原Candidate未执行；不是模型错误或600秒超时。NativeAPI/owner退出0、双卡已释放。

固定0fc
InputProcessor.assign_request_id会保留externalID并默认追加`-`加8个随机小写hex为内部ID，OutputProcessor返回externalID。独立核对官方/实际源码SHA及8个精确一一映射；没有关闭随机化、模糊截断或放宽外来请求检查。原8份SSE、body/messages、盐、tokenusage、7个decision的candidate/selected/builtin/requesting及全部计数/fault重新验证，见原字节[carry审计](native-carry-baseline-audit-v3.json)和[源码/映射](official-id-mapping-and-source-identities.json)。

后续v3是独立Candidate-only修正诊断，明确沿用**全部**原Native8作baseline，不重发Native、不选轮；新Candidate仍8条/4096/1GiB/原输入、600秒上界和原3600秒clock。原v2不会改成completed。v3仅修观测ID绑定并独立lifecycle/release，运行SDK/selector/HTTPbody未改。13dc
Candidate零生成owner提前按parent授权启动，新的actualwait
wrapper当时尚未公开，本次补发源码并如实注明；Candidate任何生成须本次修正源/证据公共读回及新release。

无论修正对照完整与否，全部partial/失败/负值/0调用保留；不能冒充未修订的原v2、正式五项或普通16GiB收益。新的paired结论必须披露观测guard修正与carry身份。
