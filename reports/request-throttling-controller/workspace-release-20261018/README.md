# 本实验三个工作区停止与资源释放

所有指定本MOD数据集完整质量、独立性能、独立核验、脱敏备份和原矩阵线上发布验收完成后，按用户既定授权仅执行三个自有专用工作区的coder
stop。没有删除workspace、home、hostPath或PVC，没有操作其他用户资源。

最新Coder身份/UUID与原计划匹配，实际cluster/head UID匹配；主机boot_id SHA与Pod内一致，六个NPU进程在停止前逐一与对应Pod
cgroup绑定。三次stop均exit0；停止后三个workspace均stopped、对应RayCluster及六Pod消失，原六进程不存在，目标六张卡的npu-smi均无进程。三个实验root在同内核hostPath继续存在。

原始终态队列及Ray实验driver日志、停止前服务日志快照、Coder账本、Pod规格、内核绑定、NPU前后快照和停止回执均私有保留。Gold旧r2状态文件残留RUNNING但真实Ray
job已STOPPED，保留历史而不恢复。独立私有核验原包31文件hash、身份/进程归属、队列终态、停止结果、NPU请求释放与持久目录保留PASS。

公开仅含脱敏身份hash、统计、私有archive
SHA及独立verifier；没有真实地址、主机名、原始生成内容或凭据。资源可用仅代表观测时刻，不授予未来租约。SWE-bench-Pro真实修复仍未测，缺已确认归属amd64
Docker宿主和支持工具调用的另冻服务profile，不能把未测写0%。
