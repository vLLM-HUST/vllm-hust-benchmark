# break-filter 有界字节诊断

原 r2 正控制 reward=0、原 5 秒 alert 等待和 r1 未评分 null 均保留。本次在同一派生镜像的新鲜诊断容器中运行原 oracle/filter，未启动浏览器
session，未运行原控制或模型。

原 oracle 输出 41 字节，filter 成功后输出 20 字节。原输出含有的 literal script 与 alert token 在新过滤输出中均不存在。该结果支持本次诊断环境中的
filter 路径解释，但不能还原已删除的原浏览器 DOM，也不能证明具体 parser 版本、ARM64 或时序原因。

诊断 pytest 的 1 项通过不是原控制通过。记录新的临时依赖身份，不冒称原已删除容器安装态。原任务、filter、oracle、tests 和 5 秒阈值均未改。1 CPU、2
GiB、cpuset 24-31、MTU 1400、精确四挂载及 34 文件封存和释放已经独审。

公开文件仅来自 whitelist.json：名称、状态、字节长度与 SHA、回执投影和受限原件索引。不含 HTML/答案正文、原命令、请求身份或私有路径。python verify.py
仅离线核验，不重做诊断或控制。
