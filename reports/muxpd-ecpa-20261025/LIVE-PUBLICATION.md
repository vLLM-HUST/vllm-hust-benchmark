# MuxPD 插件上线回验

MuxPD 已出现在[公开插件页](https://vllm-hust.sage.org.ai/plugins.html#muxpd)，提供版本化安装包、显式启用与回滚命令、受限 TP4
资格和数值问题入口。此次回验检查实际公开页面；没有执行页面命令或新增模型实验。

## 发布身份与实际页面

| 对象     | 固定身份与证据                                                                                                                                                                                         |
| -------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 软件     | [v0.2.0a1 预发行](https://github.com/vLLM-HUST/vllm-ascend-MuxPD-hust/releases/tag/v0.2.0a1)，原有 wheel、sdist 和校验和未重建                                                                         |
| 功能资格 | [19081ff 的 TP4 报告](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/19081ff39a25c3c8ac821ae7885688ea24a03e90/reports/muxpd-ecpa-20261025/RELEASE-QUALIFICATION.md)；保留原始失败与软件身份区别 |
| 目录数据 | benchmark `17ee3549ddb757c9198763f27666af34d50958f6`，由 [PR #505](https://github.com/vLLM-HUST/vllm-hust-benchmark/pull/505) 合并                                                                     |
| 网站代码 | `d1b225800743918da208e6585cd850f6013c5b0c`，由 [PR #545](https://github.com/vLLM-HUST/vllm-hust-website/pull/545) 合并；与通过最终 PR CI 的源提交整树相同                                              |
| 正常部署 | [Pages 40235244746](https://github.com/vLLM-HUST/vllm-hust-website/actions/runs/40235244746) 成功；实际公开 HTML、JS、CSS 四份文件与合并提交逐字节一致                                                 |

2026-10-25 21:28 UTC 的 Chromium 实页检查覆盖手机 390×844、桌面 1440×1000 和中英双语，共四种视图。浏览器实际读取的四张目录表在每个视图均匹配线上
manifest，并由独立比较确认这 16 份响应与 `17ee3549` 的 Git 原件逐字节一致。manifest 记录 142 份发布文件；此次没有逐一访问所有 142 份文件。

实际页面验证了运行时分类筛选、三个集成契约、实验性标记、TP4 范围、6/8 文本一致说明、数值与性能边界，以及安装、启用、serve 和下次启动回滚命令。安装指南、软件 Release、Issue
#23 在正常滚动和卡片截图后的滚动位置共通过 24 次 trial-click 与中心命中检查；面板高度不超过 70vh 并可滚动。另行匿名访问五个公开 GitHub 目标均返回 HTTP 200
和正确页面标题。浏览器未导航执行这些链接，也未执行展示的命令。

最终 PR CI 的原始浏览器附件也已重新下载，ZIP 大小和 SHA256 与 GitHub API 相符。其中八种宽度、语言、亮暗主题组合共有 48 次链接检查通过；保留了完整 receipt 和
16 张 MuxPD 截图的精确成员字节。此处的实页检查和 CI 检查是两组独立证据。

## 公开证据与边界

新的网站回验证据追加在[同一 benchmark 证据 Release](https://github.com/vLLM-HUST/vllm-hust-benchmark/releases/tag/evidence-muxpd-ecpa-20261025)，使用独立名称
`muxpd-website-publication-20261025.tar.gz`、`website-publication-archive-index.json`、`website-publication-archive-verification.json`
和 `WEBSITE-PUBLICATION-SHA256SUMS`。原来的四个设备证据资产没有覆盖；21:27 UTC 的重新下载又确认其全部 347 个成员保持原字节。21:31 UTC
的[下载回验](publication-release-download-verification.json)确认四个新资产与冻结字节及 API digest 一致，全部 116
个成员匹配[清单](website-publication-archive-index.json)。新归档 SHA256 为
`bc3e4121f89c7423a00d6bdd6a3e4ce9fed240223f1e16dcf71533215d66ec07`；[事实摘要](website-publication-facts.json)逐项指向归档内原件。

本次仅追加上线报告、证据和索引，不改目录数据；网站继续固定 `17ee3549`，不为此报告重新固定数据版本。当前包仍是受限实验版本：最终直接设备轮次自然文本 6/8 一致，未证明逐
token/文本等价或语义质量，历史约 3.1 倍 SLO Goodput 属于另一软件身份，不是当前 wheel 的性能结论。数值问题
[#23](https://github.com/vLLM-HUST/vllm-ascend-MuxPD-hust/issues/23)、性能问题 #3、Group-FX 问题 #11 和草稿 PR
#10 保持独立。

两点证据保管说明：环境重建后，尚未上传且位于 `/tmp` 的早期本地网页诊断原件已不可用，未按对话摘要重建或冒充已归档证据；此报告以重新取得的远端 CI
附件和本次实际线上检查为依据。新链接检查的首次脚本误猜了 `docs/ECPA.md`，产生 404；网站本身的链接一直是
`docs/ecpa-deployment.md`。错误输入脚本、失败收据、订正脚本与成功收据分别保留。
