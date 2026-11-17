# BidKV默认True MOD配对：v4冻结源

本目录是正式生成前冻结的25成员配置与自行编写执行源，不代表完成成绩。[source-manifest.json](source-manifest.json)的SHA为`2e28cbe231438325ba96038588ad23e2888a3d503a8bc220100a667a4475e434`。原始startup、模型启动命令、共享carrier身份固定；[初版路径修正记录](../v4-pre-generation-correction/README.md)保留，最终CPU路由检查不替代真实模型准入。

B0/B1共用官方vLLM/Ascend0.23基础、同SDK/模型/TP2与全部默认优化，仅关闭/开启BidKV策略。共同CPU拓扑修复来自[PR28固定提交b931a952](https://github.com/vLLM-HUST/vllm-hust-bidkv/commit/b931a952683eef1fdc6d9abc44a15c97a6f98905)；helper
SHA`fe00e47faa81a0a3713a368e79a53504290687f2f249e4bf693457c2f7341253`。原安装树和二进制不改、不重建算子、不更换SDK。[carrier-source-admission.json](carrier-source-admission.json)记录1264个installed文件，其中804个CANN资产和原始相对symlink。

新Native05/Bid04各自通过真实健康、实际CPU绑定/线程affinity和128个token ID检查后，按原60秒预热/300秒正式窗口完成两臂4K/16K
C8；独立复算原SSE和零复用，再执行Native全五、BidKV全五、同一新Native共同judge。冻结[完整计划](execution-plan.json)：每臂672
solver、另720计划judge输入，原任务、顺序、预算、EOS和评分器不变。C8固定长度合成资格与五项EOS推理分开。

受控环境执行入口为`python -B run_mod_v4.py --out /data/shared_upload/cybber695/bidkv-five-datasets-20261113-01/results/five-datasets-mod-04`。公开文件使用前将`.py.source`还原为`.py`，仍需合法访问受控原始输入与评分工具。生成必须等待公开源读回、真实模型准入与运行门闩，不因本目录存在而视为放行。

[Native04零生成失败](../native04-failure-v1/README.md)、[旧False中止残批](../aborted-cohort01/README.md)和旧缓存均保留；本轮使用全新namespace。生命周期父进程实际wait模型、C8/solver/judge子进程并保存返回码，fatal早停清理只处理自有进程。初始化/编译、主机/设备内存、策略加载/调用/触发与solver性能分别报告，0calls不能宣称BidKV机制收益。原问题、答案、对话、token
ID数组、完整SSE及私有工具受控保存。
