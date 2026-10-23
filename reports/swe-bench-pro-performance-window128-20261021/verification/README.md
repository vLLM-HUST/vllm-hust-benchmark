# 核账结论与request元信息路径

## 结论分级

客户端记录的正常返回调用及usage总量可从原件重算；不能据此确认Controller因果加速25%。原事件与trajectory来自同一次返回对象的两条保存路径，其一致性是持久化校验，不是两个独立外部计量系统的证明。此次另写脚本、不导入原分析或绘图代码，从归档逐条重算，并把未解决差异公开。没有新推理、mock或额外Agent。

| 核对项                       | Native | Controller |
| ---------------------------- | ------ | ---------- |
| 真实program数                | 128    | 128        |
| 客户端query开始数            | 1649   | 1826       |
| 1200秒内正常返回             | 1521   | 1698       |
| 正常返回的输出tokens         | 87494  | 109101     |
| 窗口末仍未返回               | 128    | 128        |
| 随后总预算异常               | 128    | 128        |
| 引擎stop计数器采样增量       | 1548   | 1726       |
| 引擎stop减客户端返回，未解释 | 27     | 28         |

引擎采样区间1199.653292/1198.922316秒，与精确客户端窗口并非完全相同；该差异在整臂排空后计数仍存在。没有跨层关联不能认定27/28都是本轮丢失回复，也不能排除额外未归属请求、生成后未交付或记录路径问题。当前只能说差异尚未核销，不能把两个计数口径混用。

整包35516733字节SHA与原回执相符，4235个文件逐个与归档及本地解包一致，4234个manifest条目一致（manifest自身不包含自己）。逐条按response.id匹配事件和已保存响应，内容SHA与usage相符，共3219正常返回；该操作仍不是独立外部计量。

两臂128对任务正文和初始两条消息逐字相同，cases和selection字节相同。实际设置只在run_id、API/control地址及scheduling_enabled四个字段不同，Agent预算/模型参数一致。以父batch-start
epoch重新切窗，与最早实际到达单调时钟的正常返回数相同。原始证据证明输入与配置控制，不能证明两臂后续生成内容相同。

仍未满足：单一NC顺序无重复；辅助日志观察器两臂启动不对称；引擎/调度器/程序状态顺序读取而非原子快照；引擎完成与客户端返回未逐请求闭合。因此不宣称稳定因果性能增益、同请求延迟改善或完整批次加速。

## 元信息不是没传；缺的是记录及映射

实际benchmark源码设置X-Session-ID=program_id，并在每次model.query设置X-Delivery-Request-ID=program_id:query-NNN。客户端保存这个ID、query起止、响应response.id、usage和内容SHA。此处纠正“缺少request
ID”可能造成的误解：逐请求ID已在发送代码中注入，不是只有本地自编序号而未传。

固定上游app从X-Session-ID提取program_id，调度日志及/programs实际出现本轮对应program，程序身份接入有效。转发器先remove_program_id，再执行client.post(url,json=payload)，没有转发来路headers。因此X-Delivery-Request-ID不会自动成为引擎request
ID；program_id也不应当作每次query的ID。

代理有可选delivery trace（EXPERIMENT_DELIVERY_TRACE=1），可以记录request_received、backend_body_received及ASGI
response
sends。本轮两臂实际两个代理日志增量中DELIVERY_TRACE均0，未获得这些逐请求记录；不能从“代码存在”推断“本轮已采集”。已收到回复的response.id可与客户端ID对应，但未收到回复的调用缺少代理记录的引擎响应ID，无法拿全局Prometheus计数器逐个核销。

下次需要保存的关系是：program_id + client request ID + 每次尝试序号 + engine response/request
ID；时间为代理收到、准入放行、后端响应体收到、ASGI发送开始/结束、客户端收到或异常。沿用现有ID和观测接口补齐，不能另造一套不兼容身份。完整逐请求引擎排队/首Token时间还需要引擎侧证据；非流式不能反推。

复核数据：[逐query CSV](query-records.csv)、[核账JSON](audit.json)、[输入配对哈希](input-pair-hashes.json)、[原记录SHA清单](source-records-sha256.json)。CSV不含题面、prompt、工具输出、补丁或地址。私有CSV另保留响应/内容哈希，归档原件不改写。
