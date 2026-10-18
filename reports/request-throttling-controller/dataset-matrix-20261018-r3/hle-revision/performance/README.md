# HLE Revision独立性能

文本200/图片118，各臂1预热+3正式交替；共16点、12正式/4预热排除，每遍零失败。逐遍actual output/vllm
duration、逐指标取三次正式中位数。文本395.722382→368.901514 tok/s（-6.777698%），图片285.588359→294.303169
tok/s（+3.051528%），正负全保留。max256/temp0/seed0/naturalEOS/rate inf/无客户端slot限制，不当完整解题成本或program节流因果。

私有归档SHA
ee559f53e9b98d5c4904ceee77796af1ded41f10b5fba13bb3f12565876f25a4。独立核验原始结果hash、同模态输入hash/题数、实际命令、actual
output除以vllm
duration和三正式中位数PASS。不同于Gold历史接口，本轮真实命令含--save-detailed，逐请求数据私有保留；本次延迟只核验原生aggregate，不宣称独立重算逐请求均值。公开无原始提示、生成文本、图片payload或地址。
