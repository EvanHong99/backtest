# 超高频指数复现及策略回测



## TODO

|     领域      | todo                                                    | status  |
|:-----------:|---------------------------------------------------------|:-------:|
|    data     | 计算合理freq                                                |    1    |
|    data	    | 不要overlap                                               |    1    |
|    data	    | optiver找因子，去极值，怎么划分区间（预测多少时间就按多少时间来切分切片），aggregate data |    1    |
|    data     | 时间拉长（多日data），预测周期放大，0.2频率放大，多个股票一起训练，多日一起               |    1    |
|    data     | use polar                                               |    0    |
|    model    | 日内时间段切分，分别训练model                                       |    1    |
|    model    | 不同model-automl                                          |    1    |
|    model    | 防止过拟合-cv                                                |    1    |
|    model    | mljar适配                                                 |   -1    |
|    model    | flaml适配                                                 |    1    |
|    model    | rnn                                                     |    1    |
| 	statistics | statistics胜率，盈亏比（每次盈亏）                                  |    1    |
| 	statistics | 因子IC,ICRank                                             |    1    |
| 	statistics | mse的bug?                                                | 1，没有bug |
| statistics  | 多个股票上statistics平均表现                                     |    0    |
|  strategy   | truncate小预测信号                                           |    1    |
| backtest 	  | 如果胜率可以，就跑回测净值曲线                                         |    0    |
| backtest 	  | 方向正确率，**\* 要排除连续到达阈值，且连续盈利的情况**                         |    0    |
| backtest 	  | 多只股票的通用模式                                               |    0    |
| backtest 	  | 个股current预测足够准确，也是可以用到预测指数中（不同股票可以训练成一个model            |    0    |
| backtest 	  | 预测波动率为衍生品交易提供信息                                         |    0    |
|  backtest   | 优化框架以适应日间低频交易                                           |    0    |
|  backtest   | 不同的interval（如time、num of trades、volume                  |    0    |
|  backtest   | 如果获取到的数据有延迟，或者提前获取到超短期内幕消息（比如预测等），那么有多少的预测能力            |    0    |
|  backtest   | 动态择时close                                               |    0    |
|  backtest   | 止损止盈                                                    |    0    |
|  others  	  | 寻找历史上和今天相似的情况，指数，构建因子、聚类，先去看是否有相近时刻，是否预测的走势是相近的         |    0    |
|   urgent    | 1min跳的数据，每天240条数据                                       |    1    |
|     bug     | 低价格股票重构订单簿较慢                                            |    0    |
|    todo     | 用别人的数据                                                  |    0    |

# 学到的点

1. 代码/变量保持一致性，比如日期格式，如果都是`2022-06-28`就都保持，而非中途改用`20220628`，并且它的选择不是随意的，应该是偏好于`2022-06-28`，以方便地和datetime.date对接