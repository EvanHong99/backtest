# 超高频指数复现及策略回测

## TODO

|     领域      | todo                                                                                                 | status |
|:-----------:|------------------------------------------------------------------------------------------------------|:------:|
|    data     | 计算合理freq                                                                                             |   1    |
|    data	    | 不要overlap                                                                                            |   1    |
|    data	    | optiver找因子，去极值，怎么划分区间（预测多少时间就按多少时间来切分切片），aggregate data                                              |   1    |
|    data     | 时间拉长（多日data），预测周期放大，0.2频率放大，多个股票一起训练，多日一起                                                            |   1    |
|    model    | 日内时间段切分，分别训练scaler/model                                                                             |   1    |
|    model    | 不同model-automl                                                                                       |   1    |
|    model    | 防止过拟合-cv                                                                                             |   1    |
|    model    | mljar适配                                                                                              |   -1   |
|    model    | flaml适配                                                                                              |   1    |
|    model    | autogluon适配                                                                                          |   1    |
|    model    | rnn                                                                                                  |   1    |
| 	statistics | statistics胜率，盈亏比（每次盈亏）                                                                               |   1    |
| 	statistics | 因子IC,ICRank                                                                                          |   1    |
| statistics  | 多个股票上statistics平均表现                                                                                  |   1    |
|  strategy   | truncate小预测信号                                                                                        |   1    |
| backtest 	  | 多只股票的通用模式                                                                                            |   1    |
|   urgent    | 1min跳的数据，每天240条数据                                                                                    |   1    |
|    data     | pca                                                                                                  |   -1   |
|    data     | 聚类以获得更多历史信息                                                                                          |   0    |
|    data     | use polar                                                                                            |   0    |
|    model    | ensemble models, lasso, rnn, automl                                                                  |   1    |
|    model    | auto-encoder: rnn encoder - many to one = one to many - decoder. VAE/Adversarial AE                  |   0    |
| backtest 	  | 如果胜率可以，就跑回测净值曲线                                                                                      |   0    |
| backtest 	  | 方向正确率，**\* 要排除连续到达阈值，且连续盈利的情况**                                                                      |   1    |
| backtest 	  | 个股current预测足够准确，也是可以用到预测指数中（不同股票可以训练成一个model                                                         |   0    |
| backtest 	  | 预测波动率为衍生品交易提供信息                                                                                      |   1    |
|  backtest   | 优化框架以适应日间低频交易                                                                                        |   0    |
|  backtest   | 不同的interval（如time、num of trades、volume                                                               |   0    |
|  backtest   | 如果获取到的数据有延迟，或者提前获取到超短期内幕消息（比如预测等），那么有多少的预测能力                                                         |   0    |
|  backtest   | 动态择时close                                                                                            |   0    |
|  backtest   | 止损止盈                                                                                                 |   0    |
|  others  	  | 寻找历史上和今天相似的情况，指数，构建因子、聚类，先去看是否有相近时刻，是否预测的走势是相近的                                                      |   0    |
|    todo     | 用别人的数据，复现                                                                                            |   0    |
|    todo     | 转为分类任务                                                                                               |   0    |
|    todo     | 实现并行计算因子、scaler等任务                                                                                   |   0    |
|    todo     | 转3s低频数据，量价，进行预测,预测10min                                                                              |   0    |
|    todo     | shuffle后按比例划分                                                                                        |   0    |
|    data     | 去掉开收盘5min                                                                                            |   1    |
|    model    | 过拟合[1](./sources/res/stat_贵州茅台_TabularPredictor_vol.csv) Train Data Rows:530,Train Data Columns: 260 |   0    |
|    data     | 仅使用最大权重的5只股票                                                                                         |   0    |
|     eda     | y distribution, feature distribution                                                                 |   0    |
|    data     | 多拿些数据，去掉一些小幅波动的点，尽量去拟合大的波动                                                                           |   0    |
|  backtest   | 转为分类问题，10min波动率变化方向，到阈值才算波动的明显变化                                                                     |   0    |


## Tricks

### skewness and kurtosis

![skewness and kurtosis](../res/y_distribution.png)