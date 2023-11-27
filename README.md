# 超高频指数复现及策略回测

## 已完成

|     领域      | 已完成                                                                                                  | status |

[//]: # (|:-----------:|------------------------------------------------------------------------------------------------------|:------:|)
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
|    model    | ensemble models, lasso, rnn, automl                                                                  |   1    |
| backtest 	  | 方向正确率，**\* 要排除连续到达阈值，且连续盈利的情况**                                                                      |   1    |
| backtest 	  | 预测波动率为衍生品交易提供信息                                                                                      |   1    |
|    data     | 用lobster数据，复现                                                                                        |   -1   |
|    data     | 去掉开收盘5min                                                                                            |   1    |
|    model    | 过拟合[1](./sources/res/stat_贵州茅台_TabularPredictor_vol.csv) Train Data Rows:530,Train Data Columns: 260 |   1    |
|     eda     | y distribution, feature distribution                                                                 |   1    |
|    data     | 多拿些数据，去掉一些小幅波动的点，尽量去拟合大的波动                                                                           |   1    |


## TODO

|    领域    | todo                                                                                | status |

[//]: # (|:--------:|-------------------------------------------------------------------------------------|:------:|)
|   data   | 仅使用最大权重的5只股票                                                                        |   0    |
|   data   | 聚类以获得更多历史信息                                                                         |   0    |
|   data   | use polar                                                                           |   0    |
|   data   | 实现并行计算因子、scaler等任务                                                                  |   0    |
|  model   | auto-encoder: rnn encoder - many to one = one to many - decoder. VAE/Adversarial AE |   0    |
| backtest | 如果胜率可以，就跑回测净值曲线                                                                     |   0    |
| backtest | 优化框架以适应日间低频交易                                                                       |   0    |
| backtest | 不同的interval（如time、num of trades、volume                                              |   0    |
| backtest | 如果获取到的数据有延迟，或者提前获取到超短期内幕消息（比如预测等），那么有多少的预测能力                                        |   0    |
| strategy | 动态择时close                                                                           |   0    |
| strategy | 止损止盈                                                                                |   0    |
|   task   | 看突变值                                                                                |   0    |
|   todo   | 转3s低频数据，量价，进行预测,预测10min，转为分类问题，10min波动率变化方向，到阈值才算波动的明显变化                            |   0    |
|   todo   | 历史上，每天半小时进行匹配，然后统计这半小时内的涨跌幅、开盘、交易量等等，可以发现开收盘半小时大概率（55%）是涨，隔夜大概率跳空，可以据此开发相关的策略       |   0    |
|   todo   | 为啥readme这么卡                                                                         |   0    |

基于什么开发策略，更关注precision。波动率的值不同，但只要方向正确就行
可视化波动率的预测结果
fbeta的binary到底是啥？自己写lossfunc，比如focalloss

在波动率预测，agg阶段可以使用不同的kernal，借鉴cv

真格量化 期权回测

大盘异动的时刻，权重股和券商股是否异动，其持续性怎么样，盘口怎么变


## Tricks

### skewness and kurtosis

![skewness and kurtosis](../res/preds/alleviate_skewness/y_distribution.png)

```python
def tar_weight(_y_train, bins=20):
    nrows = len(_y_train)
    hist, bin_edge = np.histogram(_y_train, bins=bins)
    hist[hist == 0] += 1  # 避免出现RuntimeWarning: divide by zero encountered in divide  bin_w = np.sqrt(nrows / hist)
    bin_idx = np.digitize(_y_train, bin_edge,
                          right=False) - 1  # series中元素是第几个bin，可能会导致溢出，因为有些值（比如0）刚好在bin边界上，所以要把这些归到第一个bin（即现在的bin_idx==1处）
    bin_idx[bin_idx == 0] += 1
    bin_idx -= 1
    bin_w = np.sqrt(nrows / hist)
    series_w = np.array([bin_w[i] for i in bin_idx])
    series_w = nrows / sum(series_w) * series_w
    series_w = pd.Series(series_w, index=_y_train.index, name='weight')
    return series_w
```

当无法读取文件时？或是别的错误？（已忘记是什么错了）
```python
import os
import sys
path = os.path.join(os.path.dirname(__file__), os.pardir)
sys.path.append(path)
```