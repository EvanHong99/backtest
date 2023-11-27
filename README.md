# ����Ƶָ�����ּ����Իز�

## �����

|     ����      | �����                                                                                                  | status |

[//]: # (|:-----------:|------------------------------------------------------------------------------------------------------|:------:|)
|    data     | �������freq                                                                                             |   1    |
|    data	    | ��Ҫoverlap                                                                                            |   1    |
|    data	    | optiver�����ӣ�ȥ��ֵ����ô�������䣨Ԥ�����ʱ��Ͱ�����ʱ�����з���Ƭ����aggregate data                                              |   1    |
|    data     | ʱ������������data����Ԥ�����ڷŴ�0.2Ƶ�ʷŴ󣬶����Ʊһ��ѵ��������һ��                                                            |   1    |
|    model    | ����ʱ����з֣��ֱ�ѵ��scaler/model                                                                             |   1    |
|    model    | ��ͬmodel-automl                                                                                       |   1    |
|    model    | ��ֹ�����-cv                                                                                             |   1    |
|    model    | mljar����                                                                                              |   -1   |
|    model    | flaml����                                                                                              |   1    |
|    model    | autogluon����                                                                                          |   1    |
|    model    | rnn                                                                                                  |   1    |
| 	statistics | statisticsʤ�ʣ�ӯ���ȣ�ÿ��ӯ����                                                                               |   1    |
| 	statistics | ����IC,ICRank                                                                                          |   1    |
| statistics  | �����Ʊ��statisticsƽ������                                                                                  |   1    |
|  strategy   | truncateСԤ���ź�                                                                                        |   1    |
| backtest 	  | ��ֻ��Ʊ��ͨ��ģʽ                                                                                            |   1    |
|   urgent    | 1min�������ݣ�ÿ��240������                                                                                    |   1    |
|    data     | pca                                                                                                  |   -1   |
|    model    | ensemble models, lasso, rnn, automl                                                                  |   1    |
| backtest 	  | ������ȷ�ʣ�**\* Ҫ�ų�����������ֵ��������ӯ�������**                                                                      |   1    |
| backtest 	  | Ԥ�Ⲩ����Ϊ����Ʒ�����ṩ��Ϣ                                                                                      |   1    |
|    data     | ��lobster���ݣ�����                                                                                        |   -1   |
|    data     | ȥ��������5min                                                                                            |   1    |
|    model    | �����[1](./sources/res/stat_����ę́_TabularPredictor_vol.csv) Train Data Rows:530,Train Data Columns: 260 |   1    |
|     eda     | y distribution, feature distribution                                                                 |   1    |
|    data     | ����Щ���ݣ�ȥ��һЩС�������ĵ㣬����ȥ��ϴ�Ĳ���                                                                           |   1    |


## TODO

|    ����    | todo                                                                                | status |

[//]: # (|:--------:|-------------------------------------------------------------------------------------|:------:|)
|   data   | ��ʹ�����Ȩ�ص�5ֻ��Ʊ                                                                        |   0    |
|   data   | �����Ի�ø�����ʷ��Ϣ                                                                         |   0    |
|   data   | use polar                                                                           |   0    |
|   data   | ʵ�ֲ��м������ӡ�scaler������                                                                  |   0    |
|  model   | auto-encoder: rnn encoder - many to one = one to many - decoder. VAE/Adversarial AE |   0    |
| backtest | ���ʤ�ʿ��ԣ����ܻز⾻ֵ����                                                                     |   0    |
| backtest | �Ż��������Ӧ�ռ��Ƶ����                                                                       |   0    |
| backtest | ��ͬ��interval����time��num of trades��volume                                              |   0    |
| backtest | �����ȡ�����������ӳ٣�������ǰ��ȡ����������Ļ��Ϣ������Ԥ��ȣ�����ô�ж��ٵ�Ԥ������                                        |   0    |
| strategy | ��̬��ʱclose                                                                           |   0    |
| strategy | ֹ��ֹӯ                                                                                |   0    |
|   task   | ��ͻ��ֵ                                                                                |   0    |
|   todo   | ת3s��Ƶ���ݣ����ۣ�����Ԥ��,Ԥ��10min��תΪ�������⣬10min�����ʱ仯���򣬵���ֵ���㲨�������Ա仯                            |   0    |
|   todo   | ��ʷ�ϣ�ÿ���Сʱ����ƥ�䣬Ȼ��ͳ�����Сʱ�ڵ��ǵ��������̡��������ȵȣ����Է��ֿ����̰�Сʱ����ʣ�55%�����ǣ���ҹ��������գ����Ծݴ˿�����صĲ���       |   0    |
|   todo   | Ϊɶreadme��ô��                                                                         |   0    |

����ʲô�������ԣ�����עprecision�������ʵ�ֵ��ͬ����ֻҪ������ȷ����
���ӻ������ʵ�Ԥ����
fbeta��binary������ɶ���Լ�дlossfunc������focalloss

�ڲ�����Ԥ�⣬agg�׶ο���ʹ�ò�ͬ��kernal�����cv

������� ��Ȩ�ز�

�����춯��ʱ�̣�Ȩ�عɺ�ȯ�̹��Ƿ��춯�����������ô�����̿���ô��


## Tricks

### skewness and kurtosis

![skewness and kurtosis](../res/preds/alleviate_skewness/y_distribution.png)

```python
def tar_weight(_y_train, bins=20):
    nrows = len(_y_train)
    hist, bin_edge = np.histogram(_y_train, bins=bins)
    hist[hist == 0] += 1  # �������RuntimeWarning: divide by zero encountered in divide  bin_w = np.sqrt(nrows / hist)
    bin_idx = np.digitize(_y_train, bin_edge,
                          right=False) - 1  # series��Ԫ���ǵڼ���bin�����ܻᵼ���������Ϊ��Щֵ������0���պ���bin�߽��ϣ�����Ҫ����Щ�鵽��һ��bin�������ڵ�bin_idx==1����
    bin_idx[bin_idx == 0] += 1
    bin_idx -= 1
    bin_w = np.sqrt(nrows / hist)
    series_w = np.array([bin_w[i] for i in bin_idx])
    series_w = nrows / sum(series_w) * series_w
    series_w = pd.Series(series_w, index=_y_train.index, name='weight')
    return series_w
```

���޷���ȡ�ļ�ʱ�����Ǳ�Ĵ��󣿣���������ʲô���ˣ�
```python
import os
import sys
path = os.path.join(os.path.dirname(__file__), os.pardir)
sys.path.append(path)
```