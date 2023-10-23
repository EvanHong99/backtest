# -*- coding=utf-8 -*-
# @File     : statistics.py
# @Time     : 2023/8/2 14:59
# @Author   : EvanHong
# @Email    : 939778128@qq.com
# @Project  : 2023.06.08超高频上证50指数计算
# @Description:
from typing import Union

import pandas as pd
from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, r2_score,accuracy_score,classification_report
import numpy as np

# import config
from support import Target


class BaseStatistics(object):
    def __init__(self, *args, **kwargs):
        self.args = args
        for key, value in kwargs.items():
            self.__setattr__(key, value)


class LobStatistics(BaseStatistics):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def report(self, manager):
        pass  # Generate report based on records from observer

    @staticmethod
    def stat_pred_error(y_true, y_pred, name='stat',task='regression',target=None)->Union[pd.Series,pd.DataFrame]:
        """

        Parameters
        ----------
        y_true :
        y_pred :
        name :
        task :
        target :
            默认为 config.target，但需要手动填写到函数中去

        Returns
        -------

        """
        if task=='regression':
            if target==Target.vol.name:
                y_true_mean=y_true.abs().mean()
                return pd.Series({'baseline_mae': np.mean(np.abs(y_true.abs()-y_true_mean)),
                                  'baseline_rmse': np.sqrt(np.square(y_true-y_true_mean).sum()),
                                  'mae': mean_absolute_error(y_true, y_pred),
                                  'mse': mean_squared_error(y_true, y_pred),
                                  'rmse': mean_squared_error(y_true, y_pred, squared=False),
                                  'r2_score': r2_score(y_true, y_pred),
                                  'explained_variance_score': explained_variance_score(y_true, y_pred)},
                                 name=name)
            else:
                return pd.Series({'baseline_mae': y_true.abs().mean(),
                                  'baseline_rmse': np.sqrt(np.square(y_true).sum()),
                                  'mae': mean_absolute_error(y_true, y_pred),
                                  'mse': mean_squared_error(y_true, y_pred),
                                  'rmse': mean_squared_error(y_true, y_pred, squared=False),
                                  'r2_score': r2_score(y_true, y_pred),
                                  'explained_variance_score': explained_variance_score(y_true, y_pred)},
                                 name=name)
        elif task=='binary':
            res=classification_report(y_true,y_pred,output_dict=True)
            res['accuracy'] = {'f1-score': res['accuracy'], 'support': res['macro avg']['support']}
            res=pd.DataFrame.from_dict(res).T
            res.columns=pd.MultiIndex.from_tuples([(name,col) for col in res.columns])
            return res
        else: raise NotImplementedError


    @staticmethod
    def stat_winrate(ret: pd.Series, signals, counterpart: bool, params=None):
        """

        :param ret: True value
        :param signals: when to long or short
        :param counterpart: If data is calculated by counterparty price
        :param params:
        :return:
        """

        desc = ret.describe()
        eff_opera = (signals != 0).sum()
        win_times = (np.logical_and(signals != 0, ret.values > 0)).sum()
        fair_times = (np.logical_and(signals != 0, ret.values == 0)).sum()
        loss_times = (np.logical_and(signals != 0, ret.values < 0)).sum()
        assert eff_opera==win_times+fair_times+loss_times

        desc['eff_opera'] = eff_opera
        desc['win_times'] = win_times
        desc['fair_times'] = fair_times
        desc['loss_times'] = loss_times
        desc['eff_opera_ratio'] = eff_opera / max(len(signals),1)
        desc['win_rate'] = win_times / max(eff_opera , 1)
        desc['fair_rate'] = fair_times / max(eff_opera , 1)
        desc['loss_rate'] = loss_times / max(eff_opera , 1)

        desc['use_counterpart'] = str(counterpart)
        if params:
            desc = pd.concat([desc, pd.Series(params, name=desc.name)], axis=0)
        return desc

    @staticmethod
    def calc_net_ret(direction, ret_long, ret_short, name):
        net_ret = np.zeros_like(direction)
        net_ret = np.where(direction == 1, ret_long, net_ret)
        net_ret = np.where(direction == -1, ret_short, net_ret)
        net_ret = pd.Series(net_ret, index=direction.index, name=name)
        return net_ret

    @classmethod
    def stat_pred_performance(cls):
        """
        统计回测胜率、单次盈亏等
        Returns
        -------

        """
        all_stats = pd.DataFrame()
        all_stats_winrate = pd.DataFrame()
        for num, (X_test, y_test, model) in enumerate(zip(self.Xs, self.ys, self.models)):
            stat = LobStatistics.stat_pred_error(y_test, y_pred,
                                                 name="{}_period{}".format(
                                                     re.findall('\w*', str(type(model)).split('.')[-1])[0], num))
            all_stats = pd.concat([all_stats, stat], axis=1)

            tempdata = self.alldatas[num][['current', 'a1_p', 'b1_p']]
            temp = pd.concat(
                [tempdata.rename(columns={k: k + '_open' for k in tempdata.columns}).shift(self.pred_n_steps),
                 tempdata.rename(columns={k: k + '_close' for k in tempdata.columns})], axis=1)
            temp = pd.concat([temp, self.y_preds[num]], axis=1).dropna(how='any')
            temp['ret'] = np.log(temp['current_close'] / temp['current_open'])  # 用于看预测方向准确率
            temp['ret_long'] = np.log(
                temp['b1_p_close'] / temp['a1_p_open'])  # 做多时对手价的收益率（仅仅是固定时间平仓，且该时间为预测的期数pred_n_steps）
            temp['ret_short'] = np.log(temp['b1_p_open'] / temp['a1_p_close'])  # short时是open/close, -log(a/b)==log(b/a)
            self.temp = temp
            direction = temp[f'pred_{self.target}_{self.pred_n_steps * 0.2}s'].apply(
                lambda x: calc_signal(x, threshold=0.0008, shift=0))
            # net_ret赚了就是正
            net_ret = cls.calc_net_ret(direction, temp['ret_long'], temp['ret_short'], name=f'period{num}')
            stat = cls.stat_winrate(net_ret, direction, counterpart=True, params=self.param)  # 统计对手价下的胜率
            net_ret1 = cls.calc_net_ret(direction, temp['ret'], -temp['ret'], name=f'period{num}')
            stat1 = cls.stat_winrate(net_ret1, direction, counterpart=False, params=self.param)  # 统计方向预测准确率
            stat['drct_acc'] = stat1['win_rate']  # 预测方向准确率
            all_stats_winrate = pd.concat([all_stats_winrate, stat], axis=1)
            self.net_ret = net_ret
            self.net_ret1 = net_ret1

        print(all_stats)
        print(all_stats_winrate)
        return all_stats, all_stats_winrate
