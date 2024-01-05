# -*- coding=utf-8 -*-
# @File     : statistics.py
# @Time     : 2023/8/2 14:59
# @Author   : EvanHong
# @Email    : 939778128@qq.com
# @Project  : 2023.06.08超高频上证50指数计算
# @Description:
import logging
from typing import Union

import pandas as pd
from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, r2_score, accuracy_score, \
    classification_report, confusion_matrix
import numpy as np

# import config
from backtest.predefined.macros import Target


class BaseStatistics(object):
    def __init__(self, *args, **kwargs):
        self.args = args
        for key, value in kwargs.items():
            self.__setattr__(key, value)


class ClassificationStatistics(BaseStatistics):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_class = -1

    @staticmethod
    def calc_precision(cm, limit=1):
        res = {}
        precision_template = "precision_{}"
        cum_precision_template = "cum_precision_{}"  # 累计precision，即当前label及更高阈值label所占所有预测之和
        num_class = cm.shape[0]
        assert num_class >= 2 * limit  # 至少多于2*limit个类别
        target_labels_pos = list(range(num_class))[-limit:]
        target_labels_neg = list(range(num_class))[:limit]
        for i in target_labels_neg:
            precision = sum(cm[:i + 1, i]) / sum(cm[:, i])
            res[precision_template.format(i)] = precision
        for i in target_labels_pos:
            precision = sum(cm[i:, i]) / sum(cm[:, i])
            res[precision_template.format(i)] = precision

        # 累计精度
        for i in target_labels_neg:
            cum_precision = sum(cm[:i + 1, :i+1].flatten()) / sum(cm[:, :i+1].flatten())
            res[cum_precision_template.format(i)] = cum_precision
        for i in target_labels_pos:
            cum_precision = sum(cm[i:, i:].flatten()) / sum(cm[:, i:].flatten())
            res[cum_precision_template.format(i)] = cum_precision

        return res

    @staticmethod
    def calc_recall(cm, limit=1):
        res = {}
        recall_template = "recall_{}"
        cum_recall_template = "cum_recall_{}"
        num_class = cm.shape[0]
        assert num_class >= 2 * limit  # 至少多于2*limit个类别
        target_labels_pos = list(range(num_class))[-limit:]
        target_labels_neg = list(range(num_class))[:limit]
        for i in target_labels_neg:
            recall = sum(cm[i, :i + 1]) / sum(cm[i, :])
            res[recall_template.format(i)] = recall
        for i in target_labels_pos:
            recall = sum(cm[i, i:]) / sum(cm[i, :])
            res[recall_template.format(i)] = recall

        # 累计recall
        for i in target_labels_neg:
            cum_recall = sum(cm[:i + 1, :i+1].flatten()) / sum(cm[:i+1, :].flatten())
            res[cum_recall_template.format(i)] = cum_recall
        for i in target_labels_pos:
            cum_recall = sum(cm[i:, i:].flatten()) / sum(cm[i:, :].flatten())
            res[cum_recall_template.format(i)] = cum_recall

        return res



    def calc_confusion_matrix(self, y_true, y_pred):
        """
        可用于多分类的confusion matrix
        Parameters
        ----------
        y_true : 
        y_pred : 

        Returns
        -------
        
        
        Examples
        --------
            ```python
                0,      1,      2
            0,	183,	252,	13
            1,	483,	2439,	507
            2,	125,	288,	6
            ```
            对于上述矩阵，分别生成precision、recall、opportunity cost矩阵


        """
        cm = confusion_matrix(y_true, y_pred)
        self.num_class = cm.shape[0]
        return cm

    @staticmethod
    def calc_huge_loss_rate(cm, limit=1) -> dict:
        """

        Parameters
        ----------
        cm :
        limit : int,
            最多limit个档位的最极端预测会被统计。例如limit=1，当前是9分类任务，则只有最极端的两个预测label（0和8）会被计算

        Returns
        -------
        res : dict,
            大损失概率

        """
        res = {}
        key_template = "huge_loss_rate_{}"
        num_class = cm.shape[0]
        assert num_class >= 2 * limit + 1  # 至少多于2*limit个类别
        target_labels_pos = list(range(num_class))[-limit:]
        target_labels_neg = list(range(num_class))[:limit]
        for i in target_labels_neg:
            huge_loss_rate = sum(cm[target_labels_pos, i]) / sum(cm[:, i])
            res[key_template.format(i)] = huge_loss_rate
        for i in target_labels_pos:
            huge_loss_rate = sum(cm[target_labels_neg, i]) / sum(cm[:, i])
            res[key_template.format(i)] = huge_loss_rate
        return res

    @staticmethod
    def calc_opportunity_cost(cm, limit=1):
        """机会成本
        .. math::
            opportunity\_cost=\\frac{预测不操作但真实情况应该操作的数量}{所有预测不操作的数量}

        Parameters
        ----------
        cm :
        limit : int,
            最多limit个档位的最极端预测之外的中庸信号会被统计。例如limit=1，当前是9分类任务，则只有最极端的两个预测label（0和8）之外的列会被计算作为no_action，而0和8行则作为应该采取操作的数量，从而计算机会成本

        Returns
        -------
        res : dict,
            机会成本

        """
        res = {}
        key_template = "opportunity_cost"
        num_class = cm.shape[0]
        assert num_class >= 2 * limit + 1  # 至少多于2*limit个类别

        labels_action = list(range(num_class))[-limit:] + list(range(num_class))[:limit]
        labels_no_action = list(range(num_class))[limit:-limit]
        if len(labels_no_action) == 0: raise ValueError()

        opportunity_cost = sum(cm[labels_action, :][:, labels_no_action].flatten()) / np.sum(
            cm[:, labels_no_action].flatten())
        res[key_template] = opportunity_cost
        return res

    def gen_stat(self, y_true, y_pred, limit=1):

        cm = self.calc_confusion_matrix(y_true, y_pred)
        res = {"cm": cm}

        precision = self.calc_precision(cm, limit=limit)
        recall = self.calc_recall(cm, limit=limit)
        huge_loss_rate = self.calc_huge_loss_rate(cm, limit=limit)
        opportunity_cost = self.calc_opportunity_cost(cm, limit=limit)

        res.update(precision)
        res.update(recall)
        res.update(huge_loss_rate)
        res.update(opportunity_cost)

        logging.warning("需要对连续值的statistics进行优化，加入y_true*y信号强度")

        return res


class LobStatistics(BaseStatistics):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def report(self, manager):
        pass  # Generate report based on records from observer

    @staticmethod
    def stat_pred_error(y_true, y_pred, name='stat', task='regression', target=None) -> Union[pd.Series, pd.DataFrame]:
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
        if task == 'regression':
            if target == Target.vol.name:
                y_true_mean = y_true.abs().mean()
                return pd.Series({'baseline_mae': np.mean(np.abs(y_true.abs() - y_true_mean)),
                                  'baseline_rmse': np.sqrt(np.square(y_true - y_true_mean).sum()),
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
        elif task == 'binary':
            res = classification_report(y_true, y_pred, output_dict=True)
            res['accuracy'] = {'f1-score': res['accuracy'], 'support': res['macro avg']['support']}
            res = pd.DataFrame.from_dict(res).T
            res.columns = pd.MultiIndex.from_tuples([(name, col) for col in res.columns])
            return res
        else:
            raise NotImplementedError

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
        assert eff_opera == win_times + fair_times + loss_times

        desc['eff_opera'] = eff_opera
        desc['win_times'] = win_times
        desc['fair_times'] = fair_times
        desc['loss_times'] = loss_times
        desc['eff_opera_ratio'] = eff_opera / max(len(signals), 1)
        desc['win_rate'] = win_times / max(eff_opera, 1)
        desc['fair_rate'] = fair_times / max(eff_opera, 1)
        desc['loss_rate'] = loss_times / max(eff_opera, 1)

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

    # @classmethod
    def stat_pred_performance(self):
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


class NetValueStatistics(BaseStatistics):
    def __init__(self, days_per_year=250, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.days_per_year = days_per_year

    def statistics(self, df: Union[pd.Series, pd.DataFrame], freq='daily', risk_free_rate=0.03):
        """

        Parameters
        ----------
        df :
            净值序列需要从1开始
        freq :
        risk_free_rate :

        Returns
        -------

        """
        df.index = pd.to_datetime(df.index)
        if isinstance(df, pd.Series):
            df = df.to_frame()
        print(f"净值序列长度：{len(df)}")

        res = pd.DataFrame()
        for col in df.columns:
            ser = df[col]

            # 总收益
            return_total = self.calc_total_return(ser)

            # 年化波动率
            volatility_annual = self.calc_annual_vol(ser, freq=freq)

            # 年化收益率
            return_annual = self.calc_annual_return(ser)

            # 夏普比率
            sharpeRatio = self.calc_sharpe(return_annual=return_annual,
                                           volatility_annual=volatility_annual,
                                           risk_free_rate=risk_free_rate,
                                           name=ser.name)

            # 最大回撤
            max_drawdown, max_drawdown_date = self.calc_max_drawdown(ser)

            # 汇总成表
            single_res = pd.Series(
                {'return_total': return_total, 'volatility_annual': volatility_annual, 'return_annual': return_annual,
                 'sharpeRatio': sharpeRatio, 'max_drawdown': max_drawdown, 'max_drawdown_date': max_drawdown_date},
                name=ser.name).T
            res = pd.concat([res, single_res], axis=1)

        return res

    def preprocess(self, ser: pd.Series):
        if ser.iloc[0, 0] != 1:
            logging.warning(f'net value series doesn\'t start from 1, {ser.head(3)}', ValueError)

    def calc_total_return(self, ser: pd.Series):
        return_total = ser.iloc[-1] - 1
        return return_total

    def calc_annual_return(self, ser: pd.Series):
        delta = ser.index[-1] - ser.index[0]
        seconds_per_year = self.days_per_year * 4 * 3600
        seconds_delta = delta.days * 4 * 3600 + delta.seconds
        years = seconds_delta / seconds_per_year

        return_annual = (ser.iloc[-1]) ** (1 / years) - 1
        return return_annual

    def calc_annual_vol(self, ser: pd.Series, freq):
        days_per_year = self.days_per_year
        # todo 自动频率识别
        # freq= ser.index[1] - ser.index[0]
        if freq == 'daily':
            volatility_annual = (ser / ser.shift(1) - 1).std() * np.sqrt(days_per_year)
        elif freq == 'monthly':
            volatility_annual = (ser / ser.shift(1) - 1).std() * np.sqrt(12)
        elif freq == 'minutes':
            volatility_annual = (ser / ser.shift(1) - 1).std() * np.sqrt(days_per_year * 4 * 60)
        elif freq == 'seconds':
            volatility_annual = (ser / ser.shift(1) - 1).std() * np.sqrt(days_per_year * 4 * 60 * 60)
        elif freq == '3s':
            volatility_annual = (ser / ser.shift(1) - 1).std() * np.sqrt(days_per_year * 4 * 60 * 20)
        else:
            raise NotImplementedError
        return volatility_annual

    def calc_sharpe(self, return_annual, volatility_annual, risk_free_rate, name):
        sharpe = (return_annual - risk_free_rate) / volatility_annual
        return sharpe

    def calc_max_drawdown(self, ser):
        temp = (np.maximum.accumulate(ser) - ser) / np.maximum.accumulate(ser)
        maxDrawdown = temp.max()
        maxDrawdown_date = pd.to_datetime(temp.idxmax())

        return maxDrawdown, maxDrawdown_date

def calc_r_squared(y_true,y_pred):
    y_mean=np.mean(y_true)
    SS_total=((y_true-y_mean)**2).sum()
    SS_res=((y_true-y_pred)**2).sum()
    return 1-SS_res/SS_total