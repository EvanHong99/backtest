# -*- coding=utf-8 -*-
# @Project  : caitong_securities
# @FilePath : 2023.10.23股指日内短期预测/backtest/strategies
# @File     : single_asset_strategy.py
# @Time     : 2023/11/29 10:56
# @Author   : EvanHong
# @Email    : 939778128@qq.com
# @Description:

from backtest.strategies.base_strategy import BaseStrategy
from backtest.predefined.macros import OrderTypeInt,TypeAction
from typing import Union

import pandas as pd
from backtest.signals.pandas_signal import PandasSignal
class SingleAssetStrategy(BaseStrategy):
    def __init__(self,max_close_timedelta):
        super().__init__()
        self.max_close_timedelta = max_close_timedelta
        self.threshold=None
        self.drift=None

    def calc_side(self,pred_ret):
        """

        :param pred_ret:
        :param threshold:
        :param drift: 用于修正偏度至正态
        :return:
        """
        if pred_ret > self.drift + self.threshold:
            return 1
        elif pred_ret < self.drift - self.threshold:
            return -1
        else:
            return 0


    def calc_type_(self, pred_ret):
        return OrderTypeInt.market.value


    def calc_price_limit(self,pred_ret):
        return 0


    def calc_quantity(self, pred_ret):
        # -1 代表全部买入
        return 100


    def calc_close_time(self, open_datetime):
        return open_datetime + self.max_close_timedelta


    def generate_signals(self, y_pred: pd.Series, stk_name: str,threshold=0.0008, drift=0,*args, **kwargs) ->Union[PandasSignal]:
        """
        
        Parameters
        ----------
        y_pred : pd.Series,
            应该是预测的收益率序列，以datetime为index
        stk_name : 
        threshold : 
        drift : 
        args : 
        kwargs : 

        Returns
        -------

        """
        self.threshold=threshold
        self.drift=drift


        # 计算开仓信号
        signals = y_pred.to_frame().agg([
            self.calc_side,
            self.calc_type_,
            self.calc_price_limit,
            self.calc_quantity
        ])
        signals.index = y_pred.index  # 不做这一步出来的index很怪，会有一个后缀
        signals.columns = ['side', 'type', 'price_limit', 'volume']
        signals.loc[:, 'stk_name'] = [stk_name] * len(signals)
        signals.loc[:, 'datetime'] = y_pred.index
        signals.loc[:, 'action'] = [TypeAction.open.name] * len(signals)
        signals.loc[:, 'seq'] = list(range(len(signals)))  # 平仓signal和开仓公用
        # todo 去除连续出现的信号

        # 计算平仓信号
        close_signals = signals.copy(deep=True)
        close_signals.loc[:, 'datetime'] = self.calc_close_time(signals['datetime'])
        close_signals.loc[:, 'action'] = [TypeAction.close.name] * len(close_signals)

        # 拼接数据
        signals = pd.concat([signals, close_signals], axis=0)
        signals = signals.set_index(['datetime', 'side'], drop=False).sort_index(ascending=True)  # 先卖后买
        # signals = y_pred.apply(lambda x: calc_side(x))
        # signals = signals.rename(str(y_pred.name).replace('pred', 'signal')).to_frame()
        # signals = LobTimePreprocessor.del_untrade_time(signals, cut_tail=True)
        return signals

    def generate_signals_percentile(self, y_pred: pd.Series, stk_name: str, percentile:Union[float,dict]=1, use_sign=True, *args, **kwargs) -> Union[
        PandasSignal,pd.DataFrame]:
        """

        Parameters
        ----------
        use_sign : bool
            区别不同符号的y_pred，用来分别生成做多做空信号
        percentile : Union[float, dict]
            位于y_pred中的多少percentile设为1
        y_pred : pd.Series,
            应该是预测的收益率序列，以datetime为index
        stk_name :
        threshold :
        drift :
        args :
        kwargs :

        Returns
        -------

        """
        def meta(y_pred):
            signals = y_pred.to_frame().agg([
                self.calc_side,
                self.calc_type_,
                self.calc_price_limit,
                self.calc_quantity
            ])
            signals.index = y_pred.index  # 不做这一步出来的index很怪，会有一个后缀
            signals.columns = ['side', 'type', 'price_limit', 'volume']
            signals.loc[:, 'stk_name'] = [stk_name] * len(signals)
            signals.loc[:, 'datetime'] = y_pred.index
            signals.loc[:, 'action'] = [TypeAction.open.name] * len(signals)
            signals.loc[:, 'seq'] = list(range(len(signals)))  # 平仓signal和开仓公用
            # todo 去除连续出现的信号

            # 计算平仓信号
            close_signals = signals.copy(deep=True)
            close_signals.loc[:, 'datetime'] = self.calc_close_time(signals['datetime'])
            close_signals.loc[:, 'action'] = [TypeAction.close.name] * len(close_signals)

            # 拼接数据
            signals = pd.concat([signals, close_signals], axis=0)
            signals = signals.set_index(['datetime', 'side'], drop=False).sort_index(ascending=True)  # 先卖后买
            # signals = y_pred.apply(lambda x: calc_side(x))
            # signals = signals.rename(str(y_pred.name).replace('pred', 'signal')).to_frame()
            # signals = LobTimePreprocessor.del_untrade_time(signals, cut_tail=True)
            return signals

        self.percentile=percentile
        self.drift=0
        if use_sign:
            # 这里忽略了y_pred==0的情况，默认这些时刻不生成信号
            y_pred_pos=y_pred.loc[(y_pred>0).values]
            y_pred_neg=y_pred.loc[(y_pred<0).values]
            pos_threshold=y_pred_pos.quantile(percentile)
            neg_threshold=y_pred_neg.quantile(1-percentile)

            # fixme: 计算正开仓信号，通过设置self.threshold来调整参数，不够优雅
            self.threshold=pos_threshold
            pos_signals=meta(y_pred_pos)
            # 计算负开仓信号
            self.threshold=-neg_threshold # neg_threshold是负数，需要加一个负号，因为calc_signal默认是根据正的threshold来计算1、0、-1
            neg_signals=meta(y_pred_neg)
            signals=pd.concat([pos_signals,neg_signals],axis=0).sort_index()

            return signals
        else: raise NotImplementedError()