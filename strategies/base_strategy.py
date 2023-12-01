# -*- coding=utf-8 -*-
# @File     : base_strategy.py
# @Time     : 2023/8/7 9:19
# @Author   : EvanHong
# @Email    : 939778128@qq.com
# @Project  : 2023.06.08超高频上证50指数计算
# @Description:

from abc import abstractmethod
from typing import Union

from backtest.signals.base_signal import BaseSingleAssetSignal,MultiAssetSignal



class BaseStrategy(object):
    """
    用于生成各种操作的信号，比如依据买入数量、平仓时间、平仓价格给出信号Signals
    """
    def __init__(self, *args, **kwargs):
        self.args = args
        for key, value in kwargs.items():
            self.__setattr__(key, value)

    @abstractmethod
    def load_data(self, *args, **kwargs):
        pass
    @abstractmethod
    def calc_side(self,pred_ret):
        pass

    @abstractmethod
    def calc_type_(self, pred_ret):
        pass


    @abstractmethod
    def calc_price_limit(self,pred_ret):
        pass


    @abstractmethod
    def calc_quantity(self, pred_ret):
        pass


    @abstractmethod
    def calc_close_time(self,pred_ret):
        pass

    @abstractmethod
    def generate_signals(self, *args, **kwargs)->Union[BaseSingleAssetSignal,MultiAssetSignal]:
        """
        输出需要统一范式，即(timestamp,side,type,price,volume)

        :return:
        """
        pass
