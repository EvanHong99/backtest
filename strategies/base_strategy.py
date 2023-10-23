# -*- coding=utf-8 -*-
# @File     : base_strategy.py
# @Time     : 2023/8/7 9:19
# @Author   : EvanHong
# @Email    : 939778128@qq.com
# @Project  : 2023.06.08超高频上证50指数计算
# @Description:

from abc import abstractmethod


class BaseStrategy(object):
    """
    用于生成各种操作的信号，比如依据买入数量、平仓时间、平仓价格给出信号Signals
    """
    def __init__(self, *args, **kwargs):
        self.args = args
        for key, value in kwargs.items():
            self.__setattr__(key, value)

    @abstractmethod
    def load_models(self, *args):
        pass

    @abstractmethod
    def load_data(self, *args):
        pass

    @abstractmethod
    def generate_signals(self, *args):
        """
        输出需要统一范式，即(timestamp,side,type,price,volume)
        :param args:
        :return:
        """
        pass
