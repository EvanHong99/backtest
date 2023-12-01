# -*- coding=utf-8 -*-
# @Project  : caitong_securities
# @FilePath : 2023.10.23股指日内短期预测/backtest/signals
# @File     : base_signal.py
# @Time     : 2023/11/24 14:19
# @Author   : EvanHong
# @Email    : 939778128@qq.com
# @Description: 简单化的交易信号，涉及到具体回测还是应该使用orders

from enum import Enum
from typing import Union


class TypeUnderlyingAsset(Enum):
    stock = 0
    etf = 1
    future=2
    option_vanilla=3
    option_toxic=4



class BaseSingleAssetSignal(object):
    """
    单个资产类型的信号，比如信号中仅有股票（可以是不同股票代码）
    应该去考虑我们买股票、买卖期货期权时涉及到的参数。比如市价限价、数量、时刻。
    signal生成放在strategies中。signals只是用来存储买卖信号、买卖多少.

    Notes
    -----
    signal和order不一样，signal用来批量指示系统进行交易。而具体的每一笔交易下单信息存储在order类中。
    signal主要由strategy来进行管理，orders则由broker进行管理。
    """

    def __init__(self, underlying_asset:Union[TypeUnderlyingAsset], *args, **kwargs):
        self.underlying_asset = underlying_asset
        self.args = args
        for k, v in kwargs.items():
            self.__setattr__(k, v)

    @property
    def datetime(self):
        return None

    @property
    def seq(self):
        return None

    @property
    def close_seq(self):
        """
        若某一行不为nan，那么说明该signal为close之前order的signal
        Returns
        -------

        """
        return None



class MultiDateSignal(object):
    """
    可以存储多日的signals
    """
    pass


class MultiAssetSignal(object):
    """
    可以存储多个资产的signals
    """
    pass