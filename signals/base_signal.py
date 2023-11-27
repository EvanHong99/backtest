# -*- coding=utf-8 -*-
# @Project  : caitong_securities
# @FilePath : 2023.10.23股指日内短期预测/backtest/signals
# @File     : base_signal.py
# @Time     : 2023/11/24 14:19
# @Author   : EvanHong
# @Email    : 939778128@qq.com
# @Description: 简单化的交易信号，涉及到具体回测还是应该使用orders

from enum import Enum


class BaseSingleAssetSignal(object):
    def __init__(self,underlying_asset,amount,):
        self.underlying_asset=underlying_asset



# 应该去考虑我们买股票、买卖期货期权时涉及到的参数。比如市价限价、数量。这一部生成放在strategies中。signals只是用来存储买卖信号、买卖多少

