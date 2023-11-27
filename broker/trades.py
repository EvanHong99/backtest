# -*- coding=utf-8 -*-
# @Project  : caitong_securities
# @FilePath : 2023.10.23股指日内短期预测/backtest/trades
# @File     : trades.py
# @Time     : 2023/11/26 21:35
# @Author   : EvanHong
# @Email    : 939778128@qq.com
# @Description:


class Trade(object):
    def __init__(self,
                 seq,
                 bid_seq,
                 ask_seq,
                 quantity,
                 price,
                 timestamp):
        self.seq = seq
        self.bid_seq = bid_seq
        self.ask_seq = ask_seq
        self.quantity = quantity
        self.price = price
        self.timestamp = timestamp

    def __str__(self):
        return str(self.__dict__)