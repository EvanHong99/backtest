# -*- coding=utf-8 -*-
# @Project  : caitong_securities
# @FilePath : 2023.10.23股指日内短期预测/backtest/orders
# @File     : orders.py
# @Time     : 2023/11/26 21:30
# @Author   : EvanHong
# @Email    : 939778128@qq.com
# @Description:

import os
import sys

path = os.path.join(os.path.dirname(__file__), os.pardir)
sys.path.append(path)

import logging
from copy import deepcopy
from typing import Union

import numpy as np
import pandas as pd
from tqdm import tqdm
from backtest.support import OrderTypeInt, OrderSideInt


class OrderTypeIntError(TypeError):
    def __init__(self, *arg):
        self.args = arg


class Order(object):
    def __init__(self, seq, timestamp, side, type_, quantity, price, filled_quantity=0, filled_amount=0,
                 last_traded_timestamp=pd.NaT, canceled_timestamp=pd.NaT, canceled_seq=0):
        """

        Parameters
        ----------
        seq :
        timestamp :
        side :
        type_ :
        quantity :
        price :
        filled_quantity :
        filled_amount :
        last_traded_timestamp :
        canceled_timestamp :
        canceled_seq :
        """
        self.seq = seq
        self.timestamp = timestamp
        self.side = side
        self.type = type_
        self.quantity = quantity
        self.price = price
        self.filled_quantity = filled_quantity
        self.can_filled = self.quantity - self.filled_quantity
        self.filled_amount = filled_amount
        self.last_traded_timestamp = last_traded_timestamp
        self.canceled_timestamp = canceled_timestamp
        self.canceled_seq = canceled_seq
        self.filled = False
        self.canceled = False

    def __str__(self):
        return "seq-{}, {}: {} @ {}. side={},type={},filled_quantity={},filled_amount={},\n last_traded_timestamp={},canceled_timestamp={},canceled_seq={}\n".format(
            self.seq, self.timestamp, self.quantity, self.price, OrderSideInt(self.side).name,
            OrderTypeInt(self.type).name, self.filled_quantity, self.filled_amount, self.last_traded_timestamp,
            self.canceled_timestamp, self.canceled_seq)

    def fill_quantity(self, timestamp, quantity, price):
        """
        全部/部分fill放在orderbook对象中处理，进而保证函数功能的原子性

        Parameters
        ----------
        timestamp :
        quantity :
        price :

        Returns
        -------

        """
        if self.filled:
            raise Exception('already filled ' + self.__str__())
        if self.canceled:
            raise Exception('already canceled ' + self.__str__())  # 限价单则限制价格
        #         if self.type==OrderTypeInt['limit'].value:
        #             assert self.price==price

        assert quantity <= self.can_filled
        self.last_traded_timestamp = timestamp
        self.filled_quantity += quantity
        self.filled_amount += price * quantity
        self.can_filled -= quantity
        # 市价单平均价格会变化
        if self.type == OrderTypeInt['market'].value:
            self.price = self.filled_amount / self.filled_quantity  # 平均价格
        self._set_filled()
        return quantity

    def _set_filled(self):
        if self.quantity == self.filled_quantity:
            self.filled = True

    def is_filled(self):
        if self.quantity == self.filled_quantity:
            self.filled = True
            return True
        else:
            return False

    # def can_filled(self):
    #     """quantity to fill"""
    #     return self.quantity - self.filled_quantity

    def cancel(self, canceled_seq, canceled_timestamp):
        self.canceled_seq = canceled_seq
        self.canceled_timestamp = canceled_timestamp
        self.canceled = True

    def to_dict(self):
        return self.__dict__
