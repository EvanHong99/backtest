# -*- coding=utf-8 -*-
# @Project  : caitong_securities
# @FilePath : 2023.10.23股指日内短期预测/backtest/predefined
# @File     : macros.py
# @Time     : 2023/11/27 13:22
# @Author   : EvanHong
# @Email    : 939778128@qq.com
# @Description:
import datetime
import os.path
import re
from enum import Enum

import h5py
import hdf5plugin
import numpy as np
import pandas as pd
from datetime import timedelta
import pickle
import json
import logging
from typing import Union, Optional
from abc import ABC, abstractmethod
from overrides import overrides


# ========================================= data =================================================
class OrderSideInt(Enum):
    bid = 66  # 买
    ask = 83  # 卖


class OrderTypeInt(Enum):
    """
    上交所 etf期权 委托类型 = [普通限价委托、市价剩余转限价委托、市价剩余撤销委托、全额即时限价委托、全额即时市价委托以及业务规则规定的其他委托类型]
    """
    limit = 76  # 限价单
    market = 77  # 市价单
    best_own = 85  # 本⽅最优
    bop = 85  # 本⽅最优


class TypeAction(Enum):
    """
    strategy给出的signal所属操作
    """
    hold=0
    open=1
    close=2

class ColTemplate(ABC):
    """
    column template
    """

    def __init__(self, side: str = None, level: int = None, target: str = None):
        self.side = side
        self.level = level
        self.target = target
        self.current = None
        self.mid_price = None

    @abstractmethod
    def __str__(self):
        pass

    @abstractmethod
    def __repr__(self):
        pass

class RQLobTemplateCaitong(ColTemplate):
    """
    column template for 财通. 数据源自于ricequant
    """

    def __init__(self, side: str = None, level: int = None, target: str = None):
        """

        Parameters
        ----------
        side : str
            `a` for ask `b` for bid
        level : int
            1~5
        target : str
            `p`:`Price`,`v`:`Volume`
        """
        super().__init__(side,level,target)
        self.current = 'current'
        self.mid_price = 'mid_price'
        self.mapper={'a':'Buy',
                     'b':'Sell',
                     'p':'Price',
                     'v':'Volume'}

    def __str__(self):
        return "{:s}{:s}{:02d}".format(self.mapper[self.side],self.mapper[self.target],self.level)

    @overrides
    def __repr__(self):
        return self.__str__()

class LobColTemplate(object):
    """
    column template for 海通. 数据源自于ricequant
    """

    def __init__(self, side: str = None, level: int = None, target: str = None):
        self.side = side
        self.level = level
        self.target = target
        self.spot = 'current'
        self.mid_price = 'mid_price'

    def __str__(self):
        return f"{self.side}{self.level}_{self.target}"

    def __repr__(self):
        return self.__str__()


# ========================================= data =================================================


# ========================================= train model =================================================
class Target(Enum):
    """
    Attributes
    ----------
    ret:
        current ret
    mid_p_ret:
        中间价的ret
    vol:
        middle price realized volatility
    wap_ret:
        wap ret
    wap_vol:
        wap realized volatility
    """
    ret = 0  # current ret
    mid_p_ret = 1  # 预测中间价的ret
    vol = 2  # middle price realized volatility
    wap_ret = 3
    wap_vol = 4
    vol_chg = 5  # 波动率的变化方向
# ========================================= train model =================================================
