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
    hold = 0
    open = 1
    close = 2


class ColMapper(object):
    def __init__(self, side_ask='a', side_bid='b', target_volume='v', target_price='p', spot='current',
                 mid_price='mid_price'):
        """这里的参数side、level、target等应该向后兼容，即去匹配 `backtest.preprocessors.preprocess.LobFeatureEngineering`中的格式，其中会固定调用'a'/'b'/'v'/'p'/

        Parameters
        ----------
        side_ask :
        side_bid :
        target_volume :
        target_price :
        spot :
        mid_price :
        """
        self.side_ask = side_ask
        self.side_bid = side_bid
        self.target_volume = target_volume
        self.target_price = target_price
        self.spot = spot
        self.mid_price = mid_price

    def get_mapper(self):
        return {'a': self.side_ask,
                'b': self.side_bid,
                'p': self.target_price,
                'v': self.target_volume}


class ColTemplate_todo(ABC):
    """
    column template
    """

    def __check_side_mapper__(self, mapper: dict):
        """
        Deprecated
        由于编写了`ColMapper`类，因此不需要这里来检验是否符合框架要求的标准

        Parameters
        ----------
        mapper :

        Returns
        -------

        """
        keys = mapper.keys()
        assert len(set(keys).difference({'a', 'b'})) == 0

    def __check_target_mapper__(self, mapper: dict):
        """
        Deprecated
        由于编写了`ColMapper`类，因此不需要这里来检验是否符合框架要求的标准

        Parameters
        ----------
        mapper :

        Returns
        -------

        """
        keys = mapper.keys()
        assert len(set(keys).difference({'v', 'p'})) == 0

    def __init__(self, side: str = None, level: int = None, target: str = None, spot='current', mid_price='mid_price',
                 mapper: ColMapper = None, *args, **kwargs):
        """这里的参数side、level、target等应该向后兼容，即去匹配 `backtest.preprocessors.preprocess.LobFeatureEngineering`中的格式，其中会固定调用'a'/'b'/'v'/'p'/

        Parameters
        ----------
        side : str,
            'a' or 'b'
        level : int,
        target :
            'v' or 'p'
        spot :
            现货列名
        mid_price :
        args :
        kwargs :
        """
        self.side = side  # 'a'/'b'
        self.level = level
        self.target = target  # 'v' or 'p'
        self.current = spot
        self.spot = spot
        self.mid_price = mid_price
        self.mapper = mapper
        if mapper is None:
            raise ValueError("you should provide a col mapper")

    @abstractmethod
    def __str__(self):
        pass

    @abstractmethod
    def __repr__(self):
        pass


class ColTemplate(ABC):
    """
    column template
    """

    def __init__(self, side: str = None, level: int = None, target: str = None, spot='current', mid_price='mid_price',
                 *args, **kwargs):
        self.side = side  # 'a'/'b'
        self.level = level
        self.target = target  # 'v' or 'p'
        self.current = spot
        self.spot = spot
        self.mid_price = mid_price

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
        super().__init__(side, level, target)
        self.current = 'current'
        self.mid_price = 'mid_price'
        self.mapper = {'a': 'Buy',
                       'b': 'Sell',
                       'p': 'Price',
                       'v': 'Volume'}

    def __str__(self):
        return "{:s}{:s}{:02d}".format(self.mapper[self.side], self.mapper[self.target], self.level)

    @overrides
    def __repr__(self):
        return self.__str__()


class LobColTemplate(ColTemplate):
    """
    column template for 海通. 数据源自于ricequant
    """

    def __init__(self, side: str = None, level: int = None, target: str = None, spot='current', mid_price='mid_price'):
        super().__init__(side=side, level=level, target=target, spot=spot, mid_price=mid_price)

    def __str__(self):
        return f"{self.side}{self.level}_{self.target}"

    def __repr__(self):
        return self.__str__()


class CITICSF_ColTemplate(ColTemplate):
    """
    column template for 中信中证资本
    """

    def __init__(self, side: str = None, level: int = None, target: str = None, spot='current', mid_price='mid_price'):
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
        super().__init__(side, level, target, spot=spot, mid_price=mid_price)
        self.mapper = {'a': 'ask',
                       'b': 'bid',
                       }
        self.mapper1 = {'a': 'a',
                        'b': 'b',
                        }

    def __str__(self):
        if self.target == 'p':
            return "{:s}{:01d}".format(self.mapper[self.side], self.level)
        elif self.target == 'v':
            return "{:s}size{:01d}".format(self.mapper1[self.side], self.level)

    @overrides
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
