# -*- coding=utf-8 -*-
# @File     : observer.py
# @Time     : 2023/8/2 12:23
# @Author   : EvanHong
# @Email    : 939778128@qq.com
# @Project  : 2023.06.08超高频上证50指数计算
# @Description:
import numpy as np
import pandas as pd


class BaseObserver(object):
    def __init__(self, *args, **kwargs):
        self.args = args
        for key, value in kwargs.items():
            self.__setattr__(key, value)


class LobObserver(BaseObserver):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def record(self, orders, datafeed, broker):
        pass  # Record orders, trades, and current portfolio status
