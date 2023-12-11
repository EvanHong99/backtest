# -*- coding=utf-8 -*-
# @Project  : caitong_securities
# @FilePath : 2023.10.23股指日内短期预测/backtest/signals
# @File     : signal_namespace.py
# @Time     : 2023/11/27 11:03
# @Author   : EvanHong
# @Email    : 939778128@qq.com
# @Description: 其实就是一个mapper，用来绑定磁盘上数据的列名和框架内部固定的调用方式
from collections import defaultdict
from typing import Union
from backtest.predefined.macros import OrderTypeInt,OrderSideInt


class BaseSignalNamespace(object):

    def __init__(self, datetime: Union[str, int] = None, *args, **kwargs):
        self.datetime = datetime
        for k, v in kwargs.items():
            self.__setattr__(k, v)
        self._value_mapper= defaultdict(dict)

    @property
    def col_mapper(self):
        return self.__dict__

    @property
    def value_mapper(self):
        return self._value_mapper

    @value_mapper.setter
    def value_mapper(self,kv_tuple):
        self._value_mapper[kv_tuple[0]]=kv_tuple[1]


    def get_value_mapper(self,col):
        """
        按照统一的的col，在self.col_mapper中获得对应的原数据列名
        Parameters
        ----------
        col :

        Returns
        -------

        """


class PandasSignalNamespace(BaseSignalNamespace):
    """
    用于标识位于磁盘的pandas dataframe signal的列名空间
    """

    def __init__(self, datetime, side, type_, quantity, price_limit=0, *args, **kwargs):
        """

        Parameters
        ----------
        datetime :
        side :
        type_ :
        quantity :
        price_limit : float, default 0
            0代表市价单
        """
        super().__init__(datetime=datetime, *args, **kwargs)
        # self.datetime = datetime
        self.side = side
        self.type_ = type_
        self.quantity = quantity
        self.price_limit = price_limit


if __name__ == '__main__':
    namespace = PandasSignalNamespace(abc=123)
    print(namespace.datetime)
    namespace.datetime = 10
    print(namespace.datetime)
    print(str(namespace.abc))
    print(namespace.__dict__)
