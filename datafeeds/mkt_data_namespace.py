# -*- coding=utf-8 -*-
# @Project  : caitong_securities
# @FilePath : 2023.10.23股指日内短期预测/backtest/datafeeds
# @File     : mkt_data_namespace.py
# @Time     : 2023/11/22 13:16
# @Author   : EvanHong
# @Email    : 939778128@qq.com
# @Description:
from typing import Union


class SampleClass(object):
    """
    sample class for properties and setters
    """

    @property
    def width(self):
        return self._width

    @width.setter
    def width(self, value):
        self._width = value

    @property
    def height(self):
        return self._height

    @height.setter
    def height(self, value):
        self._height = value

    @property
    def resolution(self):
        return self._height * self._width


class BaseMktDataNamespace(object):

    def __init__(self, datetime: Union[str, int] = None, *args, **kwargs):
        self.datetime = datetime
        for k, v in kwargs.items():
            self.__setattr__(k, v)


class PandasOHLCMktDataNamespace(BaseMktDataNamespace):
    """
    用于标识pandas dataframe data的列名空间。仅使用close等粗略的价格数据
    """

    def __init__(self,
                 datetime: Union[str, int] = None,
                 date: Union[str, int] = None,
                 symbol: Union[str, int] = None,
                 open_: Union[str, int] = None,
                 high: Union[str, int] = None,
                 low: Union[str, int] = None,
                 close: Union[str, int] = None,
                 volume: Union[str, int] = None,
                 turnover: Union[str, int] = None,
                 *args, **kwargs):
        super().__init__(datetime,*args, **kwargs)
        self.date=date
        self.symbol=symbol
        self.open_ = open_
        self.high = high
        self.low = low
        self.close = close
        self.volume = volume
        self.turnover = turnover


class PandasLobMktDataNamespace(BaseMktDataNamespace):
    """
    用订单簿价格作为回测数据，以此来衡量是否能成交到
    a5_p	a4_p	a3_p	a2_p	a1_p	b1_p	b2_p	b3_p	b4_p	b5_p	a5_v	a4_v	a3_v	a2_v	a1_v	b1_v	b2_v	b3_v	b4_v	b5_v	current
    """
    def __init__(self,
                 datetime: Union[str, int] = None,
                 *args, **kwargs):
        super().__init__(datetime,*args, **kwargs)



if __name__ == '__main__':
    namespace = PandasOHLCMktDataNamespace(abc=123)
    print(namespace.datetime)
    namespace.datetime = 10
    print(namespace.datetime)
    print(str(namespace.abc))
    print(namespace.__dict__)
