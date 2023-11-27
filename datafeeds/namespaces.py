# -*- coding=utf-8 -*-
# @Project  : caitong_securities
# @FilePath : 2023.10.23股指日内短期预测/backtest/datafeeds
# @File     : namespaces.py
# @Time     : 2023/11/22 13:16
# @Author   : EvanHong
# @Email    : 939778128@qq.com
# @Description:

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
    
class BaseNamespace(object):
    
    def __init__(self,*args,**kwargs):
        self._datetime = None
        self._open = None
        self._high = None
        self._low = None
        self._close = None
        self._volume = None
        self._turnover = None
        for k,v in kwargs.items():
            self.__setattr__(k,v)

    @property
    def datetime(self):
        return self._datetime
    
    @datetime.setter
    def datetime(self,value):
        self._datetime=value

    @property
    def open(self):
        return self._open

    @open.setter
    def open(self, value):
        self._open = value

    @property
    def high(self):
        return self._high

    @high.setter
    def high(self, value):
        self._high = value

    @property
    def low(self):
        return self._low

    @low.setter
    def low(self, value):
        self._low = value

    @property
    def close(self):
        return self._close

    @close.setter
    def close(self, value):
        self._close = value

    @property
    def volume(self):
        return self._volume

    @volume.setter
    def volume(self, value):
        self._volume = value

    @property
    def turnover(self):
        return self._turnover

    @turnover.setter
    def turnover(self, value):
        self._turnover = value



class PandasNamespace(BaseNamespace):
    """
    用于标识pandas dataframe的列名空间
    """
    # attributes
    # datetime = None
    # open = None
    # high = None
    # low = None
    # close = None
    # volume = None
    # turnover = None
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)



if __name__ == '__main__':
    namespace=PandasNamespace(abc=123)
    print(namespace.datetime)
    namespace.datetime=10
    print(namespace.datetime)
    print(namespace.abc)
