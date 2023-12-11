# -*- coding=utf-8 -*-
# @Project  : caitong_securities
# @FilePath : 2023.10.23股指日内短期预测/backtest/signals
# @File     : pandas_signal.py
# @Time     : 2023/11/27 10:36
# @Author   : EvanHong
# @Email    : 939778128@qq.com
# @Description:

import pandas as pd
import numpy as np
from typing import Union

from backtest.signals.base_signal import BaseSingleAssetSignal, TypeUnderlyingAsset
from backtest.signals.signal_namespace import PandasSignalNamespace
from backtest.predefined.macros import OrderTypeInt, OrderSideInt


class PandasSignal(BaseSingleAssetSignal):
    """
    将signal转成pandas类型
    """

    def __init__(self, underlying_asset: Union[TypeUnderlyingAsset] = TypeUnderlyingAsset.stock,
                 datetime: Union[str, int] = None,
                 side: Union[str, int] = None,
                 type_: Union[str, int] = None,
                 quantity: Union[str, int] = None,
                 price_limit: Union[str, int] = 0,
                 data=None,
                 *args, **kwargs):
        super().__init__(underlying_asset, *args, **kwargs)
        self.col_mapper = PandasSignalNamespace(datetime=datetime, side=side, type_=type_, quantity=quantity,
                                                price_limit=price_limit)
        if data is not None:
            self.load_data(data)
        else:
            self.data=None

    def get_col_value(self, df, col):
        if isinstance(col, int):
            if col==-1:return df.index
            return df.iloc[:, col]
        elif isinstance(col, str):
            return df.loc[:, col]

    def load_data(self, df):
        """

        Parameters
        ----------
        df : pd.DataFrame
            应该由IO将文件读取到内存，然后交由该类进行wrapping，处理成框架通用的调用方式

        Returns
        -------

        """

        self.data = df
        # self.datetime=self.get_col_value(df,self.col_mapper.datetime)
        # self.side=self.get_col_value(df,self.col_mapper.side)
        # self.type_=self.get_col_value(df,self.col_mapper.type_)
        # self.quantity=self.get_col_value(df,self.col_mapper.quantity)
        # self.price_limit=self.get_col_value(df,self.col_mapper.price_limit)

    @property
    def index(self):
        return self.get_col_value(self.data, self.col_mapper.datetime)

    @property
    def datetime(self):
        return self.get_col_value(self.data, self.col_mapper.datetime)

    @property
    def side(self):
        return self.get_col_value(self.data, self.col_mapper.side)

    @property
    def type_(self):
        return self.get_col_value(self.data, self.col_mapper.type_)


    @property
    def quantity(self):
        return self.get_col_value(self.data, self.col_mapper.quantity)

    @property
    def price_limit(self):
        return self.get_col_value(self.data, self.col_mapper.price_limit)


if __name__ == '__main__':
    print(PandasSignal())
