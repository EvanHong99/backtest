# -*- coding=utf-8 -*-
# @Project  : caitong_securities
# @FilePath : 2023.10.23股指日内短期预测/backtest/IO/inputs
# @File     : inputs.py
# @Time     : 2023/11/27 15:42
# @Author   : EvanHong
# @Email    : 939778128@qq.com
# @Description: 全为功能函数，不存储数据，仅存储IO时的配置参数

import pandas as pd

def ReadCSV(path, indexCols=None, encoding='utf-8'):
    """
    审慎思考，最好可以参考pymaster，用class来包装功能函数
    Parameters
    ----------
    path :
    indexCols :
    encoding :

    Returns
    -------

    """
    '''自定义CSV文件读取'''
    if encoding is None:
        encoding = self.DataEncoding
    return pd.read_csv(
        path,
        index_col=indexCols,
        encoding=encoding,
        **self.ReadPara
    )