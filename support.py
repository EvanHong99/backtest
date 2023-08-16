# -*- coding=utf-8 -*-
# @File     : support.py
# @Time     : 2023/8/2 19:05
# @Author   : EvanHong
# @Email    : 939778128@qq.com
# @Project  : 2023.06.08超高频上证50指数计算
# @Description:
import re
from enum import Enum

import h5py
import hdf5plugin
import numpy as np
import pandas as pd
from datetime import timedelta
import pickle
# import config

class OrderSideInt(Enum):
    bid = 66  # 买
    ask = 83  # 卖


class OrderTypeInt(Enum):
    limit = 76  # 限价单
    market = 77  # 市价单
    best_own = 85  # 本⽅最优
    bop = 85  # 本⽅最优

class Target(Enum):
    ret = 0  # ret
    mid_p_ret = 1  # 预测中间价的ret
    vol=2

class LobColTemplate(object):
    """
    column template
    """

    def __init__(self, side: str=None, level: int=None, target: str=None):
        self.side = side
        self.level = level
        self.target = target
        self.current='current'
        self.mid_price='mid_price'

    def __str__(self):
        return f"{self.side}{self.level}_{self.target}"

    def __repr__(self):
        return self.__str__()

def get_order_details(data_root, date, symbol):
    """
    通过h5文件提取出对应symbol的数据
    :param order_f:
    :param symbol:
    :return:
    """
    if symbol.startswith('6'):
        order_f = h5py.File(data_root + f'{date}.orders.XSHG.h5', 'r')
    else:
        order_f = h5py.File(data_root + f'{date}.orders.XSHE.h5', 'r')
    order_dset = order_f[symbol]
    order_details = pd.DataFrame.from_records(order_dset[:])
    order_details = order_details.set_index('seq', drop=False)
    order_details['timestamp'] = pd.to_datetime(order_details['timestamp'], format='%Y%m%d%H%M%S%f')
    order_details['last_traded_timestamp'] = pd.to_datetime(order_details['last_traded_timestamp'].replace(0, np.nan),
                                                            format='%Y%m%d%H%M%S%f')
    order_details['canceled_timestamp'] = pd.to_datetime(order_details['canceled_timestamp'].replace(0, np.nan),
                                                         format='%Y%m%d%H%M%S%f')
    order_details['price'] = order_details['price'] / 10000
    order_details['filled_amount'] = order_details['filled_amount'] / 10000

    temp = order_details.loc[order_details['type'] == OrderTypeInt.bop.value]
    if len(temp) > 0:
        # raise Exception('contents best of party (bop) price order')
        print('contents best of party (bop) price order')
    return order_details


def get_trade_details(data_root, date, symbol):
    """
    通过h5文件提取出对应symbol的数据
    :param trade_f:
    :param symbol:
    :return:
    """
    try:
        if symbol.startswith('6'):
            trade_f = h5py.File(data_root + f'{date}.trades.XSHG.h5', 'r')
        else:
            trade_f = h5py.File(data_root + f'{date}.trades.XSHE.h5', 'r')
        trade_dset = trade_f[symbol]
        trade_details = pd.DataFrame.from_records(trade_dset[:])
        trade_details = trade_details.set_index('seq', drop=False)
        trade_details['timestamp'] = pd.to_datetime(trade_details['timestamp'], format='%Y%m%d%H%M%S%f')
        trade_details['price'] = trade_details['price'] / 10000
        return trade_details
    except:
        print(f"KeyError: Unable to open object (object '{symbol}' doesn't exist)")
        return None

def update_date(yyyy: str, mm: str, dd: str):
    import config

    # global y
    # global m
    # global d
    # global date
    # global date1
    # global start
    # global end
    # global important_times
    # global ranges

    config.y = yyyy
    config.m = mm
    config.d = dd

    config.date = f'{config.y}{config.m}{config.d}'
    config.date1 = f'{config.y}-{config.m}-{config.d}'
    config.start = pd.to_datetime(f'{config.date1} 09:30:00')
    config.end = pd.to_datetime(f'{config.date1} 15:00:00.001')

    config.important_times = {
        'open_call_auction_start': pd.to_datetime(f'{config.date1} 09:15:00.000000'),
        'open_call_auction_end': pd.to_datetime(f'{config.date1} 09:25:00.000000'),
        'continues_auction_am_start': pd.to_datetime(f'{config.date1} 09:30:00.000000'),
        'continues_auction_am_end': pd.to_datetime(f'{config.date1} 11:30:00.000000'),
        'continues_auction_pm_start': pd.to_datetime(f'{config.date1} 13:00:00.000000'),
        'continues_auction_pm_end': pd.to_datetime(f'{config.date1} 14:57:00.000000'),
        'close_call_auction_start': pd.to_datetime(f'{config.date1} 14:57:00.000000'),
        'close_call_auction_end': pd.to_datetime(f'{config.date1} 15:00:00.000000'), }

    config.ranges = [(pd.to_datetime(f'{config.date1} 09:30:00.000'),
               pd.to_datetime(f'{config.date1} 10:30:00.000') - timedelta(milliseconds=10)),
              (pd.to_datetime(f'{config.date1} 10:30:00.000'),
               pd.to_datetime(f'{config.date1} 11:30:00.000') - timedelta(milliseconds=10)),
              (pd.to_datetime(f'{config.date1} 13:00:00.000'),
               pd.to_datetime(f'{config.date1} 14:00:00.000') - timedelta(milliseconds=10)),
              (pd.to_datetime(f'{config.date1} 14:00:00.000'),
               pd.to_datetime(f'{config.date1} 14:57:00.000') - timedelta(milliseconds=10))]
    # print(f"in function update to date {config.date} {config.date1}")
    return config.y,config.m,config.d,config.date,config.date1,config.start,config.end,config.important_times,config.ranges

def save_model(dir, filename, model):
    with open(dir + filename, 'wb') as f:
        pickle.dump(model, f, pickle.HIGHEST_PROTOCOL)

def extract_model_name(model)->str:
    return re.findall('\w*', str(type(model)).split('.')[-1])[0]

if __name__ == '__main__':
    import config

    for dd in [23,28,29]:
        update_date('2022','06',str(dd))
        print(f"update to date {config.date} {config.date1}")

