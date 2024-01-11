# -*- coding=utf-8 -*-
# @File     : support.py
# @Time     : 2023/8/2 19:05
# @Author   : EvanHong
# @Email    : 939778128@qq.com
# @Project  : 2023.06.08超高频上证50指数计算
# @Description:
import datetime
import os.path
import re
from enum import Enum

import numpy as np
import pandas as pd
from datetime import timedelta
import pickle
import json
import logging
from typing import Union, Optional, List
from abc import ABC, abstractmethod
from backtest.predefined.macros import OrderSideInt, OrderTypeInt, Target, ColTemplate, LobColTemplate, \
    RQLobTemplateCaitong


def create_dirs(dir_list: list):
    for path in dir_list:
        if not os.path.exists(path):
            os.mkdir(path)


def fill_zero(symbol: str):
    """可能从csv中读的数据symbol并不是读成str，那就会导致需要补0。比如海通买的每日更新的权重数据

    Parameters
    ----------
    symbol

    Returns
    -------

    """
    if len(symbol) < 6:
        return '0' * (6 - len(symbol)) + symbol
    else:
        return symbol


def get_order_details(data_root, date, symbol):
    """
    通过h5文件提取出对应symbol的数据
    :param order_f:
    :param symbol:
    :return:
    """
    import h5py
    import hdf5plugin
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
    import h5py
    import hdf5plugin
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


def update_date(yyyy: Union[str, int] = None, mm: Union[str, int] = None, dd: Union[str, int] = None):
    from backtest import config
    if yyyy is None:
        config.y, config.m, config.d, config.date, config.date1, config.start, config.end, config.important_times, config.ranges = None, None, None, None, None, None, None, None, None
        return config.y, config.m, config.d, config.date, config.date1, config.start, config.end, config.important_times, config.ranges

    yyyy = str(yyyy)
    mm = str(mm)
    dd = str(dd)
    mm = '0' + mm if len(mm) == 1 else mm
    dd = '0' + dd if len(dd) == 1 else dd

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

    # start=config.important_times['continues_auction_am_start']
    # end=config.important_times['continues_auction_pm_end']
    # if config.strip_time is not None:
    #     start+=config.strip_timedelta
    #     end=min(end,config.important_times['close_call_auction_end']-config.strip_timedelta)
    # config.ranges = [(start,
    #                   pd.to_datetime(f'{config.date1} 10:30:00.000000') - config.agg_timedelta),
    #                  (pd.to_datetime(f'{config.date1} 10:30:00.000000'),
    #                   pd.to_datetime(f'{config.date1} 11:30:00.000000') - config.agg_timedelta),
    #                  (pd.to_datetime(f'{config.date1} 13:00:00.000000'),
    #                   pd.to_datetime(f'{config.date1} 14:00:00.000000') - config.agg_timedelta),
    #                  (pd.to_datetime(f'{config.date1} 14:00:00.000000'),
    #                   end)]

    start = pd.to_datetime(f'{config.date1} 09:30:00.000000')
    end = pd.to_datetime(f'{config.date1} 14:57:00.000000')
    if config.strip_time is not None:
        start += config.strip_timedelta
        end = min(end, pd.to_datetime(f'{config.date1} 15:00:00.000000') - config.strip_timedelta)
    config.ranges = [(start,
                      pd.to_datetime(f'{config.date1} 10:30:00.000000')),
                     (pd.to_datetime(f'{config.date1} 10:30:00.000000'),
                      pd.to_datetime(f'{config.date1} 11:30:00.000000')),
                     (pd.to_datetime(f'{config.date1} 13:00:00.000000'),
                      pd.to_datetime(f'{config.date1} 14:00:00.000000')),
                     (pd.to_datetime(f'{config.date1} 14:00:00.000000'),
                      end)]

    return config.y, config.m, config.d, config.date, config.date1, config.start, config.end, config.important_times, config.ranges


def save_model(dir, filename, model):
    with open(dir + filename, 'wb') as f:
        pickle.dump(model, f, pickle.HIGHEST_PROTOCOL)


def get_model_name(model) -> str:
    return re.findall('\w*', str(type(model)).split('.')[-1])[0]


def __check_obedience(d: dict):
    try:
        assert len(set(d['models']) - set(d['features'])) == 0
        assert len(set(d['features']) - set(d['orderbooks'])) == 0
    except AssertionError as e:
        logging.warning(f"set(d['models'])-set(d['features']) {set(d['models']) - set(d['features'])}")
        logging.warning(f"set(d['features'])-set(d['orderbooks']) {set(d['features']) - set(d['orderbooks'])}")


def load_status(is_tick=False):
    import config
    path = 'backtest/complete_status.json'
    if is_tick:
        path = 'backtest/complete_status_tick.json'
    with open(config.root + path, 'r', encoding='utf8') as fr:
        config.complete_status = json.load(fr)
    print("load_status", config.complete_status)
    __check_obedience(config.complete_status)


def save_status(is_tick=False):
    import config
    path = 'backtest/complete_status.json'
    if is_tick:
        path = 'backtest/complete_status_tick.json'
    with open(config.root + path, 'w', encoding='utf8') as fw:
        json.dump(config.complete_status, fw, ensure_ascii=False, indent=4)
    # print("save_status", config.complete_status)
    # __check_obedience(config.complete_status)


def realized_volatility(series):
    return np.sqrt(np.sum(series ** 2))


def save_concat_data(data_dict, name, mid_path):
    import backtest.config as config
    if not mid_path.endswith('/'):
        mid_path += '/'
    if not os.path.exists(config.data_root + mid_path):
        os.mkdir(config.data_root + mid_path)
    with open(config.data_root + mid_path + f"{name}.pkl", 'wb') as fw:
        pickle.dump(data_dict, fw, pickle.HIGHEST_PROTOCOL)


def load_concat_data(name, mid_path):
    import pickle
    import backtest.config as config
    with open(config.data_root + mid_path + f"{name}.pkl", 'rb') as fr:
        data_dict = pickle.load(fr)
    return data_dict





def continuous2discrete(ret: pd.Series, drift=0, pos_threshold: Union[float, list] = 0.001,
                        neg_threshold: Union[float, list] = None, start_from_zero=False, name=None,
                        fill_na: Optional[Union[float, int]] = None):
    """
    将连续值转为离散的bin中


    Parameters
    ----------
    start_from_zero :
        生成的label是否从0开始，从而符合pytorch多分类的标签要求
    ret :
    pos_threshold :
    drift : 用于修正偏度至正态
    neg_threshold :
    name :
    fill_na :
        是否用fill_na填充nan。若为None，那么drop nan

    Returns
    -------
    array like, bin_idx, 即每个值所在的bin的编号

    Notes
    --------
    由于``np.digitize``是左闭右开，因此存在边界点会被分到左边的bin中，但是这种概率极小，因此在针对ret这种数字精度高且连续的值来说影响不大



    """
    if name is None and isinstance(ret, pd.Series):
        name = ret.name
    else:
        name = name

    # 先将格式进行统一，即将float的阈值转化为list类型
    if isinstance(pos_threshold, float):
        pos_threshold = [pos_threshold]
    if isinstance(neg_threshold, float):
        neg_threshold = [neg_threshold]
    if neg_threshold is None:
        neg_threshold = [-x for x in pos_threshold[::-1]]
    assert (np.array(pos_threshold) > 0).all() and (np.array(neg_threshold) < 0).all()

    if isinstance(pos_threshold, float):
        raise Exception("代码不应该能运行到该部分")
        if neg_threshold is None:
            neg_threshold = -pos_threshold
        assert pos_threshold >= 0 and neg_threshold <= 0
        index = ret.index
        if ret.isna().any():
            if fill_na is None:
                logging.warning("数据含nan")
                # ret=ret.dropna()
            else:
                ret = ret.fillna(fill_na)
        idx_isna = ret.isna()
        ret = np.where(np.logical_and(~idx_isna, ret > drift + pos_threshold), 1, ret)
        ret = np.where(np.logical_and(~idx_isna, ret < drift + neg_threshold), -1, ret)
        ret = np.where(
            np.logical_and(~idx_isna, np.logical_and(ret <= drift + pos_threshold, ret >= drift + neg_threshold)), 0,
            ret)
        ret = pd.Series(ret, index=index, name=name)
        if start_from_zero:
            ret += 1
        return ret
    elif isinstance(pos_threshold, list):
        bin_edge = [min(ret)] + neg_threshold + pos_threshold
        bin_edge = sorted(list(set(bin_edge)))  # 一定要有序
        num_classes = len(bin_edge)
        bin_idx = np.digitize(ret, bin_edge, right=False) - 1
        if start_from_zero:
            ...  # do nothing
        else:
            bin_idx -= num_classes // 2

        return bin_idx

    else:
        raise NotImplementedError()


def normalize_code(symbol_list: Union[list, int, str, List[int,], pd.Series]):
    if isinstance(symbol_list, (list, pd.Series)):
        res = []
        for symbol in symbol_list:
            res.append("{:06d}".format(symbol))
        return res
    else:
        raise NotImplementedError()


def to_bins(series: pd.Series, bins):
    """
    常用于将收益率序列划分到不同的label中

    Parameters
    ----------
    series :
    bins : int,

    Returns
    -------

    """
    logging.warning("若显式地定义了划分bin的值，请使用support.continuous2discrete",FutureWarning)
    assert bins % 2 == 1
    hist, bin_edge = np.histogram(series, bins=bins)
    hist[hist == 0] += 1  # 避免出现RuntimeWarning: divide by zero encountered in divide  bin_w = np.sqrt(nrows / hist)
    bin_idx = np.digitize(series, bin_edge,
                          right=False) - 1  # series中元素是第几个bin，可能会导致溢出，因为有些值（比如0）刚好在bin边界上，所以要把这些归到第一个bin（即现在的bin_idx==1处）
    bin_idx[bin_idx == 0] += 1
    bin_idx -= 1
    bin_value = list(range(-(bins // 2), bins // 2 + 1, 1))
    series_value = np.array([bin_value[i] for i in bin_idx])
    series_value = pd.Series(series_value, index=series.index, name=series.name)
    return series_value


if __name__ == '__main__':
    import config

    load_status()
    print(config.complete_status)
