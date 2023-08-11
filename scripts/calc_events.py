# -*- coding=utf-8 -*-
# @File     : calc_events.py
# @Time     : 2023/8/10 15:08
# @Author   : EvanHong
# @Email    : 939778128@qq.com
# @Project  : 2023.06.08超高频上证50指数计算
# @Description:

from utils import OrderTypeInt, OrderSideInt, get_order_details, get_trade_details
import logging
from collections import defaultdict
from copy import deepcopy
from datetime import timedelta

import h5py
import hdf5plugin
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sortedcontainers import SortedDict
from tqdm import tqdm

from support import OrderTypeInt, OrderSideInt
from config import *
from preprocess import LobTimePreprocessor, LobCleanObhPreprocessor

import os


def calc_events(trade_details,order_details):
    """
    ATT和BTT相较于原文做了化简
    :param data_root:
    :param date:
    :param symbol:
    :return:
    """

    l = len(order_details)
    BLO = np.where(np.logical_and(order_details['side'] == OrderSideInt.bid.value,
                                  order_details['type'] == OrderTypeInt.limit.value), np.ones(l), np.zeros(l))
    ALO = np.where(np.logical_and(order_details['side'] == OrderSideInt.ask.value,
                                  order_details['type'] == OrderTypeInt.limit.value), np.ones(l), np.zeros(l))
    BMO = np.where(np.logical_and(order_details['side'] == OrderSideInt.bid.value,
                                  order_details['type'] == OrderTypeInt.market.value), np.ones(l), np.zeros(l))
    AMO = np.where(np.logical_and(order_details['side'] == OrderSideInt.ask.value,
                                  order_details['type'] == OrderTypeInt.market.value), np.ones(l), np.zeros(l))
    order_events = pd.DataFrame({'seq': order_details['seq'].values,
                                 'BLO': BLO,
                                 'ALO': ALO,
                                 'BMO': BMO,
                                 'AMO': AMO,
                                 'timestamp': order_details['timestamp']}).set_index('seq')

    trade_bid_orders = order_details.loc[trade_details['bid_seq']]
    trade_ask_orders = order_details.loc[trade_details['ask_seq']]
    l = len(trade_details)
    BTT = np.where(trade_bid_orders['quantity'].values >= trade_ask_orders['quantity'].values, np.ones(l), np.zeros(l))
    ATT = np.where(trade_ask_orders['quantity'].values >= trade_bid_orders['quantity'].values, np.ones(l), np.zeros(l))
    trade_events = pd.DataFrame({'seq': trade_details['seq'].values, 'BTT': BTT, 'ATT': ATT,
                                 'timestamp': trade_details['timestamp']}).set_index('seq')

    events = pd.concat([order_events, trade_events], axis=0).fillna(0).sort_index()
    events = events.set_index('timestamp').sort_index()
    events = LobTimePreprocessor.del_untrade_time(events.sort_index(), cut_tail=False)
    events = events.groupby(level=0).sum()
    return events


if __name__ == '__main__':
    for date in ["20220623","20220628","20220629"]:
        for stk_name in ["贵州茅台"]:
            trade_details = get_trade_details(data_root=data_root, date=date, symbol=code_dict[stk_name])
            order_details = get_order_details(data_root=data_root, date=date, symbol=code_dict[stk_name])

            events = calc_events(trade_details,order_details)
            events.to_csv(detail_data_root+f"{date}_{stk_name}_events.csv")
