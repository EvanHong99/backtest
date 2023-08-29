# -*- coding=utf-8 -*-
# @File     : temp.py
# @Time     : 2023/8/29 10:27
# @Author   : EvanHong
# @Email    : 939778128@qq.com
# @Project  : 2023.06.08超高频上证50指数计算
# @Description: 将旧的status转为value为list的status

import os
import sys
path = os.path.join(os.path.dirname(__file__), os.pardir)
sys.path.append(path)

from collections import defaultdict
import config
from backtester import LobBackTester
from brokers.broker import Broker
from config import *
from datafeeds.datafeed import LobDataFeed
from observers.observer import LobObserver
from preprocessors.preprocess import AggDataPreprocessor
from strategies import LobStrategy
from support import *

if __name__ == '__main__':
    load_status(is_tick=True)

    # orderbooks
    stk_name_list = config.complete_status['orderbooks']
    temp = {}
    for k in stk_name_list:
        if k in ['AAPL', 'AMZN', 'GOOG', 'INTC', 'MSFT']:
            temp[k] = ["2012-06-21"]
        else:
            temp[k] = ["2022-06-23", "2022-06-28", "2022-06-29"]
    config.complete_status['orderbooks']=temp

    # features
    temp1 = defaultdict(list)
    for r, d, f in os.walk(r'D:\Work\INTERNSHIP\海通场内\2023.06.08超高频上证50指数计算\data\tick_data/'):
        for ff in f:
            if 'clean' not in ff: continue
            (date, stk_name, suffix, _) = ff.split('_')
            date = str(pd.to_datetime(date).date())
            temp1[stk_name].append(date)
    temp1

    temp.update(temp1)

    stk_name_list = config.complete_status['features']
    temp = {}
    for k in stk_name_list:
        if k in ['AAPL', 'AMZN', 'GOOG', 'INTC', 'MSFT']:
            temp[k] = ["2012-06-21"]
        else:
            temp[k] = ["2022-06-23", "2022-06-28", "2022-06-29"]
    temp
    counter = defaultdict(lambda: defaultdict(int))
    for r, d, f in os.walk(r'D:\Work\INTERNSHIP\海通场内\2023.06.08超高频上证50指数计算\data\tick_data/'):
        for ff in f:
            if 'feature' not in ff: continue
            (date, stk_name, suffix) = ff.split('_')
            date = str(pd.to_datetime(date).date())
            counter[stk_name][date] += 1

    temp1 = defaultdict(list)
    for stk_name in counter.keys():
        for d, v in counter[stk_name].items():
            if v == 4: temp1[stk_name].append(d)

    temp.update(temp1)
    config.complete_status['features']=temp
    save_status(is_tick=True)