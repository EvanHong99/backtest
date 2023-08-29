# -*- coding=utf-8 -*-
# @File     : batch_calc_features.py
# @Time     : 2023/7/28 14:54
# @Author   : EvanHong
# @Email    : 939778128@qq.com
# @Project  : 2023.06.08超高频上证50指数计算
# @Description: 计算features并保存

import os
import sys

import pandas as pd

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

    f_list = defaultdict(list)
    for r, d, f in os.walk(data_root+'tick_data/'):
        for filename in f:
            if filename == 'placeholder': continue
            if 'clean_obh' not in filename: continue
            parts = filename.split('_')
            _date = str(pd.to_datetime(parts[0]).date())
            stk_name = parts[1]
            if config.complete_status['features'].get(stk_name) is not None:
                if _date in config.complete_status['features'].get(stk_name):
                    continue
            f_list[stk_name].append(_date)
            f_list[stk_name] = sorted(list(set(f_list[stk_name])))

    for stk_name in f_list.keys():
        if stk_name not in list(code_dict.keys()): continue
        if stk_name in config.exclude: continue
        # if stk_name in config.complete_status['features']: continue
        for _date in f_list[stk_name]:
            _date=_date.replace('-','')
            yyyy = _date[:4]
            mm = _date[4:6]
            dd = _date[-2:]
            update_date(yyyy, mm, dd)
            print('start', stk_name, yyyy, mm, dd)
            datafeed = LobDataFeed()
            # strategy = LobStrategy(max_close_timedelta=timedelta(minutes=int(freq[:-3]) * pred_n_steps))
            broker = Broker(cash=1e6, commission=1e-3)
            observer = LobObserver()

            self = LobBackTester(model_root=model_root,
                                 file_root=data_root+'tick_data/',
                                 dates=[],  # todo 确认一致性是否有bug
                                 stk_names=[],
                                 levels=5,
                                 target=Target.ret.name,
                                 freq=freq,
                                 pred_n_steps=pred_n_steps,
                                 use_n_steps=use_n_steps,
                                 drop_current=drop_current,
                                 datafeed=datafeed,
                                 strategy=None,
                                 broker=broker,
                                 observer=observer,
                                 statistics=None,
                                 )
            self.stk_name = stk_name

            self.alldata[config.date][stk_name] = self.load_data(file_root=self.file_root, date=config.date,
                                                                 stk_name=stk_name,load_obh=True,load_vol_tov=True,load_events=False)  # random freq
            self.alldatas[config.date][stk_name] = self.calc_features(
                self.alldata[config.date][stk_name], level=use_level, to_freq=min_freq)  # min_freq freq

            dp = AggDataPreprocessor()
            self.alldatas[config.date][stk_name] = [dp.agg_features(feature,use_events=False) for feature in
                                                    self.alldatas[config.date][stk_name]]

            for i, feature in enumerate(self.alldatas[config.date][stk_name]):
                feature.to_csv(data_root+'tick_data/' + FILE_FMT_feature.format(config.date, stk_name, str(i)))
            print('finish', stk_name, yyyy, mm, dd)
            if config.complete_status['features'].get(stk_name) is None:
                config.complete_status['features'][stk_name]=[config.date1]
            else:config.complete_status['features'][stk_name].append(config.date1)
            save_status(is_tick=True)

