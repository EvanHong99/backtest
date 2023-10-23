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
from preprocessors.preprocess import AggDataPreprocessor, LobTimePreprocessor
from strategies import LobStrategy
from support import *

if __name__ == '__main__':
    counter=0
    load_status(is_tick=True)
    mid_path=f'concat_tick_data_{config.target}_10min_10min/'

    # 全部文件
    dd = defaultdict(list)
    for r, d, f in os.walk(data_root + 'tick_data/'):
        for filename in f:
            if filename == 'placeholder': continue
            if 'clean_obh' not in filename: continue
            parts = filename.split('_')
            _date = str(pd.to_datetime(parts[0]).date())
            stk_name = parts[1]
            dd[stk_name].append(_date)
    # 去掉已经处理过的文件
    f_list = defaultdict(list)
    for stk_name in dd.keys():
        if config.complete_status['features'].get(stk_name) is not None:
            proc_dates=config.complete_status['features'].get(stk_name)
            # proc_dates=[]
            for _date in dd[stk_name]:
                if _date not in proc_dates:
                    f_list[stk_name].append(_date)
                    f_list[stk_name] = sorted(list(set(f_list[stk_name])))

    all_data_dict=defaultdict(dict)
    # 加载之前处理过的数据
    if os.path.exists(config.data_root+mid_path+'all_data_dict.pkl'):
        all_data_dict=load_concat_data('all_data_dict',mid_path)
        all_data_dict=defaultdict(dict, all_data_dict)
    for stk_name in f_list.keys():
        if stk_name not in list(code_dict.keys()): continue
        if stk_name in config.exclude: continue
        # if stk_name in config.complete_status['features']: continue
        for _date in f_list[stk_name]:
            _date = _date.replace('-', '')
            yyyy = _date[:4]
            mm = _date[4:6]
            dd = _date[-2:]
            update_date(yyyy, mm, dd)
            print('start', stk_name, yyyy, mm, dd)
            datafeed = LobDataFeed()
            # strategy = LobStrategy(max_close_timedelta=timedelta(minutes=int(freq[:-3]) * pred_n_steps))
            broker = Broker(cash=1e6, commission=1e-3)
            observer = LobObserver()
            dp = AggDataPreprocessor()
            tp = LobTimePreprocessor()
            self = LobBackTester(model_root=model_root,
                                 file_root=data_root + 'tick_data/',
                                 dates=[],  # 一致性
                                 stk_names=[],
                                 levels=5,
                                 target=config.target,
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
                                                                 stk_name=stk_name, load_obh=True, load_vol_tov=True,
                                                                 load_events=False)  # random freq
            self.alldatas[config.date][stk_name] = self.calc_features(
                self.alldata[config.date][stk_name], level=use_level, to_freq=min_freq)  # min_freq freq
            self.alldatas[config.date][stk_name] = [tp.del_untrade_time(feature,cut_tail=True,strip=config.strip_time) for feature in
                                                    self.alldatas[config.date][stk_name]]
            self.alldatas[config.date][stk_name] = [dp.agg_features(feature, use_events=False) for feature in
                                                    self.alldatas[config.date][stk_name]]
            all_data_dict[config.date][stk_name]=self.alldatas[config.date][stk_name]
            # 每个feature生成一个文件
            # for i, feature in enumerate(self.alldatas[config.date][stk_name]):
            #     feature.to_csv(data_root + 'tick_data/' + FILE_FMT_feature.format(config.date, stk_name, str(i)))
            print('finish', stk_name, yyyy, mm, dd)
            if config.complete_status['features'].get(stk_name) is None:
                config.complete_status['features'][stk_name] = [config.date1]
            else:
                config.complete_status['features'][stk_name].append(config.date1)
            save_status(is_tick=True)
            counter+=1
            if counter%100==0:
                # save
                save_concat_data(dict(all_data_dict), 'all_data_dict', mid_path=mid_path)
    # save
    save_concat_data(dict(all_data_dict),'all_data_dict',mid_path=mid_path)
