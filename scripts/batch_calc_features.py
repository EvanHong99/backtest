# -*- coding=utf-8 -*-
# @File     : batch_calc_features.py
# @Time     : 2023/7/28 14:54
# @Author   : EvanHong
# @Email    : 939778128@qq.com
# @Project  : 2023.06.08超高频上证50指数计算
# @Description: 计算features并保存


import os

from backtester import LobBackTester
from brokers.broker import Broker
from config import *
import config
from datafeeds.datafeed import LobDataFeed
from observers.observer import LobObserver
from preprocessors.preprocess import AggDataPreprocessor
from strategies import LobStrategy
from support import Target, update_date

if __name__ == '__main__':
    f_list=[]
    for r,d,f in os.walk(detail_data_root):
        print(f)
        for filename in f:
            if filename=='placeholder':continue
            parts=filename.split('_')
            yyyy=parts[0][:4]
            mm = parts[0][4:6]
            dd = parts[0][-2:]
            stk_name=parts[1]
            f_list.append((yyyy,mm,dd,stk_name))
    f_list=sorted(list(set(f_list)))


    for yyyy,mm,dd,stk_name in f_list:
        if stk_name not in list(code_dict.keys()): continue
        update_date(yyyy,mm, dd)
        print('start',stk_name,yyyy,mm,dd)
        datafeed = LobDataFeed()
        strategy = LobStrategy(max_close_timedelta=timedelta(minutes=int(freq[:-3]) * pred_n_steps))
        broker = Broker(cash=1e6, commission=1e-3)
        observer = LobObserver()

        self = LobBackTester(model_root=model_root,
                             file_root=detail_data_root,
                             dates=[],  # todo 确认一致性是否有bug
                             stk_names=[],
                             levels=5,
                             target=Target.ret.name,
                             freq=freq,
                             pred_n_steps=pred_n_steps,
                             use_n_steps=use_n_steps,
                             drop_current=drop_current,
                             datafeed=datafeed,
                             strategy=strategy,
                             broker=broker,
                             observer=observer,
                             statistics=None,
                             )
        self.stk_name = stk_name

        self.alldata[config.date][stk_name] = self.load_data(file_root=self.file_root, date=config.date,
                                                             stk_name=stk_name)  # random freq
        self.alldatas[config.date][stk_name] = self.preprocess_data(
            self.alldata[config.date][stk_name],level=use_level,to_freq=min_freq)  # min_freq

        dp = AggDataPreprocessor()
        # to agg_freq
        self.alldatas[config.date][stk_name] = [
            dp.agg_features(feature, agg_freq=agg_freq, pred_n_steps=pred_n_steps, use_n_steps=use_n_steps) for feature
            in self.alldatas[config.date][stk_name]]

        for i,feature in enumerate(self.alldatas[config.date][stk_name]):
            feature.to_csv(detail_data_root + FILE_FMT_feature.format(config.date,stk_name,str(i)))
        print('finish',stk_name,yyyy,mm,dd)
