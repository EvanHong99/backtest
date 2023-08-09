# -*- coding=utf-8 -*-
# @File     : backtester.py
# @Time     : 2023/8/2 11:15
# @Author   : EvanHong
# @Email    : 939778128@qq.com
# @Project  : 2023.06.08超高频上证50指数计算
# @Description:

from collections import defaultdict
from copy import deepcopy

import numpy as np
import pandas as pd

from datafeed import BaseDataFeed
from strategies import BaseStrategy, LobStrategy
from broker import BaseBroker, Broker
from observer import BaseObserver, LobObserver
from statistics import BaseStatistics, LobStatistics
from typing import Callable, Dict, List, Union, Sequence, Tuple, Type, Union

from abc import abstractmethod

import re

import numpy as np

from statistics import LobStatistics
from preprocess import LobFeatureEngineering, LobTimePreprocessor, ShiftDataPreprocessor
from datafeed import LobDataFeed, LobModelFeed
from config import *
from strategies.base_strategy import BaseStrategy
from support import Target


class BaseTester(object):
    def __init__(self, *args, **kwargs):
        self.args = args
        for key, value in kwargs.items():
            self.__setattr__(key, value)

    @abstractmethod
    def run(self):
        pass


# class Screen(object):
#     @property
#     def width(self):
#         return self._width
#
#     @width.setter
#     def width(self, value):
#         self._width = value
#
#     @property
#     def height(self):
#         return self._height
#
#     @width.setter
#     def height(self, value):
#         self._height = value
#
#     @property
#     def resolution(self):
#         return self._height * self._width


class LobBackTester(BaseTester):
    """
    the whole project contains these files:
    1. backtester.py: the center and controler of the backtest project.
    2. ret.py: the main api to read/save ret from csv/xlsx files, and preprocess them for backtester. I have a lot of assets. And my ret frequency is 0.01s.
    3. strategies/base_strategy.py: strategies implementation and signal generation for backtester
    4. broker.py: including classes "Order", "Trade", "Broker" and other things you need.
    5. observer.py: recorder and logger for backtester
    6. statistics.py: result generator and visualization for backtester

    .. [#] Lean (Quantconnect): https://github.com/QuantConnect/Lean/blob/master/Documentation/2-Overview-Detailed-New.png
    .. [#] backtrader
    """

    def __init__(self,
                 model_root:str,
                 file_root:str,
                 dates:List[str],
                 stk_names:List[str],
                 levels:int,
                 target:str,
                 freq:str,
                 pred_n_steps:int,
                 use_n_steps:int,
                 drop_current=False,
                 datafeed: Union[LobDataFeed] = None,
                 strategy: Union[LobStrategy] = None,
                 broker: Union[Broker] = None,
                 observer: Union[LobObserver] = None,
                 statistics: Union[LobStatistics] = None,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.datafeed = datafeed
        self.strategy = strategy
        self.broker = broker
        self.observer = observer
        self.statistics = statistics

        self.clean_obh = None
        self.models = []
        self.model_root = model_root
        self.file_root = file_root
        self.dates = dates
        self.stk_names = stk_names
        self.levels = levels
        self.target = target
        self.freq = freq
        self.pred_n_steps = pred_n_steps
        self.use_n_steps = use_n_steps
        self.drop_current = drop_current
        self.param = {
            'drop_current': self.drop_current,
            'pred_n_steps': self.pred_n_steps,
            'target': self.target,
            'use_n_steps': self.use_n_steps,
        }
        
        self.alldata=defaultdict(dict) # {dates:{stk_name:ret}}
        self.alldatas=defaultdict(dict) # {dates:{stk_name:[data1,data2,...]}}
        self.all_signals=defaultdict(pd.DataFrame)

    def load_models(self, model_root, stk_name):
        model_loader = LobModelFeed(model_root=model_root, stk_name=stk_name)
        self.models = model_loader.models
        return self.models

    def load_data(self, file_root, date, stk_name) -> pd.DataFrame:
        """

        :param file_root:
        :param date:
        :param stk_name:
        :return: pd.DataFrame,(clean_obh_dict+vol_tov).asfreq(freq='10ms', method='ffill')
        """
        datafeed = LobDataFeed(file_root=file_root, date=date, stk_name=stk_name)
        self.clean_obh = datafeed.load_clean_obh(snapshot_window=self.levels)
        self.vol_tov = datafeed.load_vol_tov()
        data = pd.concat([self.clean_obh, self.vol_tov], axis=1).ffill()
        # 必须先将clean_obh填充到10ms，否则交易频率是完全不规律的，即可能我只想用5个frame的数据来预测，但很可能用上了十秒的信息
        data = data.asfreq(freq='10ms', method='ffill')
        return data

    def calc_features(self, df):
        # todo: 时间不连续、不规整，过于稀疏，归一化细节
        # LobTimePreprocessor.split_by_trade_period(self.alldata)
        fe = LobFeatureEngineering()
        feature = fe.generate(df)
        return feature

    # testit
    def preprocess_data(self,data)->list:
        """
        将数据划分为4份，每份一小时
        :return:
        """
        ltp = LobTimePreprocessor()
        # 必须先将数据切分，否则会导致11:30和13:00之间出现跳变
        alldatas = ltp.split_by_trade_period(data)
        alldatas = [ltp.add_head_tail(cobh, head_timestamp=pd.to_datetime(s),
                                           tail_timestamp=pd.to_datetime(e)) for cobh, (s, e) in
                         zip(alldatas, ranges)]
        # 不能对alldatas change freq，否则会导致损失数据点
        self.features = [self.calc_features(data) for data in alldatas]
        self.features = [ltp.add_head_tail(feature, head_timestamp=pd.to_datetime(s),
                                           tail_timestamp=pd.to_datetime(e)) for feature, (s, e) in
                         zip(self.features, ranges)]
        self.features = [ltp.change_freq(feature, freq=freq) for feature in self.features]

        # self.alldata = pd.concat([self.alldata, self.feature], axis=1)

        # 过早change freq会导致损失数据点
        alldatas = [ltp.change_freq(d, freq=freq) for d in alldatas]

        alldatas = [pd.merge(data, feature, left_index=True, right_index=True) for data, feature in
                         zip(alldatas, self.features)]

        return alldatas

    def transform_data(self, alldatas,stk_name):
        """
        主要是归一化和跳取数据，用于信号生成和回测，无需打乱
        :param alldatas:
        :return:
        """
        Xs = []
        ys = []
        for num in range(len(alldatas)):
            dp = ShiftDataPreprocessor()

            X, y = dp.get_flattened_Xy(alldatas, num, self.target, self.pred_n_steps, self.use_n_steps,
                                       self.drop_current)
            param = dp.sub_illegal_punctuation(str(self.param))
            dp.load_scaler(scaler_root, FILE_FMT_scaler.format(stk_name, num, param))

            cols = X.columns
            index = X.index
            X = pd.DataFrame(dp.scaler.transform(X), columns=cols, index=index)

            X = X.iloc[::self.use_n_steps]
            y = y.iloc[::self.use_n_steps]

            Xs.append(X)
            ys.append(y)

        return Xs, ys

    def run(self):
        """
        todo 需要增加多股票、多日期回测
        :return: 
        """
        
        for date in self.dates:
            for stk_name in self.stk_names:
    
                self.models = self.load_models(self.model_root, stk_name)
                self.alldata[date][stk_name] = self.load_data(file_root=self.file_root, date=date, stk_name=stk_name)
                self.alldatas[date][stk_name] = self.preprocess_data(self.alldata[date][stk_name])
                self.Xs, self.ys = self.transform_data(self.alldatas[date][stk_name],stk_name)
        
                y_preds = pd.Series()
                for num, (X_test, y_test, model) in enumerate(zip(self.Xs, self.ys, self.models)):
                    y_pred = model.predict(X_test)
                    y_pred = pd.Series(y_pred,
                                       index=X_test.index + timedelta(milliseconds=int(self.freq[:-2]) * self.pred_n_steps),
                                       name=f'pred_{self.target}_{self.pred_n_steps * 0.2}s').sort_index()
                    y_preds=pd.concat([y_preds, y_pred], axis=0)
                y_preds=y_preds.sort_index()
                # 单个股票的signals concat到所有signals上
                signals=self.strategy.generate_signals(y_preds, stk_name=stk_name, threshold=0.0008, drift=0)
                self.all_signals[date]=pd.concat([self.all_signals[date],signals],axis=0)
        print(self.all_signals)


        # start trade
        # :param signals: dict, {date:all_signals for all stks}
        # :param clean_obh_dict: dict, {date:{stk_name:ret <pd.DataFrame>}}
        self.broker.load_data(self.alldata)
        for date in self.dates:
            for stk_name in self.stk_names:
                signals=self.all_signals[date].sort_index()

                # todo 逐个signal进行模拟
                # for signal in signals: #(timestamp,stk_name,side,type,price_limit,volume)
                #     self.broker.execute(signal)

                # 批量交易
                revenue_dict,ret_dict,aligned_signals_dict=self.broker.batch_execute(signals,date,self.stk_names)


                stat_revenue=self.statistics.stat_winrate(revenue_dict[date][stk_name],aligned_signals_dict[date][stk_name]['side_open'],counterpart=True,params=None)
                stat_ret=self.statistics.stat_winrate(ret_dict[date][stk_name],aligned_signals_dict[date][stk_name]['side_open'],counterpart=True,params=None)

                stat_revenue.to_csv(res_root+f"{date}_{stk_name}_stat_revenue.csv")
                stat_ret.to_csv(res_root+f"{date}_{stk_name}_stat_ret.csv")

        return revenue_dict,ret_dict,aligned_signals_dict


