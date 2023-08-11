# -*- coding=utf-8 -*-
# @File     : backtester.py
# @Time     : 2023/8/2 11:15
# @Author   : EvanHong
# @Email    : 939778128@qq.com
# @Project  : 2023.06.08超高频上证50指数计算
# @Description:
import datetime
import warnings
from abc import abstractmethod
from collections import defaultdict
from typing import List, Union

import pandas as pd

from broker import Broker
from config import *
from support import *
from datafeed import LobDataFeed, LobModelFeed
from observer import LobObserver
from preprocess import LobFeatureEngineering, LobTimePreprocessor, ShiftDataPreprocessor, AggDataPreprocessor
from statistics import LobStatistics
from strategies import LobStrategy


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
                 model_root: str,
                 file_root: str,
                 dates: List[str],
                 stk_names: List[str],
                 levels: int,
                 target: str,
                 freq: str,
                 pred_n_steps: int,
                 use_n_steps: int,
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

        self.alldata = defaultdict(dict)  # {dates:{stk_name:ret}}
        self.alldatas = defaultdict(dict)  # {dates:{stk_name:[data1,data2,...]}}
        self.all_signals = defaultdict(pd.DataFrame)

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
        self.datafeed = LobDataFeed(file_root=file_root, date=date, stk_name=stk_name)
        self.clean_obh = self.datafeed.load_clean_obh(snapshot_window=self.levels)
        self.vol_tov = self.datafeed.load_vol_tov()
        self.events = self.datafeed.load_events()
        # self.trade_details,self.order_details=self.datafeed.load_details(data_root,date,code_dict[stk_name])
        data = pd.concat([self.clean_obh, self.vol_tov, self.events], axis=1).ffill()
        # 必须先将clean_obh填充到10ms，否则交易频率是完全不规律的，即可能我只想用5个frame的数据来预测，但很可能用上了十秒的信息
        data = data.asfreq(freq='10ms', method='ffill')
        return data

    def calc_features(self, df, level):
        # todo: 时间不连续、不规整，过于稀疏，归一化细节
        fe = LobFeatureEngineering()
        feature = fe.generate(df, level=level)
        feature = pd.concat([df, feature], axis=1)
        feature.index = pd.to_datetime(feature.index)
        feature = feature.sort_index()
        feature = fe.agg_features(feature, agg_freq="5min")
        return feature

    # testit
    def preprocess_data(self, data) -> list:
        """
        将数据划分为4份，每份一小时
        :param data: 10ms
        :return:
        """
        ltp = LobTimePreprocessor()
        # 必须先将数据切分，否则会导致11:30和13:00之间出现跳变
        alldatas = ltp.split_by_trade_period(data)
        alldatas = [ltp.add_head_tail(cobh, head_timestamp=pd.to_datetime(s),
                                      tail_timestamp=pd.to_datetime(e)) for cobh, (s, e) in
                    zip(alldatas, config.ranges)]
        # 不能对alldatas change freq，否则会导致损失数据点
        self.features = [self.calc_features(data, level=use_level) for data in alldatas]
        self.features = [ltp.add_head_tail(feature, head_timestamp=pd.to_datetime(s),
                                           tail_timestamp=pd.to_datetime(e)) for feature, (s, e) in
                         zip(self.features, config.ranges)]
        # self.features = [ltp.change_freq(feature, freq=freq) for feature in self.features]
        # 过早change freq会导致损失数据点
        # alldatas = [ltp.change_freq(d, freq=freq) for d in alldatas]
        # alldatas = [pd.merge(data, feature, left_index=True, right_index=True) for data, feature in
        #                  zip(alldatas, self.features)]

        return self.features

    def scale_data(self, alldatas, stk_name):
        Xs = []
        for num in range(len(alldatas)):
            dp = AggDataPreprocessor()
            param = dp.sub_illegal_punctuation(str(self.param))
            dp.load_scaler(scaler_root, FILE_FMT_scaler.format(stk_name, num, param))

            X = alldatas[num]
            cols = X.columns
            index = X.index
            X = pd.DataFrame(dp.scaler.transform(X), columns=cols, index=index)

            Xs.append(X)

        return Xs

    def match_y(self, Xs, features, used_timedelta,
                pred_timedelta, target: str):
        """

        :param X:
        :param features:
        :param used_timedelta: 使用多久的数据
        :param pred_timedelta: 预测多少秒以后的target
        :param target: class 'Target', ret, mid_p_ret
        :return:
        """
        _Xs=[]
        _ys=[]
        for X,feature in zip(Xs,features):

            start_time = X.index
            tar_time = start_time + used_timedelta + pred_timedelta
            if target == Target.ret.name:
                tar_col = LobColTemplate().current
            elif target == Target.mid_p_ret.name:
                tar_col = LobColTemplate().mid_price
            else:
                raise NotImplementedError()
            tar = feature[tar_col]

            # return 类型的target
            available_time = [True if x in feature.index else False for x in tar_time]
            start_time = start_time[available_time]
            tar_time = tar_time[available_time]

            X = X.loc[start_time]
            y = np.log(tar.loc[tar_time] / tar.loc[start_time])

            _Xs.append(X)
            _ys.append(y)

            # 波动率型
            ...

        return _Xs,_ys

    def transform_data(self, alldatas, stk_name):
        """
        主要是归一化和跳取数据，用于信号生成和回测，无需打乱
        :param alldatas:
        :return:
        """
        warnings.warn(f"{self.transform_data} will be deprecated", FutureWarning)
        # raise FutureWarning(f"{self.transform_data} will be deprecated")
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

        :return: 
        """

        for date in self.dates:
            for stk_name in self.stk_names:
                self.stk_name = stk_name

                self.models = self.load_models(self.model_root, stk_name)
                # 10ms data
                self.alldata[date][stk_name] = self.load_data(file_root=self.file_root, date=date, stk_name=stk_name)
                self.alldatas[date][stk_name] = self.preprocess_data(self.alldata[date][stk_name])
                # self.Xs, self.ys = self.transform_data(self.alldatas[date][stk_name],stk_name)
                self.Xs = self.scale_data(self.alldatas[date][stk_name], stk_name)
                self.Xs,self.ys = self.match_y(self.Xs, self.alldatas[date][stk_name],
                                        used_timedelta=timedelta(milliseconds=int(freq[:-2]) * use_n_steps),
                                        pred_timedelta=timedelta(milliseconds=int(freq[:-2]) * pred_n_steps),
                                        target=Target.mid_p_ret.name)

                y_preds = pd.Series()
                for num, (X_test, y_test, model) in enumerate(zip(self.Xs, self.ys, self.models)):
                    y_pred = model.predict(X_test)
                    y_pred = pd.Series(y_pred,
                                       index=X_test.index + timedelta(
                                           milliseconds=int(self.freq[:-2]) * self.pred_n_steps),
                                       name=f'pred_{self.target}_{self.pred_n_steps * 0.2}s').sort_index()
                    y_preds = pd.concat([y_preds, y_pred], axis=0)
                y_preds = y_preds.sort_index()
                # 单个股票的signals concat到所有signals上
                signals = self.strategy.generate_signals(y_preds, stk_name=stk_name, threshold=0.0008, drift=0)
                self.all_signals[date] = pd.concat([self.all_signals[date], signals], axis=0)
        print(self.all_signals)

        # start trade
        # :param signals: dict, {date:all_signals for all stks}
        # :param clean_obh_dict: dict, {date:{stk_name:ret <pd.DataFrame>}}
        self.broker.load_data(self.alldata)
        # todo 需要增加多股票、多日期回测
        for date in self.dates:
            for stk_name in self.stk_names:
                signals = self.all_signals[date].sort_index()

                # todo 逐个signal进行模拟
                # for signal in signals: #(timestamp,stk_name,side,type,price_limit,volume)
                #     self.broker.execute(signal)

                # 批量交易
                revenue_dict, ret_dict, aligned_signals_dict = self.broker.batch_execute(signals, date, self.stk_names)

                stat_revenue = self.statistics.stat_winrate(revenue_dict[date][stk_name],
                                                            aligned_signals_dict[date][stk_name]['side_open'],
                                                            counterpart=True, params=None)
                stat_ret = self.statistics.stat_winrate(ret_dict[date][stk_name],
                                                        aligned_signals_dict[date][stk_name]['side_open'],
                                                        counterpart=True, params=None)

                stat_revenue.to_csv(res_root + f"{date}_{stk_name}_stat_revenue.csv")
                stat_ret.to_csv(res_root + f"{date}_{stk_name}_stat_ret.csv")

        return revenue_dict, ret_dict, aligned_signals_dict
