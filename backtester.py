# -*- coding=utf-8 -*-
# @File     : backtester.py
# @Time     : 2023/8/2 11:15
# @Author   : EvanHong
# @Email    : 939778128@qq.com
# @Project  : 2023.06.08超高频上证50指数计算
# @Description:
from __future__ import annotations

import logging
import os
import warnings
from abc import abstractmethod
from collections import defaultdict
from copy import deepcopy
from typing import List, Union

from sklearn.utils import shuffle

from backtest.config import *
import backtest.config as config
from backtest.support import *
from backtest.datafeeds.datafeed import LobDataFeed, LobModelFeed, PandasOHLCDataFeed, BaseDataFeed
from backtest.datafeeds.mkt_data_namespace import PandasLobMktDataNamespace, PandasOHLCMktDataNamespace
from backtest.signals.pandas_signal import PandasSignal
from backtest.recorders.observer import LobObserver, BtObserver
from backtest.preprocessors.preprocess import LobFeatureEngineering, LobTimePreprocessor, ShiftDataPreprocessor, \
    AggDataPreprocessor
from backtest.statistic_tools.statistics import LobStatistics
from backtest.strategies import LobStrategy
from backtest.broker.broker import Broker, StockBroker
from backtest.broker.orders import Order
from backtest.broker.trades import Trade
from backtest.recorders.transactions import Transaction
from backtest.recorders.position import Position
from backtest.recorders.portfolio import *
from backtest.signals.pandas_signal import PandasSignal
from backtest.strategies.single_asset_strategy import SingleAssetStrategy


class BaseTester(object):
    def __init__(self, *args, **kwargs):
        self.args = args
        for key, value in kwargs.items():
            self.__setattr__(key, value)

    @property
    def position(self):
        return self._position

    @position.setter
    def position(self, _position: Union[Position]):
        self._position = _position

    @abstractmethod
    def run(self):
        pass


class Screen(object):
    """
    sample class for properties and setters
    """

    @property
    def width(self):
        return self._width

    @width.setter
    def width(self, value):
        self._width = value

    @property
    def height(self):
        return self._height

    @height.setter
    def height(self, value):
        self._height = value

    @property
    def resolution(self):
        return self._height * self._width


class LobBackTester(BaseTester):
    """
    the whole project contains these files:
    1. backtester.py: the center and controler of the backtest project.
    2. ret.py: the main api to read/save ret from csv/xlsx files, and preprocess them for backtester. I have a lot of assets. And my ret frequency is 0.01s.
    3. strategies/base_strategy.py: strategies implementation and signal generation for backtester
    4. broker.py: including classes "Order", "Trade", "Broker" and other things you need.
    5. observer.py: recorder and logger for backtester
    6. statistics.py: result generator and visualization for backtester

    References
    ----------
    .. [#] Lean (Quantconnect): https://github.com/QuantConnect/Lean/blob/master/Documentation/2-Overview-Detailed-New.png
    .. [#] backtrader
    """

    def __init__(self,
                 model_root: str,
                 file_root: str,
                 dates: List[str | int],
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
        """

        Parameters
        ----------
        model_root :
        file_root :
        dates :
        stk_names :
        levels : int,
            n level 量价数据
        target :
        freq :
        pred_n_steps :
        use_n_steps :
        drop_current : bool,
            是否需要在特征中去掉current
        datafeed :
        strategy :
        broker :
        observer :
        statistics :
        args :
        kwargs :
        """
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

    def load_models(self, model_root, stk_name, model_class):
        model_loader = LobModelFeed(model_root=model_root, stk_name=stk_name, model_class=model_class)
        self.models = model_loader.models
        return self.models

    def load_data(self, file_root, date, stk_name, load_obh=True, load_vol_tov=True, load_events=True) -> pd.DataFrame:
        """

        :param file_root:
        :param date:
        :param stk_name:
        :return: pd.DataFrame,(clean_obh_dict+vol_tov), random freq
        """
        self.datafeed = LobDataFeed()
        dfs = []
        if load_obh:
            self.clean_obh = self.datafeed.load_clean_obh(file_root=file_root, date=date, stk_name=stk_name,
                                                          snapshot_window=self.levels)
            dfs.append(self.clean_obh)
        if load_vol_tov:
            self.vol_tov = self.datafeed.load_vol_tov(file_root=file_root, date=date, stk_name=stk_name)
            dfs.append(self.vol_tov)
        if load_events:
            self.events = self.datafeed.load_events(file_root=file_root, date=date, stk_name=stk_name)
            dfs.append(self.events)
        # self.trade_details,self.order_details=self.datafeed.load_details(data_root,date,code_dict[stk_name])
        data = pd.concat(dfs, axis=1).ffill()
        return data

    def _calc_features(self, df, level, to_freq=None):
        """

        Parameters
        ----------
        df:
            original frequency
        level
        to_freq

        Returns
        -------

        """
        # todo: 时间不连续、不规整，过于稀疏，归一化细节
        fe = LobFeatureEngineering()
        df = df.groupby(level=0).last()
        feature = fe.generate_cross_section(df, level=level)
        feature = feature.dropna(how='all')
        feature = pd.concat([df, feature], axis=1)
        feature.index = pd.to_datetime(feature.index)
        feature = feature.sort_index()
        # 必须先将clean_obh填充到10ms，否则交易频率是完全不规律的，即可能我只想用5个frame的数据来预测，但很可能用上了十秒的信息
        if to_freq is not None:
            feature = feature.asfreq(freq=to_freq, method='ffill')
        return feature

    # testit
    def calc_features(self, data, level, to_freq=None) -> list:
        """
        将数据划分为4份，每份一小时
        :param data: 10ms
        :return: 10ms
        """
        ltp = LobTimePreprocessor()
        # 必须先将数据切分，否则会导致11:30和13:00之间出现跳变
        alldatas = ltp.split_by_trade_period(data)
        # 不能对alldatas change freq，否则会导致损失数据点
        alldatas = [ltp.add_head_tail(cobh, head_timestamp=pd.to_datetime(s),
                                      tail_timestamp=pd.to_datetime(e)) for cobh, (s, e) in
                    zip(alldatas, config.ranges)]
        self.features = [self._calc_features(data, level=level, to_freq=to_freq) for data in alldatas]  # 尚未agg
        self.features = [ltp.add_head_tail(feature, head_timestamp=pd.to_datetime(s),
                                           tail_timestamp=pd.to_datetime(e)) for feature, (s, e) in
                         zip(self.features, config.ranges)]

        self.features = [feature.fillna(0) for feature in self.features]
        return self.features

    def scale_data(self, alldatas, stk_name, data_pp):
        """

        :param alldatas:
        :param stk_name:
        :param data_pp: data preprocessor
        :return:
        """
        Xs = []
        for num in range(len(alldatas)):
            param = data_pp.sub_illegal_punctuation(str(self.param))
            data_pp.load_scaler(scaler_root, FILE_FMT_scaler.format(stk_name, num, param))

            X = alldatas[num]
            cols = X.columns
            index = X.index
            X = pd.DataFrame(data_pp.scaler.transform(X), columns=cols, index=index)

            Xs.append(X)

        return Xs

    def match_y(self, Xs: list, features: list, used_timedelta,
                pred_timedelta, target: str, frolling=False):
        """
        fixme: 需要完善该接口

        Parameters
        ----------
        Xs: list
            一天中4个小时的数据
        features: list
            一天中4个小时的特征
        used_timedelta
            使用多久的数据
        pred_timedelta
            预测多少秒以后的target
        target
            class 'Target', ret, mid_p_ret
        frolling: default False
            原feature是否forward rolling，即frolling使用的是[t-n,t)的数据进行agg。

        Returns
        -------

        """
        logging.warning("deprecated", DeprecationWarning)
        logging.warning("请确保正确使用frolling", FutureWarning)

        _Xs = []
        _ys = []
        for X, feature in zip(Xs, features):

            start_time = X.index
            tar_time = start_time + pred_timedelta
            if not frolling:
                tar_time += used_timedelta
            # 波动率型
            if target == Target.vol.name:
                ...
                continue

            # return 类型的target
            if target == Target.ret.name:
                tar_col = LobColTemplate().current
            elif target == Target.mid_p_ret.name:
                tar_col = LobColTemplate().mid_price
            else:
                raise NotImplementedError()
            tar = feature[tar_col]

            available_time = [True if x in feature.index else False for x in tar_time]
            start_time = start_time[available_time]
            tar_time = tar_time[available_time]

            X = X.loc[start_time]
            y = np.log(tar.loc[tar_time] / tar.loc[start_time])

            _Xs.append(X)
            _ys.append(y)

        return _Xs, _ys

    def transform_data(self, alldatas, stk_name):
        """
        主要是归一化和跳取数据，用于信号生成和回测，无需打乱
        :param alldatas:
        :return:
        """
        warnings.warn(f"{self.transform_data} will be deprecated", DeprecationWarning)
        # raise DeprecationWarning(f"{self.transform_data} will be deprecated")
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

    def run_bt(self):
        """
        仅回测，不处理数据，不训练scalers、模型

        Notes
        -----
        明确进行回测的是哪些股票，哪些日期
        """
        self.models = self.load_models(self.model_root, 'general', model_class='automl')  # 默认会加载4个时间段的models

        # 明确进行回测的是哪些股票，哪些日期
        f_dict = defaultdict(list)  # {stk_name:(yyyy, mm, dd)}
        for date in self.dates:
            parts = str(pd.to_datetime(date).date()).split('-')
            yyyy = parts[0]
            mm = parts[1]
            dd = parts[2]
            for stk_name in self.stk_names:
                self.stk_name = stk_name
                f_dict[stk_name].append((yyyy, mm, dd))
        f_dict = {k: sorted(list(set(v))) for k, v in f_dict.items()}  # 去重

        # 读数据
        data_dict = defaultdict(lambda: defaultdict(list))  # data_dict={date:{stkname:[data0,data1,data2,data3]}
        tar_dict = defaultdict(dict)  # data_dict={date:{stkname:tar_data}}
        datafeed = LobDataFeed()
        for stk_name, date_tuples in f_dict.items():
            for yyyy, mm, dd in date_tuples:
                update_date(yyyy, mm, dd)
                try:
                    for num in range(4):
                        feature = datafeed.load_feature(detail_data_root, config.date, stk_name, num)
                        data_dict[config.date][stk_name].append(feature.dropna(how='all'))
                except FileNotFoundError as e:
                    print("missing feature", stk_name, yyyy, mm, dd)
                    continue

                # target
                tar = None
                self.alldata[config.date][stk_name] = datafeed.load_clean_obh(detail_data_root, config.date, stk_name,
                                                                              snapshot_window=use_level,
                                                                              use_cols=[
                                                                                  str(LobColTemplate('a', 1, 'p')),
                                                                                  str(LobColTemplate('a', 1, 'v')),
                                                                                  str(LobColTemplate('b', 1, 'p')),
                                                                                  str(LobColTemplate('b', 1, 'v')),
                                                                                  str(LobColTemplate().current)])
                temp = self.alldata[config.date][stk_name].asfreq(freq=min_freq, method='ffill')
                shift_rows = int(pred_timedelta / min_timedelta)  # 预测 pred_timedelta 之后的涨跌幅

                if config.target == Target.mid_p_ret.name:
                    tar = (temp[str(LobColTemplate('a', 1, 'p'))] + temp[str(LobColTemplate('b', 1, 'p'))]) / 2
                    tar = np.log(tar / tar.shift(shift_rows))  # log ret
                elif config.target == Target.ret.name:
                    tar = temp[LobColTemplate().current]
                    tar = np.log(tar / tar.shift(shift_rows))  # log ret
                elif config.target == Target.vol.name:
                    # 波动率
                    ...
                tar = LobTimePreprocessor().del_untrade_time(tar, cut_tail=True)  # 不能忘
                tar_dict[config.date][stk_name] = tar
                print("load", detail_data_root, stk_name, config.date)

        dp = AggDataPreprocessor()
        # X_test_dict = defaultdict(lambda: defaultdict(pd.DataFrame))
        # y_test_dict = defaultdict(lambda: defaultdict(pd.Series))
        for date, stk_data in list(data_dict.items()):
            for stk_name, features in stk_data.items():
                self.Xs, self.ys = [], []
                for num, feature in enumerate(features):
                    X, y = dp.align_Xy(feature, tar_dict[date][stk_name],
                                       pred_timedelta=pred_timedelta)  # 最重要的是对齐X y
                    # X_test_dict[stk_name][num] = pd.concat([X_test_dict[stk_name][num], X], axis=0)
                    # y_test_dict[stk_name][num] = pd.concat([y_test_dict[stk_name][num], y], axis=0)

                    # scale X data
                    dp.load_scaler(scaler_root, FILE_FMT_scaler.format(stk_name, num, '_'))
                    X, = dp.std_scale(X, refit=False)
                    self.Xs.append(X)
                    self.ys.append(y)

                y_preds = pd.Series()
                for num, (X_test, y_test, model) in enumerate(zip(self.Xs, self.ys, self.models)):
                    y_pred = model.predict(X_test)
                    y_pred = pd.Series(y_pred, index=y_test.index,
                                       name=f'pred_{self.target}_{self.pred_n_steps * 0.2}s').sort_index()
                    y_preds = pd.concat([y_preds, y_pred], axis=0)
                y_preds = y_preds.sort_index()
                # 单个股票的signals concat到所有signals上
                signals = self.strategy.generate_signals(y_preds, stk_name=stk_name, threshold=0.001, drift=0)
                self.all_signals[date] = pd.concat([self.all_signals[date], signals], axis=0)
        print(self.all_signals)

        # start trade
        # :param signals: dict, {date:all_signals for all stks}
        # :param clean_obh_dict: dict, {date:{stk_name:ret <pd.DataFrame>}}
        self.broker.load_data(self.alldata)
        # todo 需要增加多股票、多日期回测
        revenue_dict, ret_dict, aligned_signals_dict = None, None, None
        for date in list(data_dict.keys()):
            for stk_name in self.stk_names:
                signals = self.all_signals[date].sort_index()

                # todo 逐个signal进行模拟
                # for signal in signals: #(timestamp,stk_name,side,type,price_limit,volume)
                #     self.broker.execute(signal)

                # 批量交易
                revenue_dict, ret_dict, aligned_signals_dict = self.broker.batch_execute(signals, use_dates=None,
                                                                                         use_stk_names=None)

                stat_revenue = self.statistics.stat_winrate(revenue_dict[date][stk_name],
                                                            aligned_signals_dict[date][stk_name]['side_open'],
                                                            counterpart=True, params=None)
                stat_ret = self.statistics.stat_winrate(ret_dict[date][stk_name],
                                                        aligned_signals_dict[date][stk_name]['side_open'],
                                                        counterpart=True, params=None)

                stat_revenue.to_csv(res_root + f"{date}_{stk_name}_stat_revenue_pred{pred_timedelta}.csv")
                stat_ret.to_csv(res_root + f"{date}_{stk_name}_stat_ret_pred{pred_timedelta}.csv")

        return revenue_dict, ret_dict, aligned_signals_dict

    def run(self):
        """old version

        deprecated

        :return: 
        """
        raise DeprecationWarning("run has been deprecated")

        for date in self.dates:
            for stk_name in self.stk_names:
                self.stk_name = stk_name
                # 默认会加载4个时间段的models
                self.models = self.load_models(self.model_root, 'general', model_class='automl')  # 默认会加载4个时间段的models

                self.alldata[date][stk_name] = self.load_data(file_root=self.file_root, date=date,
                                                              stk_name=stk_name)  # random freq

                self.alldatas[date][stk_name] = self.calc_features(
                    self.alldata[date][stk_name], level=use_level, to_freq=min_freq)  # min_freq, 10ms

                dp = AggDataPreprocessor()
                # agg_freq=1min
                self.alldatas[date][stk_name] = [dp.agg_features(feature) for feature in self.alldatas[date][stk_name]]

                # self.Xs, self.ys = self.transform_data(self.alldatas[date][stk_name],stk_name)

                self.Xs = self.scale_data(self.alldatas[date][stk_name], stk_name, data_pp=dp)
                self.Xs, self.ys = self.match_y(self.Xs, self.alldatas[date][stk_name],
                                                used_timedelta=timedelta(minutes=int(freq[:-3]) * use_n_steps),
                                                pred_timedelta=timedelta(minutes=int(freq[:-3]) * pred_n_steps),
                                                target=Target.mid_p_ret.name,
                                                frolling=False)

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
        revenue_dict, ret_dict, aligned_signals_dict = None, None, None
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


class SingleAssetBackTester(BaseTester):
    """
    the whole project contains these files:
    1. backtester.py: the center and controler of the backtest project.
    2. ret.py: the main api to read/save ret from csv/xlsx files, and preprocess them for backtester. I have a lot of assets. And my ret frequency is 0.01s.
    3. strategies/base_strategy.py: strategies implementation and signal generation for backtester
    4. broker.py: including classes "Order", "Trade", "Broker" and other things you need.
    5. observer.py: recorder and logger for backtester
    6. statistics.py: result generator and visualization for backtester

    References
    ----------
    .. [#] Lean (Quantconnect): https://github.com/QuantConnect/Lean/blob/master/Documentation/2-Overview-Detailed-New.png
    .. [#] backtrader
    """

    def __init__(self,
                 dates: List[str | int],
                 target: str,
                 freq: str,
                 strategy: Union[LobStrategy, SingleAssetStrategy] = None,
                 broker: Union[StockBroker] = None,
                 recorder: Union[LobObserver, BtObserver] = None,
                 statistics: Union[LobStatistics] = None,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.strategy = strategy
        self.broker = broker
        self.recorder = recorder
        self.statistics = statistics

        self.dates = dates
        self.target = target
        self.freq = freq
        self.stk_names = list()

        self.alldata = defaultdict(dict)  # {dates:{stk_name:ret}}
        self.alldatas = defaultdict(dict)  # {dates:{stk_name:[data1,data2,...]}}
        # self.all_signals = defaultdict(pd.DataFrame)
        self.all_signals = None
        self.mkt_data = None

    # def load_models(self, model_root, stk_name, model_class):
    #     model_loader = LobModelFeed(model_root=model_root, stk_name=stk_name, model_class=model_class)
    #     self.models = model_loader.models
    #     return self.models
    #
    # def load_data(self, file_root, date, stk_name,load_obh=True,load_vol_tov=True,load_events=True) -> pd.DataFrame:
    #     """
    #
    #     :param file_root:
    #     :param date:
    #     :param stk_name:
    #     :return: pd.DataFrame,(clean_obh_dict+vol_tov), random freq
    #     """
    #     self.datafeed = LobDataFeed()
    #     dfs=[]
    #     if load_obh:
    #         self.clean_obh = self.datafeed.load_clean_obh(file_root=file_root, date=date, stk_name=stk_name,
    #                                                   snapshot_window=self.levels)
    #         dfs.append(self.clean_obh)
    #     if load_vol_tov:
    #         self.vol_tov = self.datafeed.load_vol_tov(file_root=file_root, date=date, stk_name=stk_name)
    #         dfs.append(self.vol_tov)
    #     if load_events:
    #         self.events = self.datafeed.load_events(file_root=file_root, date=date, stk_name=stk_name)
    #         dfs.append(self.events)
    #     # self.trade_details,self.order_details=self.datafeed.load_details(data_root,date,code_dict[stk_name])
    #     data = pd.concat(dfs, axis=1).ffill()
    #     return data
    #
    # def _calc_features(self, df, level, to_freq=None):
    #     """
    #
    #     Parameters
    #     ----------
    #     df:
    #         original frequency
    #     level
    #     to_freq
    #
    #     Returns
    #     -------
    #
    #     """
    #     # todo: 时间不连续、不规整，过于稀疏，归一化细节
    #     fe = LobFeatureEngineering()
    #     df = df.groupby(level=0).last()
    #     feature = fe.generate_cross_section(df, level=level)
    #     feature=feature.dropna(how='all')
    #     feature = pd.concat([df, feature], axis=1)
    #     feature.index = pd.to_datetime(feature.index)
    #     feature = feature.sort_index()
    #     # 必须先将clean_obh填充到10ms，否则交易频率是完全不规律的，即可能我只想用5个frame的数据来预测，但很可能用上了十秒的信息
    #     if to_freq is not None:
    #         feature = feature.asfreq(freq=to_freq, method='ffill')
    #     return feature
    #
    # # testit
    # def calc_features(self, data, level, to_freq=None) -> list:
    #     """
    #     将数据划分为4份，每份一小时
    #     :param data: 10ms
    #     :return: 10ms
    #     """
    #     ltp = LobTimePreprocessor()
    #     # 必须先将数据切分，否则会导致11:30和13:00之间出现跳变
    #     alldatas = ltp.split_by_trade_period(data)
    #     # 不能对alldatas change freq，否则会导致损失数据点
    #     alldatas = [ltp.add_head_tail(cobh, head_timestamp=pd.to_datetime(s),
    #                                   tail_timestamp=pd.to_datetime(e)) for cobh, (s, e) in
    #                 zip(alldatas, config.ranges)]
    #     self.features = [self._calc_features(data, level=level, to_freq=to_freq) for data in alldatas]  # 尚未agg
    #     self.features = [ltp.add_head_tail(feature, head_timestamp=pd.to_datetime(s),
    #                                        tail_timestamp=pd.to_datetime(e)) for feature, (s, e) in
    #                      zip(self.features, config.ranges)]
    #
    #     self.features = [feature.fillna(0) for feature in self.features]
    #     return self.features
    #
    # def scale_data(self, alldatas, stk_name, data_pp):
    #     """
    #
    #     :param alldatas:
    #     :param stk_name:
    #     :param data_pp: data preprocessor
    #     :return:
    #     """
    #     Xs = []
    #     for num in range(len(alldatas)):
    #         param = data_pp.sub_illegal_punctuation(str(self.param))
    #         data_pp.load_scaler(scaler_root, FILE_FMT_scaler.format(stk_name, num, param))
    #
    #         X = alldatas[num]
    #         cols = X.columns
    #         index = X.index
    #         X = pd.DataFrame(data_pp.scaler.transform(X), columns=cols, index=index)
    #
    #         Xs.append(X)
    #
    #     return Xs
    #
    # def match_y(self, Xs: list, features: list, used_timedelta,
    #             pred_timedelta, target: str, frolling=False):
    #     """
    #     fixme: 需要完善该接口
    #
    #     Parameters
    #     ----------
    #     Xs: list
    #         一天中4个小时的数据
    #     features: list
    #         一天中4个小时的特征
    #     used_timedelta
    #         使用多久的数据
    #     pred_timedelta
    #         预测多少秒以后的target
    #     target
    #         class 'Target', ret, mid_p_ret
    #     frolling: default False
    #         原feature是否forward rolling，即frolling使用的是[t-n,t)的数据进行agg。
    #
    #     Returns
    #     -------
    #
    #     """
    #     logging.warning("deprecated", DeprecationWarning)
    #     logging.warning("请确保正确使用frolling", FutureWarning)
    #
    #     _Xs = []
    #     _ys = []
    #     for X, feature in zip(Xs, features):
    #
    #         start_time = X.index
    #         tar_time = start_time + pred_timedelta
    #         if not frolling:
    #             tar_time += used_timedelta
    #         # 波动率型
    #         if target == Target.vol.name:
    #             ...
    #             continue
    #
    #         # return 类型的target
    #         if target == Target.ret.name:
    #             tar_col = LobColTemplate().current
    #         elif target == Target.mid_p_ret.name:
    #             tar_col = LobColTemplate().mid_price
    #         else:
    #             raise NotImplementedError()
    #         tar = feature[tar_col]
    #
    #         available_time = [True if x in feature.index else False for x in tar_time]
    #         start_time = start_time[available_time]
    #         tar_time = tar_time[available_time]
    #
    #         X = X.loc[start_time]
    #         y = np.log(tar.loc[tar_time] / tar.loc[start_time])
    #
    #         _Xs.append(X)
    #         _ys.append(y)
    #
    #     return _Xs, _ys
    #
    # def transform_data(self, alldatas, stk_name):
    #     """
    #     主要是归一化和跳取数据，用于信号生成和回测，无需打乱
    #     :param alldatas:
    #     :return:
    #     """
    #     warnings.warn(f"{self.transform_data} will be deprecated", DeprecationWarning)
    #     # raise DeprecationWarning(f"{self.transform_data} will be deprecated")
    #     Xs = []
    #     ys = []
    #     for num in range(len(alldatas)):
    #         dp = ShiftDataPreprocessor()
    #
    #         X, y = dp.get_flattened_Xy(alldatas, num, self.target, self.pred_n_steps, self.use_n_steps,
    #                                    self.drop_current)
    #         param = dp.sub_illegal_punctuation(str(self.param))
    #         dp.load_scaler(scaler_root, FILE_FMT_scaler.format(stk_name, num, param))
    #
    #         cols = X.columns
    #         index = X.index
    #         X = pd.DataFrame(dp.scaler.transform(X), columns=cols, index=index)
    #
    #         X = X.iloc[::self.use_n_steps]
    #         y = y.iloc[::self.use_n_steps]
    #
    #         Xs.append(X)
    #         ys.append(y)
    #
    #     return Xs, ys

    def add_market_data(self, data: Union[PandasOHLCDataFeed]):
        def split_datafeed(df: Union[PandasOHLCDataFeed]):
            data_dict = defaultdict(dict)
            temp = df.data.groupby(by=[df.namespace.date, df.namespace.symbol])
            for gg in temp:
                self.dates.append(str(gg[0][0]))
                self.stk_names.append(gg[0][1])
                data_dict[str(gg[0][0])][gg[0][1]] = gg[1]
            self.dates = sorted(self.dates)
            self.stk_names = sorted(self.stk_names)
            return data_dict

        self.data_dict = split_datafeed(data)
        self.alldatas = self.data_dict
        self.mkt_data = data
        self.broker.load_data(self.data_dict)

    def add_signals(self, signal: Union[PandasSignal]):
        self.all_signals = signal

    def run(self, save_root='./'):
        """old version

        deprecated

        :return:
        """

        # 单个股票的signals concat到所有signals上
        # signals = self.strategy.generate_signals(self.all_signals.data, stk_name=stk_name, threshold=0.0008, drift=1)
        # self.all_signals.loc[date] = pd.concat([self.all_signals[date], signals], axis=0)
        # print(self.all_signals)

        # start trade
        # :param signals: dict, {date:all_signals for all stks}
        # :param clean_obh_dict: dict, {date:{stk_name:ret <pd.DataFrame>}}
        # self.broker.load_data(self.alldata)
        # todo 需要增加多股票、多日期回测
        for date in self.dates:
            for stk_name in self.stk_names:
                signals = self.all_signals.loc[date].sort_index()

                # todo 逐个signal进行模拟
                # for signal in signals: #(timestamp,stk_name,side,type,price_limit,volume)
                #     self.broker.execute(signal)

                # 批量交易
                revenue_dict, ret_dict, aligned_signals_dict = self.broker.batch_execute(signals, [date],
                                                                                         self.stk_names,
                                                                                         commission=self.broker.commission)
                self.recorder.revenue_dict.update(revenue_dict)
                self.recorder.ret_dict.update(ret_dict)
                self.recorder.aligned_signals_dict.update(aligned_signals_dict)
                self.recorder.res_dict[date][stk_name] = (revenue_dict, ret_dict, aligned_signals_dict)

                stat_revenue = self.statistics.stat_winrate(revenue_dict[date][stk_name],
                                                            aligned_signals_dict[date][stk_name]['side_open'],
                                                            counterpart=True, params=None)
                stat_ret = self.statistics.stat_winrate(ret_dict[date][stk_name],
                                                        aligned_signals_dict[date][stk_name]['side_open'],
                                                        counterpart=True, params=None)

                stat_revenue.to_csv(save_root + f"{date}_{stk_name}_stat_revenue.csv")
                stat_ret.to_csv(save_root + f"{date}_{stk_name}_stat_ret.csv")

        return self.recorder.res_dict


if __name__ == '__main__':
    stk_names = ["贵州茅台", "中信证券"]
    update_date('2022', '06', '29')
    datafeed = LobDataFeed()
    strategy = LobStrategy(max_close_timedelta=timedelta(minutes=int(freq[:-3]) * pred_n_steps))
    broker = Broker(cash=1e6, commission=1e-3)
    observer = LobObserver()
    statistics = LobStatistics()

    bt = LobBackTester(model_root=model_root,
                       file_root=detail_data_root,
                       dates=['2022-06-29'],  # todo 确认一致性是否有bug
                       stk_names=stk_names,
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
                       statistics=statistics,
                       )

    bt.run_bt()  #
    # bt.run()  #

    print()
