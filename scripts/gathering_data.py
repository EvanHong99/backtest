# -*- coding=utf-8 -*-
# @File     : gathering_data.py
# @Time     : 2023/7/28 14:54
# @Author   : EvanHong
# @Email    : 939778128@qq.com
# @Project  : 2023.06.08超高频上证50指数计算
# @Description: 将不同股票数据都整合到一起，从而为模型训练提供足量的数据

import logging
import pandas as pd
import numpy as np
from math import sqrt
from dateutil import relativedelta
import matplotlib.pyplot as plt
import os
from datetime import timedelta
import statsmodels.api as sm
import warnings
import time as t
from copy import deepcopy
from pandas.tseries.offsets import MonthBegin,MonthEnd
import h5py
import hdf5plugin
from enum import Enum
from collections import defaultdict,OrderedDict
import json
from sortedcontainers import SortedDict
import time
from tqdm import tqdm
import time

import h5py
import hdf5plugin
import json
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle
import re
import time as t
import warnings
from collections import defaultdict
from copy import deepcopy
from datetime import timedelta
from dateutil import relativedelta
from flaml import AutoML
from sklearn.cluster import OPTICS, DBSCAN
from sklearn.linear_model import LassoCV
from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import ParameterGrid, train_test_split
from utils import *
from typing import Tuple, Optional,List
from datetime import datetime
from sklearn.preprocessing import StandardScaler


class ColTplt(object):
    """
    column template
    """

    def __init__(self, side: str, level: int, target: str):
        self.side = side
        self.level = level
        self.target = target

    def __str__(self):
        return f"{self.side}{self.level}_{self.target}"


class LOBFeatureEngineering(object):
    """
    todo: 买入意愿因子《高频因子的现实与幻想》，需要利用9:30开盘后半小时内的数据构建该因子
    """

    def __init__(self):
        self.ap = {k: str(ColTplt('a', k, 'p')) for k in range(1, 6)}
        self.bp = {k: str(ColTplt('b', k, 'p')) for k in range(1, 6)}
        self.av = {k: str(ColTplt('a', k, 'v')) for k in range(1, 6)}
        self.bv = {k: str(ColTplt('b', k, 'v')) for k in range(1, 6)}
        self.curr = 'current'

    def calc_wap(self, df, level, cross=True):
        """Function to calculate first WAP"""
        if cross:
            wap = (df[self.bp[level]] * df[self.av[level]] + df[self.ap[level]] * df[self.bv[level]]) / (
                    df[self.bv[level]] + df[self.av[level]])
        else:
            wap = (df[self.ap[level]] * df[self.av[level]] + df[self.bp[level]] * df[self.bv[level]]) / (
                    df[self.bv[level]] + df[self.av[level]])
        name = f'wap{level}'
        if cross: name += '_c'
        return wap.rename(name)

    def calc_mid_price(self, df):
        return (df[self.ap[1]] + df[self.bp[1]]).rename('mid_price') / 2

    def calc_spread(self, df):
        return (df[self.ap[1]] - df[self.bp[1]]).rename('spread')

    def calc_relative_spread(self, df):
        return (self.calc_spread(df) / self.calc_mid_price(df)).rename('relative_spread')

    def calc_price_breadth(self, df):
        """

        :param df:
        :return:

        .. [#] High-Frequency TradingAspects of market liquidity（Bervas，2006）
        """

        pb_b = (df[self.curr] - df[self.bp[1]]).rename('price_breadth_b')
        pb_a = (df[self.ap[1]] - df[self.curr]).rename('price_breadth_a')
        return pb_b, pb_a

    def calc_depth_imbalance(self):
        """

        :return:

        .. [#] Cao, C., Hansch, O., & Wang, X. (2009). The information content of an open limit‐order book. Journal of Futures Markets: Futures, Options, and Other Derivative Products, 29(1), 16-41.
        """
        pass

    def calc_height_imbalance(self):
        """

        :return:

        .. [#] Cao, C., Hansch, O., & Wang, X. (2009). The information content of an open limit‐order book. Journal of Futures Markets: Futures, Options, and Other Derivative Products, 29(1), 16-41.
        """
        pass

    def calc_spread_tick(self, df, num_levels=5):
        """
        盘口信息是判断个股短期走势的重要依据，
        短期的市场价格是取决于当前 买盘 需求量
        和卖盘 供给 量构建的均衡价格。因此，我们判断 盘口 买卖挂单的相对强弱对于股票价格的
        短期走势具有统计意义上显著的预判作用。 当需求远大于供给时，均衡价格将上移； 相反
        地， 当供给远大于需求时，均衡价格将下降。

        .. math::
            bid=\\Sigma_{i=1}^{num\\_levels}bidprice_i*bidvol_i*w_i\\newline
            ask=\\Sigma_{i=1}^{num\\_levels}askprice_i*askvol_i*w_i\\newline
            w_i=1-\\frac{i-1}{num\\_levels}\\newline
            spread\\_tick=(bid-ask)/(bid+ask)

        .. [#] 2019-09-05_天风证券_市场微观结构探析系列之二：订单簿上的alpha

        :param num_levels: 档位数量
        :return:
        """
        weights = {}
        for i in range(1, num_levels + 1):
            weights[i] = 1 - (i - 1) / num_levels

        bid = ask = 0
        for i in range(1, num_levels + 1):
            bid += df[self.bp[i]] * df[self.bv[i]] * weights[i]
            ask += df[self.ap[i]] * df[self.av[i]] * weights[i]

        spread_tick = (bid - ask) / (bid + ask)
        spread_tick = spread_tick.rename('spread_tick')
        return spread_tick

    def calc_realized_volatility(self, series):
        return np.sqrt(np.sum(series ** 2))

    def calc_volume_order_imbalance(self, df: pd.DataFrame):
        """
        .. [#] Reference From <Order imbalance Based LobStrategy in High Frequency Trading>
        """
        current_bid_price = df[self.bp[1]]
        bid_price_diff = current_bid_price - current_bid_price.shift()
        current_bid_vol = df[self.bv[1]]
        bvol_diff = current_bid_vol - current_bid_vol.shift()
        # 右结合，最后一个where是直接return新的array，而不是去修改bid_price_diff
        bid_increment = np.where(bid_price_diff > 0, current_bid_vol,
                                 np.where(bid_price_diff < 0, 0,
                                          np.where(bid_price_diff == 0, bvol_diff, bid_price_diff)))

        current_ask_price = df[self.ap[1]]
        ask_price_diff = current_ask_price - current_ask_price.shift()
        current_ask_vol = df[self.av[1]]
        avol_diff = current_ask_vol - current_ask_vol.shift()
        ask_increment = np.where(ask_price_diff < 0, current_ask_vol,
                                 np.where(ask_price_diff > 0, 0,
                                          np.where(ask_price_diff == 0, avol_diff, ask_price_diff)))

        res = pd.Series(bid_increment - ask_increment, index=df.index, name='voi')

        return res

    def calc_buy_sell_pressure(self, df, level1, level2, method='MID'):
        """
        利用mid price计算买卖压力差
        :param df:
        :param level1:
        :param level2:
        :param method:
        :return:

        .. [#] [天风证券_买卖压力失衡——利用高频数据拓展盘口数据](https://bigquant.com/wiki/pdfjs/web/viewer.html?file=/wiki/static/upload/2b/2bc961b3-e365-4afb-aa98-e9f9d9fa299e.pdf)
        """
        assert level1 < level2
        levels = np.arange(level1, level2 + 1)
        if method == "MID":
            M = self.calc_mid_price(df)
        else:
            raise NotImplementedError('price_weighted_pressure')

        # 需要注意，如果不适用middle price那么可能分母会出现nan
        bid_d = np.array([M / (M - df[self.bp[s]]) for s in levels])
        # bid_d = [_.replace(np.inf,0) for _ in bid_d]
        bid_denominator = sum(bid_d)
        bid_weights = bid_d / bid_denominator
        press_buy = sum([df[self.bv[i + 1]] * w for i, w in enumerate(bid_weights)])

        ask_d = np.array([M / (df[self.ap[s]] - M) for s in levels])
        # ask_d = [_.replace(np.inf,0) for _ in ask_d]
        ask_denominator = sum(ask_d)
        ask_weights = ask_d / ask_denominator
        press_sell = sum([df[self.av[i + 1]] * w for i, w in enumerate(ask_weights)])

        res = pd.Series((np.log(press_buy) - np.log(press_sell)).replace([-np.inf, np.inf], np.nan),
                        name='buy_sell_pressure')
        return res

    def generate(self, clean_obh):
        self.waps = [self.calc_wap(clean_obh, level=i, cross=False) for i in range(1, 6)] + [
            self.calc_wap(clean_obh, level=i, cross=True) for i in range(1, 6)]
        self.spread = self.calc_spread(clean_obh)
        self.breadth_b, self.breadth_a = self.calc_price_breadth(clean_obh)
        self.rs = self.calc_relative_spread(clean_obh)
        self.st = self.calc_spread_tick(clean_obh, num_levels=5)
        self.voi = self.calc_volume_order_imbalance(clean_obh)
        self.bsp = self.calc_buy_sell_pressure(clean_obh, level1=1, level2=5, method='MID')

        self.features = pd.concat(
            self.waps + [self.spread, self.breadth_b, self.breadth_a, self.rs, self.st, self.voi, self.bsp], axis=1)
        return self.features


class LOBPreprocessor(object):
    def __init__(self):
        pass

    def del_untrade_time(self, df, cut_tail=True):
        """

        :param df:
        :param cut_tail: 去掉尾盘3min竞价
        :return:
        """
        is_series = False
        if isinstance(df, pd.Series):
            is_series = True
            df = df.to_frame()
        end_time = pd.to_datetime(f'{date1} 15:00:00.00') if not cut_tail else pd.to_datetime(f'{date1} 14:57:00.00')
        temp = pd.concat([df.loc[pd.to_datetime(f'{date1} 09:30:00.00'):pd.to_datetime(f'{date1} 11:30:00.00')],
                          df.loc[pd.to_datetime(f'{date1} 13:00:00.00'):end_time]])
        # if temp.index[0] > pd.to_datetime(f'{date1} 09:30:00.00'):
        #     temp.loc[pd.to_datetime(f'{date1} 09:30:00.00')] = df.loc[:pd.to_datetime(f'{date1} 09:30:00.00')].iloc[-1]
        # if temp.index[-1] < pd.to_datetime(f'{date1} 15:00:00.00'):
        #     temp.loc[pd.to_datetime(f'{date1} 15:00:00.00')] = df.loc[:pd.to_datetime(f'{date1} 15:00:00.01')].iloc[-1]
        if is_series:
            temp = temp.iloc[:, 0]
        temp = temp.sort_index()
        return temp

    def add_head_tail(self, df, head_timestamp=pd.to_datetime(f'{date1} 09:30:00.000'),
                      tail_timestamp=pd.to_datetime(f'{date1} 14:57:00.000')):
        # df=df.dropna(how='any',axis=0)
        try:
            assert df.index[0] >= head_timestamp and df.index[-1] <= tail_timestamp
        except Exception as e:
            print('add_head_tail', df.index[0], head_timestamp, df.index[-1], tail_timestamp)
            raise e
        df.loc[pd.to_datetime(tail_timestamp)] = df.iloc[-1]
        df.loc[pd.to_datetime(head_timestamp)] = df.iloc[0]
        df = df.sort_index()
        return df

    def change_freq(self, df, freq='200ms'):
        return df.asfreq(freq='10ms', method='ffill').asfreq(freq=freq)

    def align_price(self, df, prices):
        pass

def split_by_trade_period(df, ranges):
    res = [df.loc[s:e] for s, e in ranges]
    return res


def update_param(last_param, new_param, keys=None):
    if keys is None:
        keys = ['num', 'target', 'pred_n_steps', 'use_n_steps', 'drop_current']
    changed = False
    for key in keys:
        if last_param[key] != new_param[key]: changed = True
        last_param[key] = new_param[key]
    return changed


def stat_pred(y_true, y_pred, name='stat'):
    return pd.Series({'baseline_mae': y_true.abs().mean(),
                      'baseline_rmse': np.sqrt(np.square(y_true).sum()),
                      'mae': mean_absolute_error(y_true, y_pred),
                      'mse': mean_squared_error(y_true, y_pred),
                      'rmse': mean_squared_error(y_true, y_pred, squared=False),
                      'r2_score': r2_score(y_true, y_pred),
                      'explained_variance_score': explained_variance_score(y_true, y_pred)},
                     name=name)

def std_scale(train,*args):
    res=[]
    def _meta(data:pd.DataFrame,scaler):
        cols=data.columns
        index=data.index
        return pd.DataFrame(scaler.transform(data),index=index,columns=cols)
    scaler = StandardScaler()
    scaler.fit(train)
    train=_meta(train,scaler)
    res.append(train)
    for other in args:
        other=_meta(other,scaler)
        res.append(other)
    return tuple(res)


class ResultSaver(object):
    def __init__(self,param,*args,**kwargs):
        # print(args)
        self.param=param
        self.args=args
        for key, value in kwargs.items():
            self.__setattr__(key,value)


stk_name = '贵州茅台'
symbol = code_dict[stk_name]
trade_details = get_trade_details(data_root, date, symbol)
current = trade_details.set_index('timestamp')['price'].rename('current').groupby(level=0).last()
obh = pd.read_csv(res_root + f'order_book_history_{stk_name}.csv', index_col=0)
obh.index = pd.to_datetime(obh.index)
obh.columns = obh.columns.astype(float)
obh_v = split_volume(obh)
obh_p = split_price(obh)
clean_obh = pd.concat([obh_p, obh_v, current], axis=1).ffill()
clean_obh = del_untrade_time(clean_obh, cut_tail=True).dropna(how='any', axis=0)

clean_obh.to_csv(data_root + f'个股交易细节/clean_obh_{date1}_{stk_name}.csv')

ranges = [(pd.to_datetime(f'{date1} 09:30:00.000'),
           pd.to_datetime(f'{date1} 10:30:00.000') - timedelta(milliseconds=10)),
          (pd.to_datetime(f'{date1} 10:30:00.000'),
           pd.to_datetime(f'{date1} 11:30:00.000') - timedelta(milliseconds=10)),
          (pd.to_datetime(f'{date1} 13:00:00.000'),
           pd.to_datetime(f'{date1} 14:00:00.000') - timedelta(milliseconds=10)),
          (pd.to_datetime(f'{date1} 14:00:00.000'),
           pd.to_datetime(f'{date1} 14:57:00.000') - timedelta(milliseconds=10))]
freq = '200ms'
pp = LOBPreprocessor()
self = LOBFeatureEngineering()


clean_obh = pd.read_csv(res_root + f'clean_obh_{stk_name}.csv', index_col=0)
clean_obh.index = pd.to_datetime(clean_obh.index)

clean_obh = clean_obh.sort_index()
clean_obh = pp.del_untrade_time(clean_obh, cut_tail=True)
clean_obh = pp.add_head_tail(clean_obh)

# note 必须先将clean_obh填充到10ms，否则交易频率是完全不规律的
clean_obh = clean_obh.asfreq(freq='10ms', method='ffill')
clean_obhs = split_by_trade_period(clean_obh, ranges)

clean_obhs = [pp.add_head_tail(cobh, head_timestamp=pd.to_datetime(s),
                               tail_timestamp=pd.to_datetime(e)) for cobh, (s, e) in
              zip(clean_obhs, ranges)]

# todo: 时间不连续、不规整，过于稀疏，归一化细节
features = [self.generate(cobh) for cobh in clean_obhs]
features = [pp.add_head_tail(feature, head_timestamp=pd.to_datetime(s),
                             tail_timestamp=pd.to_datetime(e)) for feature, (s, e) in
            zip(features, ranges)]
features = [pp.change_freq(feature, freq=freq) for feature in features]

clean_obhs = [pp.change_freq(cobh, freq=freq) for cobh in clean_obhs]
alldatas = [pd.merge(feature, cobh, left_index=True, right_index=True) for feature, cobh in zip(features, clean_obhs)]

self = LOBFeatureEngineering()
features = [self.generate(cobh) for cobh in clean_obhs]
# 若只关注上午（为了去掉中午休市带来的误差），每笔交易时间间隔mean=0 days 00:00:00.185008736，median=0 days 00:00:00.110000
# pd.Series(feature.loc[:f'{date1} 11:30:00.01'].index[1:]-feature.loc[:f'{date1} 11:30:00.01'].index[:-1]).describe()

# todo: 时间不连续、不规整，过于稀疏，归一化细节
features = [pp.add_head_tail(feature, head_timestamp=pd.to_datetime(s),
                             tail_timestamp=pd.to_datetime(e) - timedelta(milliseconds=10)) for feature, (s, e) in
            zip(features, ranges)]
features = [pp.change_freq(feature, freq=freq) for feature in features]

clean_obhs = [pp.add_head_tail(cobh, head_timestamp=pd.to_datetime(s),
                               tail_timestamp=pd.to_datetime(e) - timedelta(milliseconds=10)) for cobh, (s, e) in
              zip(clean_obhs, ranges)]
clean_obhs = [pp.change_freq(cobh, freq=freq) for cobh in clean_obhs]

alldatas = [pd.merge(feature, cobh, left_index=True, right_index=True) for feature, cobh in zip(features, clean_obhs)]



# 训练
results={}

last_param = defaultdict(int)
X_train, X_test, y_train, y_test = None, None, None, None
params = ParameterGrid({'num': list(range(4)),
                        # 'model':[('lgbm','rf'),],
                        'pred_n_steps': [300],
                        'use_n_steps': [50],
                        'target': ['ret'],
                        'drop_current': [False]})
for param in params:
    all_y_pred = pd.Series()
    all_stats = pd.DataFrame()
    models = []
    # 只有影响data的参数变了才会更新data
    if update_param(last_param, param):
        print('*' * 20, f'start new {param["num"]}', '*' * 20)
        num = param['num']
        target = param['target']  # ret,current
        pred_n_steps = param['pred_n_steps']  # 预测20s，即100个steps
        use_n_steps = param['use_n_steps']  # 利用use_n_steps个steps的数据去预测pred_n_steps之后的涨跌幅
        drop_current = param['drop_current']  # 是否将当前股价作为因子输入给模型

        current = alldatas[num]['current']
        y = np.log(current.shift(-pred_n_steps) / current)
        y = y.rename(f'{target}_{pred_n_steps * 0.2}s')

        if drop_current:
            X = alldatas[num].drop(columns=['current'])
        else:
            X = alldatas[num]
        X = pd.concat(
            [X.shift(i).rename(columns={col: col + f'_lag{i}' for col in X.columns}) for i in range(use_n_steps)],
            axis=1)

        # x y已经对齐，只需要进行同样的裁剪，去掉nan就可以
        y = y.iloc[use_n_steps:-pred_n_steps]
        X = X.iloc[use_n_steps:-pred_n_steps]

        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.9, random_state=1, shuffle=True)

        X_train,X_test=std_scale(X_train,X_test)
        X_train, X_test, y_train, y_test=X_train.iloc[::use_n_steps], X_test.iloc[::use_n_steps], y_train.iloc[::use_n_steps], y_test.iloc[::use_n_steps]

        # X_train, X_rem, y_train, y_rem = train_test_split(X,y, train_size=0.8, random_state=1, shuffle=True)
        # X_valid, X_test, y_valid, y_test = train_test_split(X_rem,y_rem, test_size=0.5, random_state=1, shuffle=True)
    else:
        print("="*10,"warning: ret not updated","="*10)

    # linear
    # model = LassoCV(cv=5, n_jobs=1, random_state=0)
    # model.fit(X_train, y_train)
    # y_pred = model.predict(X_test)

    # flaml
    automl_settings = {
        "time_budget": 60,  # in seconds
        "metric": 'mse',
        "task": 'regression',
        "log_file_name": model_root + f"{stk_name}_training{num}.log",
        "verbose":1, # int, default=3 | Controls the verbosity, higher means more messages.
    }
    model = AutoML()
    model.fit(X_train, y_train, **automl_settings)
    # Print the best model
    print(f'best model for period {num}', model.model.estimator)
    y_pred = model.predict(X_test)
    models.append(deepcopy(model))

    all_y_pred = pd.concat(
        [all_y_pred, pd.Series(y_pred, index=X_test.index, name=f'pred_{target}_{pred_n_steps * 0.2}s')], axis=0)
    stat = stat_pred(y_test, y_pred,
                     name="{}_period{}".format(re.findall('\w*', str(type(model)).split('.')[-1])[0], num))
    all_stats = pd.concat([all_stats, stat], axis=1)
    # plot_importance(lgbm,max_num_features=20,figsize=(12,10),dpi=320,grid=False)
    rs=ResultSaver(param=param,stk_name=stk_name,num=num,all_y_pred=all_y_pred,all_stats=all_stats,models=model,automl_settings=automl_settings)
    results[str(param)]=rs

    print('*' * 20, f'finish {num}', '*' * 20)

all_stats