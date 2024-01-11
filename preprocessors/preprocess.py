# -*- coding=utf-8 -*-
# @File     : preprocess.py
# @Time     : 2023/8/2 18:32
# @Author   : EvanHong
# @Email    : 939778128@qq.com
# @Project  : 2023.06.08超高频上证50指数计算
# @Description: 需要保证该文件clean，不要加入过多无用算法
import logging
import re
from copy import deepcopy
from datetime import timedelta

import numpy as np
import pandas as pd
from numba import jit
from sklearn.preprocessing import StandardScaler

from backtest.config import *
from backtest import config
from backtest.support import *
from backtest.support import update_date
import pickle
from abc import abstractmethod
from typing import Union, List


class BasePreprocessor(object):
    def __init__(self, *args, **kwargs):
        self.args = args
        for key, value in kwargs.items():
            self.__setattr__(key, value)


class BaseDataPreprocessor(BasePreprocessor):
    """
    主要用于对特征工程后的数据进行进一步预处理，从而适应各个模型
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.scaler = StandardScaler()

    def std_scale(self, data, refit=True, *args):
        res = []

        def _meta(data: pd.DataFrame, scaler):
            cols = data.columns
            index = data.index
            return pd.DataFrame(scaler.transform(data), index=index, columns=cols)

        if refit: self.scaler.fit(data)
        data = _meta(data, self.scaler)
        res.append(data)
        for other in args:
            other = _meta(other, self.scaler)
            res.append(other)
        if len(res) == 1:
            return res[0]
        return tuple(res)

    def sub_illegal_punctuation(self, string):
        return re.sub('\W+', '_', string)

    def save_scaler(self, dir, file_name):
        # file_name=re.sub('\W+','_', file_name)
        with open(dir + file_name, 'wb') as f:
            pickle.dump(self.scaler, f)

    def load_scaler(self, dir, file_name):
        # file_name=re.sub('\W+','_', file_name)
        with open(dir + file_name, 'rb') as f:
            self.scaler = pickle.load(f)

    @classmethod
    def get_flattened_Xy(cls, *args, **kwargs):
        pass

    @abstractmethod
    def get_stacked_Xy(self, *args, **kwargs):
        """
        例如5行X对应1行y
        Parameters
        ----------
        args
        kwargs

        Returns
        -------

        """
        pass

    @abstractmethod
    def align_Xy(self, *args, **kwargs):
        pass


class ShiftDataPreprocessor(BaseDataPreprocessor):
    """
    对feature进行平移拼接，生成可输入模型的数据

    NOTE
    ----
        很可能导致维度爆炸，因为增加使用的数据点的数量，会使得最终特征数呈线性倍数增长

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def shift_X(X, use_n_steps):
        XX = pd.concat(
            [X.shift(i).rename(columns={col: col + f'_lag{i}' for col in X.columns}) for i in
             range(use_n_steps)],
            axis=1)
        return XX

    @classmethod
    def get_flattened_Xy(cls, alldatas, num, target, pred_n_steps, use_n_steps, drop_current):
        if target == Target.ret.name:
            current = alldatas[num]['current']
            y = np.log(current.shift(-pred_n_steps) / current)
            y = y.rename(f'{target}_{pred_n_steps * 0.2}s')
        elif target == Target.mid_p_ret.name:
            mid_price = alldatas[num]['mid_price']
            y = np.log(mid_price.shift(-pred_n_steps) / mid_price)
            y = y.rename(f'{target}_{pred_n_steps * 0.2}s')
        else:
            raise NotImplementedError("align_Xy target not defined")

        if drop_current:
            X = alldatas[num].drop(columns=['current'])
        else:
            X = alldatas[num]

        X = cls.shift_X(X, use_n_steps)

        # x y已经对齐，只需要进行同样的裁剪，去掉nan就可以
        y = y.iloc[use_n_steps:-pred_n_steps]
        X = X.iloc[use_n_steps:-pred_n_steps]

        return X, y


class AggDataPreprocessor(BaseDataPreprocessor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def calc_realized_volatility(series: pd.Series, **kwargs):
        """annualized realized volatility
        realized volatility和ret序列的std不一样在于，std是减去ret的均值，而realized volatility可以看做是减去0

        Notes
        -----
        分母采用样本大小-1，即样本realized volatility

        Parameters
        ----------
        series

        Returns
        -------

        References
        ----------
        [1] https://www.realvol.com/VolFormula.htm
        [2] https://vinsight.shnyu.edu.cn/about_volatility.php
        """
        # assert (series>=0).values.all() # 已在外部代码保证这一点，出于效率考虑，暂时注释该行

        # if kwargs.get('n') is None:
        #     # freq = series.index.freq
        #     logging.warning('AggDataPreprocessor.calc_realized_volatility: freq is none')
        #     freq=(series.index[1:] - series.index[:-1]).median()
        #     n=252*4*60*60/freq.seconds
        # else:
        #     n=kwargs['n']
        n = 1
        temp = series.dropna()
        res = pd.Series(data=[np.sqrt(np.matmul(temp.T, temp) / (len(temp) - 1) * n)], index=[series.index[-1]])
        return res

    @staticmethod
    def realized_volatility(series: pd.Series):
        """RSS, root sum of squares?"""
        raise DeprecationWarning('realized_volatility')
        return np.sqrt(np.sum(series ** 2))

    @staticmethod
    def diff(series: pd.Series):
        return series.iloc[-1] - series.iloc[0]

    @staticmethod
    def _calc_decay_weight(m, H):
        """计算衰减加权权重
        考虑到距离当前时间较远日的数据对于当前因子取值的影响应该更小，而近期数据的影响应该更大，因此我们在加权收益上进行时间半衰的权重倾斜

        Parameters
        ----------
        m:
            收益加权的窗口长度
        H:
            𝐻 为半衰期

        Returns
        -------

        References
        ----------
        [1] 2023-07-14_东方证券_因子选股系列之九十四：UMR2.0，风险溢价视角下的动量反转统一框架再升级.pdf
        """
        weights = np.array([np.power(2, -(m - j + 1) / H) for j in range(1, m + 1)])
        weights = weights / sum(weights)
        return weights

    @classmethod
    def momentum_UMR(cls, features: pd.DataFrame):
        """对原算法进行了简化，去掉了return的部分

        Parameters
        ----------
        features:
            所有特征

        split:
            分为n等分，计算风险调整加权收益

        Returns
        -------

        References
        ----------
        [1] 2023-07-14_东方证券_因子选股系列之九十四：UMR2.0，风险溢价视角下的动量反转统一框架再升级.pdf
        """
        if len(features) == 0: return pd.DataFrame({
            'risk_vol_avg': [np.nan],
            'risk_vol_std': [np.nan],
            'risk_ret_std': [np.nan],
            'risk_ret_skew': [np.nan],
        })
        split = 5  # 分为n等分，计算风险调整加权收益
        H = 2
        agg_rows = step_rows = int(len(features) / split)
        weights = cls._calc_decay_weight(split - 1, H=H)  # 最后一个frame不需要加权
        # _features = features.iloc[::step_rows]
        # returns = _features['wap1_ret']
        # 计算风险系数risk，不同代理变量有不同计算方法
        # risk_vol_avg = features['volume'].rolling(agg_rows, min_periods=agg_rows, step=step_rows).mean() - \
        #                _features['volume']
        # risk_ret_std = (features['wap1_ret']*100000).rolling(agg_rows, min_periods=agg_rows, step=step_rows).std() - \
        #                (_features['wap1_ret']*100000)
        # risk_ret_skew = features['wap1_ret'].rolling(agg_rows, min_periods=agg_rows, step=step_rows).skew() - \
        #                 _features['wap1_ret']
        risk_vol_avg = features['volume'].rolling(agg_rows, min_periods=agg_rows, step=step_rows).mean()
        risk_vol_std = features['volume'].rolling(agg_rows, min_periods=agg_rows, step=step_rows).std()
        risk_ret_std = (features['wap1_ret'] * 100000).rolling(agg_rows, min_periods=agg_rows, step=step_rows).mean()
        risk_ret_skew = (features['wap1_ret'] * 100000).rolling(agg_rows, min_periods=agg_rows, step=step_rows).skew()

        # 对原算法进行了简化，去掉了return的部分
        UMR = pd.DataFrame({
            'risk_vol_avg': [np.sum(weights * risk_vol_avg.iloc[:-1]) - risk_vol_avg.iloc[-1]],
            'risk_vol_std': [np.sum(weights * risk_vol_std.iloc[:-1]) - risk_vol_std.iloc[-1]],
            'risk_ret_std': [np.sum(weights * risk_ret_std.iloc[:-1]) - risk_ret_std.iloc[-1]],
            'risk_ret_skew': [np.sum(weights * risk_ret_skew.iloc[:-1]) - risk_ret_skew.iloc[-1]],
        })
        return UMR

    def agg_features(self, features: pd.DataFrame, use_events=True):
        """
        用于将非常高频的数据agg为高度抽象的特征，如mean、std、realized vol等，并且对agg的大小不同可以构造出动量特征
        Parameters
        ----------
        features
            ['ALO','AMO','ATT','BLO','BMO','BTT','a1_p','a1_v','a21_p_gap','a2_p','a2_v','a32_p_gap','a3_p','a3_v','a43_p_gap','a4_p','a4_v','a54_p_gap','a5_p','a5_v','b1_p','b1_v','b21_p_gap','b2_p','b2_v','b32_p_gap','b3_p','b3_v','b43_p_gap','b4_p','b4_v','b54_p_gap','b5_p','b5_v','bid_ask_volume_ratio1','bid_ask_volume_ratio2','bid_ask_volume_ratio3','bid_ask_volume_ratio4','bid_ask_volume_ratio5','buy_sell_pressure','cum_turnover','cum_vol','current','hi_2','hi_3','hi_4','li_1','li_2','li_3','li_4','mid_price','price_breadth_a','price_breadth_b','relative_spread','spread','spread_tick','turnover','voi','volume','wap1','wap1_c','wap2','wap2_c','wap3','wap3_c','wap4','wap4_c','wap5','wap5_c']
        Returns
        -------

        """

        agg_rows = int(agg_timedelta / min_timedelta)
        step_rows = int(pred_timedelta / min_timedelta)

        agg_mapper = {k: [np.mean, np.std] for k in features.columns}
        agg_diff = {k: [self.diff] for k in ['cum_turnover', 'cum_vol']}
        agg_rv = {k: [self.calc_realized_volatility] for k in ['wap1_ret', 'wap2_ret', 'mid_p_ret']}
        agg_sum = {k: [np.sum] for k in ['ALO', 'AMO', 'ATT', 'BLO', 'BMO', 'BTT']}
        agg_mapper.update(agg_diff)
        agg_mapper.update(agg_rv)
        if use_events:
            agg_mapper.update(agg_sum)

        # 加入realized vol因子，需要注意该df是2 level header
        last_n_rows = [agg_rows, int(agg_rows / 2)]
        # last_n_rows = [agg_rows]
        dfs = []
        for last_n in last_n_rows:
            agg_features = features.rolling(last_n, min_periods=last_n, step=step_rows, closed='left',
                                            center=False).agg(agg_mapper)  # 删除了median
            agg_features.columns = ['_'.join(col) for col in agg_features.columns]

            mom_features = pd.concat([self.momentum_UMR(table) for table in
                                      features.rolling(last_n, min_periods=last_n, step=step_rows, closed='left',
                                                       center=False,
                                                       method='table')])

            mom_features.index = agg_features.index
            temp = pd.concat([agg_features, mom_features], axis=1)
            temp.columns = [col + '_' + str(last_n) for col in temp.columns]
            dfs.append(temp)

        agg_features = pd.concat(dfs, axis=1)

        return agg_features

    @classmethod
    def get_flattened_Xy(cls, alldatas, num, target, pred_n_steps, use_n_steps, drop_current):
        raise NotImplementedError

    def align_Xy(self, X, y, pred_timedelta):
        start_time = X.index
        tar_time = start_time + pred_timedelta

        available_time = [True if t in y.index else False for t in tar_time]
        start_time = start_time[available_time]
        tar_time = tar_time[available_time]

        X = X.loc[start_time]
        y = y.loc[tar_time]

        return X, y


class LobCleanObhPreprocessor(BasePreprocessor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def split_price(order_book_history: pd.DataFrame, window_size=-1):
        """
        将order_book_history分解为price和volume的两个2d dataframes
        :param window_size: 几档盘口信息，10代表买卖各10档
        :param order_book_history:
        :return:
        """

        def meta(row: pd.Series):
            return row.dropna().index.tolist()

        temp = pd.DataFrame(order_book_history.apply(lambda x: meta(x), axis=1, result_type="expand"))
        origin_level = int(temp.shape[1] / 2)
        temp.columns = [f'a{i}_p' for i in range(origin_level, 0, -1)] + [f'b{i}_p' for i in range(1, origin_level + 1)]
        if window_size == -1:
            window_size = origin_level

        levels = [f'a{i}_p' for i in range(window_size, 0, -1)] + [f'b{i}_p' for i in range(1, window_size + 1)]

        temp = temp.loc[:, levels]
        temp.index = order_book_history.index
        return temp

    @staticmethod
    def split_volume(order_book_history: pd.DataFrame, window_size=-1):
        """
        将order_book_history分解为price和volume的两个2d dataframes
        :param window_size: 几档盘口信息，10代表买卖各10档
        :param order_book_history:
        :return:
        """

        def _meta(row: pd.Series):
            return row.dropna().values

        temp = pd.DataFrame(order_book_history.apply(lambda x: _meta(x), axis=1, result_type="expand"))
        origin_level = int(temp.shape[1] / 2)
        temp.columns = [f'a{i}_v' for i in range(origin_level, 0, -1)] + [f'b{i}_v' for i in range(1, origin_level + 1)]
        if window_size == -1:
            window_size = origin_level
        levels = [f'a{i}_v' for i in range(window_size, 0, -1)] + [f'b{i}_v' for i in range(1, window_size + 1)]
        temp = temp.loc[:, levels]
        temp.columns = levels
        temp.index = order_book_history.index
        return temp

    @staticmethod
    def split_volume_price(order_book_history_dict: dict, window_size=-1):
        """
        将order_book_history_dict分解为vol_df,price_df
        Parameters
        ----------
        order_book_history_dict
        window_size

        Returns
        -------

        """
        if window_size == -1:
            window_size = len(list(order_book_history_dict.values())[0])
        volumes = [f'a{i}_v' for i in range(window_size, 0, -1)] + [f'b{i}_v' for i in range(1, window_size + 1)]
        prices = [f'a{i}_p' for i in range(window_size, 0, -1)] + [f'b{i}_p' for i in range(1, window_size + 1)]
        # order_book_history_dict={k: {for kk in order_book_history_dict[k].keys()} for k in order_book_history_dict.items()}
        vol_dict = {k: [vv for vv in v.values()] for k, v in order_book_history_dict.items()}
        price_dict = {k: [vv for vv in v.keys()] for k, v in order_book_history_dict.items()}
        vol_df = pd.DataFrame.from_dict(vol_dict, orient='index')
        price_df = pd.DataFrame.from_dict(price_dict, orient='index')
        # 转置成价格从高到低
        vol_df = vol_df.loc[:, vol_df.columns[::-1]]
        price_df = price_df.loc[:, price_df.columns[::-1]]
        vol_df.columns = volumes
        price_df.columns = prices
        return vol_df, price_df

    # @staticmethod
    # def _gen_clean_obh(datafeed, snapshot_window):
    #     """
    #
    #     :return:
    #     """
    #     # order_book_history = datafeed.order_book_history
    #     # assert len(order_book_history) > 0
    #     # obh = order_book_history
    #     # try:
    #     #     obh.index = pd.to_datetime(obh.index)
    #     # except:
    #     #     obh = obh.T
    #     #     obh.index = pd.to_datetime(obh.index)
    #     # obh = obh.sort_index(ascending=True)
    #     # obh.columns = obh.columns.astype(float)
    #     # obh_v = LobCleanObhPreprocessor.split_volume(obh, window_size=snapshot_window)
    #     # obh_p = LobCleanObhPreprocessor.split_price(obh, window_size=snapshot_window)
    #     obh_p,obh_v=LobCleanObhPreprocessor.split_volume_price()
    #     current = datafeed.current
    #     mid_p = (obh_p[str(LobColTemplate('a', 1, 'p'))] + obh_p[str(LobColTemplate('a', 1, 'p'))]).rename('mid_p') / 2
    #
    #     clean_obh = pd.concat([obh_p, obh_v, current, mid_p], axis=1).ffill().bfill()
    #     clean_obh.index = pd.to_datetime(clean_obh.index)
    #     clean_obh = LobTimePreprocessor.del_untrade_time(clean_obh, cut_tail=True)
    #     clean_obh = LobTimePreprocessor.add_head_tail(clean_obh,
    #                                                   head_timestamp=config.important_times[
    #                                                       'continues_auction_am_start'],
    #                                                   tail_timestamp=config.important_times['continues_auction_pm_end'])
    #
    #     return clean_obh

    # @staticmethod
    # def save_clean_obh(clean_obh, file_root, date, stk_name):
    #     clean_obh.to_csv(file_root + FILE_FMT_clean_obh.format(date, stk_name))
    #
    # @staticmethod
    # def gen_and_save(datafeed, save_root, date: str, stk_name: str, snapshot_window):
    #     clean_obh = LobCleanObhPreprocessor._gen_clean_obh(datafeed, snapshot_window)
    #     LobCleanObhPreprocessor.save_clean_obh(clean_obh, save_root, date, stk_name)

    def run_batch(self):
        """
        批量处理
        :return:
        """
        pass

def str2timedelta( time_str: str, multiplier: int = None) -> Optional[datetime.timedelta]:
    raise PendingDeprecationWarning("该函数将被废弃，请改用`backtest.preprocessors.preprocess.GeneralTimePreprocessor`")
    if time_str is None:
        return timedelta(seconds=0)

    if multiplier is None: multiplier = 1
    time_str_list = time_str.split(' ')
    delta = timedelta(seconds=0)
    for time_str in time_str_list:
        if time_str.endswith('min'):
            td = timedelta(minutes=int(time_str[:-3]) * multiplier)
        elif time_str.endswith('m'):
            td = timedelta(minutes=int(time_str[:-1]) * multiplier)
        elif time_str.endswith('ms'):
            td = timedelta(milliseconds=int(time_str[:-2]) * multiplier)
        elif time_str.endswith('s'):
            td = timedelta(seconds=int(time_str[:-1]) * multiplier)
        else:
            raise NotImplementedError("in config")
        delta += td
    return delta

class GeneralTimePreprocessor(BasePreprocessor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.daily_timedeltas = {
            'open_call_auction_start': timedelta(hours=9, minutes=15),
            'open_call_auction_end': timedelta(hours=9, minutes=25),
            'continues_auction_am_start': timedelta(hours=9, minutes=30),
            'continues_auction_am_end': timedelta(hours=11, minutes=30),
            'continues_auction_pm_start': timedelta(hours=13, minutes=00),
            'continues_auction_pm_end': timedelta(hours=14, minutes=57),
            'close_call_auction_start': timedelta(hours=14, minutes=57),
            'close_call_auction_end': timedelta(hours=15, minutes=00), }

    def str2timedelta(self, time_str: str, multiplier: int = None) -> Optional[datetime.timedelta]:
        if time_str is None:
            return timedelta(seconds=0)

        if multiplier is None: multiplier = 1
        time_str_list = time_str.split(' ')
        delta = timedelta(seconds=0)
        for time_str in time_str_list:
            if time_str.endswith('min'):
                td = timedelta(minutes=int(time_str[:-3]) * multiplier)
            elif time_str.endswith('m'):
                td = timedelta(minutes=int(time_str[:-1]) * multiplier)
            elif time_str.endswith('ms'):
                td = timedelta(milliseconds=int(time_str[:-2]) * multiplier)
            elif time_str.endswith('s'):
                td = timedelta(seconds=int(time_str[:-1]) * multiplier)
            else:
                raise NotImplementedError("in config")
            delta += td
        return delta

    def del_untrade_time_(self, df: Union[pd.Series, pd.DataFrame, List[Union[pd.Series, pd.DataFrame]]],
                          cut_tail=True,
                          strip: list = None,
                          use_default=False
                          ):
        """

        Parameters
        ----------
        df :
        cut_tail :
        strip :
        use_default : bool,
            if true，那就使用默认的strip=['5m', '1m', '1m', '5m']

        Returns
        -------

        """
        if strip is None:
            strip = [None, None, None, None]
        if use_default:
            strip=['5m', '1m', '1m', '5m']
        if cut_tail and strip[3] is None:
            strip[3] = '3m'
        for i in range(len(strip)):
            if strip[i] is None:
                strip[i] = '0s'

        placeholder = '2000-01-01'  # 仅用于构建datetime从而能实现时间运算，
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        index = df.index.time.astype(str)
        idx = np.full(shape=(len(index)), fill_value=True, dtype=bool)
        idx1 = np.full(shape=(len(index)), fill_value=True, dtype=bool)
        # 上下午必须分开，因为两个时间段不能同时用logical_and
        # 上午
        if strip[0] is not None:
            idx = np.logical_and(idx, index >= str(
                (pd.to_datetime(f"{placeholder} 09:30:00") + self.str2timedelta(strip[0])).time()))
        if strip[1] is not None:
            idx = np.logical_and(idx, index <= str(
                (pd.to_datetime(f"{placeholder} 11:30:00") - self.str2timedelta(strip[1])).time()))
        # 下午
        if strip[2] is not None:
            idx1 = np.logical_and(idx1, index >= str(
                (pd.to_datetime(f"{placeholder} 13:00:00") + self.str2timedelta(strip[2])).time()))
        if strip[3] is not None:
            idx1 = np.logical_and(idx1, index <= str(
                (pd.to_datetime(f"{placeholder} 15:00:00") - self.str2timedelta(strip[3])).time()))
        idx = np.logical_or(idx, idx1)
        return df.loc[idx]

    def del_untrade_time(self, df: Union[pd.Series, pd.DataFrame, List[Union[pd.Series, pd.DataFrame]]], cut_tail=True,
                         strip: str = None,
                         drop_last_row=False,
                         split_df=False,
                         pad_margin=False):
        """

        Parameters
        ----------
        drop_last_row :
            去掉最后一行
        df
        cut_tail
            去掉尾盘3min竞价
        strip
            去掉开收盘n min的数据
        split_df
            是否返回上下午分开的df
        pad_margin
            是否要ffill填充上下午开收盘的时刻数据（如填充09:30:00、11:30:00从而保证上午数据在asfreq或是resample之后能够整齐）

        Returns
        -------

        """

        def _meta(df, cut_tail, strip, split_df, pad_margin):
            """
            所有数据遵从ffill的思路
            """
            date_today = pd.to_datetime(df.index[0].date())

            morning_start_time = date_today + self.daily_timedeltas['continues_auction_am_start']
            morning_end_time = date_today + self.daily_timedeltas['continues_auction_am_end']
            afternoon_start_time = date_today + self.daily_timedeltas['continues_auction_pm_start']
            afternoon_end_time = date_today + self.daily_timedeltas[
                'close_call_auction_end'] if not cut_tail else date_today + self.daily_timedeltas[
                'continues_auction_pm_end']
            if strip is not None:
                strip_timedelta = str2timedelta(strip)
                morning_start_time = date_today + self.daily_timedeltas['continues_auction_am_start'] + strip_timedelta
                afternoon_end_time = min(date_today + self.daily_timedeltas['close_call_auction_end'] - strip_timedelta,
                                         afternoon_end_time)

            # 判断几个特殊时间点是否有数据，没有数据则用最近的数据来填充
            a = df.loc[morning_start_time:date_today + self.daily_timedeltas['continues_auction_am_end']]  # 左闭右闭
            b = df.loc[date_today + self.daily_timedeltas['continues_auction_pm_start']:afternoon_end_time]  # 左闭右闭
            if pad_margin:
                # ffill填充上下午开收盘的时刻数据（如填充09: 30:00、11: 30:00从而保证上午数据在asfreq或是resample之后能够整齐）
                pad_morning_start = df.loc[df.index < morning_start_time].iloc[-1].to_frame().T
                pad_morning_end = df.loc[df.index < morning_end_time].iloc[-1].to_frame().T
                pad_afternoon_start = df.loc[df.index < afternoon_start_time].iloc[-1].to_frame().T
                pad_afternoon_end = df.loc[df.index < afternoon_end_time].iloc[-1].to_frame().T
                for i, (pad, dt) in enumerate(
                        [(pad_morning_start, morning_start_time), (pad_morning_end, morning_end_time),
                         (pad_afternoon_start, afternoon_start_time), (pad_afternoon_end, afternoon_end_time)]):
                    if dt in df.index: continue
                    pad.index = pad.index.map(lambda x: dt)
                    if i < 2:
                        a = pd.concat([a, pad], axis=0)
                    elif i >= 2:
                        b = pd.concat([b, pad], axis=0)

            a = a.sort_index()
            b = b.sort_index()
            if split_df:
                if drop_last_row:
                    b = b.iloc[:-1]
                temp = (a, b)
            else:
                temp = pd.concat([a, b], axis=0)
                temp = temp.sort_index()
                if drop_last_row:
                    temp = temp.iloc[:-1]

            return temp

        if isinstance(df, list):
            return [_meta(_df, cut_tail=cut_tail, strip=strip, split_df=split_df, pad_margin=pad_margin) for _df in df]
        elif isinstance(df, pd.DataFrame) or isinstance(df, pd.Series):
            return _meta(df, cut_tail=cut_tail, strip=strip, split_df=split_df, pad_margin=pad_margin)


class LobTimePreprocessor(BasePreprocessor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self.freq = '200ms'
        raise PendingDeprecationWarning(
            "`LobTimePreprocessor`将被淘汰，请使用`backtest.preprocessors.preprocess.GeneralTimePreprocessor`")

    @staticmethod
    def del_middle_hours(df: Union[pd.Series, pd.DataFrame], morning_end: datetime.time = None,
                         afternoon_begin: datetime.time = None, margin='left'):
        """
        去掉日内中间不交易的一段时间，并可以设置参数使得可以去掉中午收盘前一段时间的不可使用数据
        Parameters
        ----------
        df :
        morning_end :
        afternoon_begin :
        margin :
            both 左闭右闭
            left 左闭右开
            right 左开右闭
            none 左开右开

        Returns
        -------

        """
        assert not (morning_end is None and afternoon_begin is None)
        if morning_end is None:
            return df.loc[df.index.time >= afternoon_begin]
        elif afternoon_begin is None:
            return df.loc[df.index.time < morning_end]
        else:
            if margin == 'left':
                return df.loc[(df.index.time < morning_end) | (df.index.time >= afternoon_begin)]
            else:
                raise NotImplementedError()

    @staticmethod
    def del_untrade_time(df: Union[pd.Series, pd.DataFrame, List[Union[pd.Series, pd.DataFrame]]], cut_tail=True,
                         strip: str = None,
                         drop_last_row=False,
                         split_df=False):
        """

        Parameters
        ----------
        df
        cut_tail
            去掉尾盘3min竞价
        strip
            去掉开收盘n min的数据
        split_df
            是否返回上下午分开的df

        Returns
        -------

        """

        def _meta(df, cut_tail, strip, split_df):
            original_date = config.date
            current_yyyy, current_mm, current_dd = str(df.index[0].date()).split('-')
            update_date(current_yyyy, current_mm, current_dd)
            # date_today=df.index[0].date()
            # self.daily_timedeltas

            start_time = config.important_times['continues_auction_am_start']
            end_time = config.important_times['close_call_auction_end'] if not cut_tail else config.important_times[
                'continues_auction_pm_end']
            if strip is not None:
                strip_timedelta = str2timedelta(strip)
                start_time = config.important_times['continues_auction_am_start'] + strip_timedelta
                end_time = min(config.important_times['close_call_auction_end'] - strip_timedelta, end_time)
            a = df.loc[str(start_time):str(config.important_times['continues_auction_am_end'])]
            b = df.loc[str(config.important_times['continues_auction_pm_start']):str(end_time)]
            if split_df:
                a = a.sort_index()
                b = b.sort_index()
                if drop_last_row:
                    b = b.iloc[:-1]
                temp = (a, b)
            else:
                temp = pd.concat([a, b], axis=0)
                temp = temp.sort_index()
                if drop_last_row:
                    temp = temp.iloc[:-1]

            if original_date is not None:
                update_date(original_date[:4], original_date[4:6], original_date[6:])
            else:
                update_date(None, None, None)

            return temp

        if isinstance(df, list):
            return [_meta(_df, cut_tail=cut_tail, strip=strip, split_df=split_df) for _df in df]
        elif isinstance(df, pd.DataFrame) or isinstance(df, pd.Series):
            return _meta(df, cut_tail=cut_tail, strip=strip, split_df=split_df)

    @staticmethod
    def add_head_tail(df, head_timestamp=None, tail_timestamp=None, date_=None, bfill_head=False):
        """

        Parameters
        ----------
        df :
        head_timestamp :
        tail_timestamp :
        date_ :
        bfill_head : bool,
            是否要将第一行用第二行的数据fill

        Returns
        -------

        """
        if head_timestamp is None and tail_timestamp is None and date_ is None:
            raise ValueError(
                f"at least one of the params `date_` and (`head_timestamp`,`tail_timestamp`) should be not None")
        elif head_timestamp is None and tail_timestamp is None:
            # date_ is not none
            head_timestamp = pd.to_datetime(f"{date_} 09:30:00")
            tail_timestamp = pd.to_datetime(f"{date_} 14:57:00")
        else:
            raise ValueError("you should provide (`head_timestamp`,`tail_timestamp`) simultaneously")
        try:
            assert df.index[0] >= head_timestamp and df.index[-1] <= tail_timestamp
        except Exception as e:
            print('add_head_tail', df.index[0], head_timestamp, df.index[-1], tail_timestamp)
            raise e

        res = df.copy(deep=True)
        res.loc[pd.to_datetime(tail_timestamp)] = df.iloc[-1].copy(deep=True)
        if bfill_head:
            res.loc[pd.to_datetime(head_timestamp)] = df.iloc[0].copy(deep=True)
        else:
            res.loc[pd.to_datetime(head_timestamp)] = np.nan
        res = res.sort_index()
        return res

    @staticmethod
    def split_by_trade_period(df):
        res = [df.loc[s:e] for s, e in config.ranges]
        res = [LobTimePreprocessor.add_head_tail(cobh, head_timestamp=pd.to_datetime(s),
                                                 tail_timestamp=pd.to_datetime(e)) for cobh, (s, e) in
               zip(res, config.ranges)]
        return res

    @staticmethod
    def change_freq(df, freq):
        """deprecated"""
        logging.warning("change_freq deprecated", DeprecationWarning)
        return df.asfreq(freq='10ms', method='ffill').asfreq(freq=freq)

    @staticmethod
    def align_price(df, prices):
        pass

    # @staticmethod
    # def split_change_freq(df,freq='200ms'):
    #     dfs=LobTimePreprocessor.split_by_trade_period(df)
    #     clean_obhs


class LobFeatureEngineering(object):
    """计算订单簿截面因子.

    Attributes
    ----------


    """

    def __init__(self, curr='current', ColFormatter=LobColTemplate, limit_ratio=0.2):
        """

        Parameters
        ----------
        curr : str
            最新价列名
        ColFormatter :
        limit_ratio :
            涨跌停
        """
        self.ap = {k: str(ColFormatter('a', k, 'p')) for k in range(1, 11)}
        self.bp = {k: str(ColFormatter('b', k, 'p')) for k in range(1, 11)}
        self.av = {k: str(ColFormatter('a', k, 'v')) for k in range(1, 11)}
        self.bv = {k: str(ColFormatter('b', k, 'v')) for k in range(1, 11)}
        self.curr = curr
        self.limit_ratio = limit_ratio

    def calc_reaches_limit(self, df):
        """
        todo 利用昨收计算涨跌停阈值

        Parameters
        ----------
        df :

        Returns
        -------

        """
        # _open=df[self.curr].iloc[0]
        # upper=_open*
        pass

    def calc_buy_intense(self):
        """
        买入意愿因子《高频因子的现实与幻想》，需要利用9:30开盘后半小时内的数据构建该因子
        :return:
        """
        pass

    def calc_wap(self, df, level, cross=True):
        """Function to calculate level WAP

        References
        ----------
        [1] optiver金牌算法. https://mp.weixin.qq.com/s/Pe4i3I9-ErYFE9uL5B5pvQ
        """
        if cross:
            wap = (df[self.bp[level]] * df[self.av[level]] + df[self.ap[level]] * df[self.bv[level]]) / (
                    df[self.bv[level]] + df[self.av[level]])
        else:
            wap = (df[self.ap[level]] * df[self.av[level]] + df[self.bp[level]] * df[self.bv[level]]) / (
                    df[self.bv[level]] + df[self.av[level]])
        name = f'wap{level}'
        if cross: name += '_cross'
        wap = wap.rename(name)
        wap = wap.replace([np.inf, -np.inf], np.nan)
        return wap

    def calc_cum_wap(self, df, level, cross=True):
        """Function to calculate corresponding weighted average price. 各档位volume*price的加权，而非一档价格按照volume加权

        Notes
        -----
        这个函数可能导致价格疯狂跳动，因为高档位的报撤单是非常频繁的

        References
        ----------
        [1] optiver金牌算法. https://mp.weixin.qq.com/s/Pe4i3I9-ErYFE9uL5B5pvQ
        """
        a_v = 0
        b_v = 0
        a_pv = 0
        b_pv = 0
        for i in range(1, level + 1):
            b_v += df[self.bv[i]]
            a_v += df[self.av[i]]
            if cross:
                a_pv += df[self.ap[level]] * df[self.bv[level]]
                b_pv += df[self.bp[level]] * df[self.av[level]]
            else:
                a_pv += df[self.ap[level]] * df[self.av[level]]
                b_pv += df[self.bp[level]] * df[self.bv[level]]
        wap = (a_pv + b_pv) / (a_v + b_v)
        name = f'cum_wap{level}'
        if cross: name += '_c'
        wap = wap.rename(name)
        wap = wap.replace([np.inf, -np.inf], np.nan)
        return wap

    def calc_cum_vol_wap(self, df, cum_level, cross=True):
        """用对手side累积volume来加权1档价格

        Parameters
        ----------
        df :
        cum_level : int,
            累积level个档位的volume
        cross :

        Returns
        -------

        """
        a_v = 0
        b_v = 0
        for i in range(1, cum_level + 1):
            b_v += df[self.bv[i]]
            a_v += df[self.av[i]]

        if cross:
            wap = (df[self.bp[1]] * a_v + df[self.ap[1]] * b_v) / (a_v + b_v)
        else:
            wap = (df[self.bp[1]] * b_v + df[self.ap[1]] * a_v) / (a_v + b_v)
        name = f'cum_vol_wap{cum_level}'
        if cross: name += '_c'
        wap = wap.rename(name)
        wap = wap.replace([np.inf, -np.inf], np.nan)
        return wap

    def calc_mid_price(self, df, process_zeros=True):
        """

        Parameters
        ----------
        df :
        process_zeros :
            米筐数据可能会在涨停时显示buy price为0.处理逻辑为

        Returns
        -------

        """
        idx_bp_zero = df[self.bp[1]] == 0
        idx_ap_zero = df[self.ap[1]] == 0
        if process_zeros:
            # if np.logical_and(idx_bp_zero,idx_ap_zero).any(): raise ValueError()
            # fill ask price
            idx = np.logical_and(idx_ap_zero, ~idx_bp_zero)
            df.loc[idx, self.ap[1]] = df[self.bp[1]].loc[idx] + 0.01  # add a tick
            # fill bid price
            idx = np.logical_and(idx_bp_zero, ~idx_ap_zero)
            df.loc[idx, self.bp[1]] = df[self.ap[1]].loc[idx] - 0.01
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

    def calc_length_imbalance(self, df, level):
        """
        quantity imbalance

        :return:

        .. [#] Cao, C., Hansch, O., & Wang, X. (2009). The information content of an open limit‐order book. Journal of Futures Markets: Futures, Options, and Other Derivative Products, 29(1), 16-41. 第30页
        """
        QR_level = (df[self.av[level]] - df[self.bv[level]]) / (df[self.av[level]] + df[self.bv[level]])
        QR_level = pd.Series(QR_level, index=df.index, name=f"li_{level}")
        return QR_level

    def calc_height_imbalance(self, df, level):
        """
        和原文有出入，按照原文公式可能会出现inf，因为分母为0.根据文章所表达含义我进行了bid price计算部分的修改

        :return:

        .. [#] Cao, C., Hansch, O., & Wang, X. (2009). The information content of an open limit‐order book. Journal of Futures Markets: Futures, Options, and Other Derivative Products, 29(1), 16-41. 第30页
        """
        assert level >= 2
        # 原文
        # nominator = (df[self.ap[level]] - df[self.ap[level - 1]]) - (df[self.bp[level]] - df[self.bp[level - 1]])
        # denominator = (df[self.ap[level]] - df[self.ap[level - 1]]) + (df[self.bp[level]] - df[self.bp[level - 1]])
        # 修改
        nominator = (df[self.ap[level]] - df[self.ap[level - 1]]) - (df[self.bp[level - 1]] - df[self.bp[level]])
        denominator = (df[self.ap[level]] - df[self.ap[level - 1]]) + (df[self.bp[level - 1]] - df[self.bp[level]])
        HR_level = pd.Series(nominator / denominator, index=df.index, name=f"hi_{level}")
        return HR_level

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
        cum_vol = np.cumsum((series + 1) ** 2)
        denominator = np.arange(1, len(series) + 1, 1)
        realized_vol = (cum_vol / denominator - 1).rename('realized_vol')
        return realized_vol

    def calc_volume_order_imbalance(self, df: pd.DataFrame):
        """
        .. [#] Reference From <Order imbalance Based Strategy in High Frequency Trading>
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

        References
        ----------
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
        idx = np.isinf(bid_d)
        bid_d[idx] = np.nan
        bid_denominator = np.nansum(bid_d, axis=0).reshape(-1, 1).T
        bid_weights = np.divide(bid_d, bid_denominator, where=~np.isnan(bid_denominator))
        press_buy = sum([df[self.bv[i + 1]] * w for i, w in enumerate(bid_weights)])

        ask_d = np.array([M / (df[self.ap[s]] - M) for s in levels])
        # ask_d = [_.replace(np.inf,0) for _ in ask_d]
        idx = np.isinf(ask_d)
        ask_d[idx] = np.nan
        ask_denominator = np.nansum(ask_d, axis=0).reshape(-1, 1).T
        ask_weights = np.divide(ask_d, ask_denominator, where=~np.isnan(ask_denominator))
        press_sell = sum([df[self.av[i + 1]] * w for i, w in enumerate(ask_weights)])

        res = pd.Series((press_buy - press_sell).replace([-np.inf, np.inf], np.nan), name='buy_sell_pressure')
        return res

    def calc_gaps(self, df, level=5):
        """

        :param df:
        :return:

        .. [#] Price jump prediction in Limit Order Book. Ban Zheng, Eric Moulines, Fr ́ed ́eric Abergel https://arxiv.org/pdf/1204.1381.pdf
        """

        def _meta_price_gap(df, i, side: str):
            """price gap"""
            res = None
            if side == 'b':
                res = df[self.bp[i + 1]] - df[self.bp[i]]
            elif side == 'a':
                res = df[self.ap[i + 1]] - df[self.ap[i]]
            return res.rename(f"{side}{i + 1}{i}_p_gap")

        # res = pd.DataFrame()
        l = []
        for i in range(level - 1, 0, -1):
            gap = _meta_price_gap(df, i, 'a')
            l.append(gap)
        for i in range(1, level):
            gap = _meta_price_gap(df, i, 'b')
            l.append(gap)
        res = pd.concat(l, axis=1)
        return res

    # def calc_event_dummies(self, trade_details,order_details):
    #     """
    #     .. [#] Price jump prediction in Limit Order Book. Ban Zheng, Eric Moulines, Fr ́ed ́eric Abergel https://arxiv.org/pdf/1204.1381.pdf
    #
    #     :param events: a Series of integers indicating the type of events
    #     :param df:
    #     :return: one-hot dummies
    #
    #     """
    #     vs = ["BLO", "ALO", "BMO", "AMO", "BTT", "ATT"]
    #     mapper = {k+1: vs[k] for k in range(6)}
    #     event_df = pd.get_dummies(events.apply(lambda x: mapper[x]))
    #     return event_df

    def calc_bid_ask_volume_ratio(self, df: pd.DataFrame, level=5):
        """

        :param df:
        :param level:
        :return:

        .. [x] Price jump prediction in Limit Order Book. Ban Zheng, Eric Moulines,Fr´ed´eric Abergel
        """
        df1 = deepcopy(df)
        cols = [self.av[i] for i in range(1, level + 1)] + [self.bv[i] for i in range(1, level + 1)]
        # df1.loc[:,cols] = np.exp(df1[cols])

        bavr = 1
        res = pd.DataFrame()
        for i in range(1, level + 1):
            bavr *= np.clip(df1[self.bv[i]] / df1[self.av[i]], a_min=-32,
                            a_max=32)  # limit the value between the range of float32
            W_i = pd.Series(bavr, name=f'bid_ask_volume_ratio{i}')
            res = pd.concat([res, W_i], axis=1)

        return res

    def calc_window_avg_order_amount(self):
        """
         一段时间窗口内的平均订单笔数
        :return:
        """
        pass

    def calc_window_avg_trade_amount(self):
        """
         一段时间窗口内的平均成交笔数
        :return:
        """
        pass

    def calc_window_avg_order_vol(self):
        """
         一段时间窗口内的平均订单量
        :return:
        """
        pass

    def calc_window_avg_trade_vol(self):
        """
         一段时间窗口内的平均成交量
        :return:
        """
        pass

    def calc_window_weighted_avg_order_vol(self):
        """
         一段时间窗口内的加权平均订单量
        :return:
        """
        pass

    def calc_window_weighted_avg_trade_vol(self):
        """
         一段时间窗口内的加权平均成交量
        :return:
        """
        pass

    def calc_momentum(self):
        """
         计算一些动量因子
        :return:
        """
        pass

    def generate_cross_section(self, clean_obh, level=5):
        """
        计算截面features，不包含不同时刻数据所构建的因子
        :param clean_obh:
        :param level:
        :return:
        """

        self.mp = self.calc_mid_price(clean_obh)
        # 好像没啥用，因为在10ms的数据中，该列绝大部分都是0。但好像如果这么看，那么所有因子都是稀疏的，因为diff后大部分时间都是0
        self.spread = self.calc_spread(clean_obh)
        self.breadth_b, self.breadth_a = self.calc_price_breadth(clean_obh)
        self.rs = self.calc_relative_spread(clean_obh)
        self.st = self.calc_spread_tick(clean_obh, num_levels=level)
        self.voi = self.calc_volume_order_imbalance(clean_obh)
        self.bsp = self.calc_buy_sell_pressure(clean_obh, level1=1, level2=level, method='MID')
        self.gaps = self.calc_gaps(clean_obh, level=level)
        self.bavr = self.calc_bid_ask_volume_ratio(clean_obh, level=level)

        self.waps = [self.calc_wap(clean_obh, level=i, cross=True) for i in range(1, level + 1)] + [
            self.calc_wap(clean_obh, level=i, cross=False) for i in range(1, level + 1)]
        self.lis = [self.calc_length_imbalance(clean_obh, level=i) for i in range(1, level)]
        self.his = [self.calc_height_imbalance(clean_obh, level=i) for i in range(2, level)]

        self.mp_ret = np.log(self.mp, where=np.logical_and(~np.isnan(self.mp), self.mp > 0)).diff().rename("mid_p_ret")
        self.wap1_ret = np.log(self.waps[0],
                               where=np.logical_and(~np.isnan(self.waps[0]), self.waps[0] > 0)).diff().rename(
            "wap1_ret")
        self.wap2_ret = np.log(self.waps[1],
                               where=np.logical_and(~np.isnan(self.waps[1]), self.waps[1] > 0)).diff().rename(
            "wap2_ret")

        self.features = pd.concat(
            self.waps
            + self.lis
            + self.his
            + [self.mp,
               self.spread,
               self.breadth_b,
               self.breadth_a,
               self.rs,
               self.st,
               self.voi,
               self.bsp,
               # self.volatility, # 会逐渐减小为0
               self.gaps,
               self.bavr,
               self.mp_ret,
               self.wap1_ret,
               self.wap2_ret,
               ],
            axis=1)
        # fixme 将2 level header转为1 level
        # df_feature.columns = ['_'.join(col) for col in df_feature.columns]  # time_id is changed to time_id_

        return self.features

    # def agg_features(self, features: pd.DataFrame, agg_freq: str):
    #     features = features.resample(agg_freq).agg([np.mean, np.std, np.median])
    #     return features


class ImbalancedDataPreprocessor(BaseDataPreprocessor):
    """
    References
    ----------
    [1] Werner de Vargas, V., Schneider Aranda, J.A., dos Santos Costa, R. et al. Imbalanced data preprocessing techniques for machine learning: a systematic mapping study. Knowl Inf Syst 65, 31–57 (2023). https://doi.org/10.1007/s10115-022-01772-8
    """

    def undersampling(self):
        pass

    def oversampling(self):
        pass

    def hybrid_sampling(self):
        pass


if __name__ == '__main__':
    pass
