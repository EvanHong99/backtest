# -*- coding=utf-8 -*-
# @File     : preprocess.py
# @Time     : 2023/8/2 18:32
# @Author   : EvanHong
# @Email    : 939778128@qq.com
# @Project  : 2023.06.08超高频上证50指数计算
# @Description: 需要保证该文件clean，不要加入过多无用算法
import re
from copy import deepcopy
from datetime import timedelta

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from config import *
from support import LobColTemplate, Target
import pickle
from abc import abstractmethod


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
        self.scaler = None

    def std_scale(self, train, *args):
        res = []

        def _meta(data: pd.DataFrame, scaler):
            cols = data.columns
            index = data.index
            return pd.DataFrame(scaler.transform(data), index=index, columns=cols)

        self.scaler = StandardScaler()
        self.scaler.fit(train)
        train = _meta(train, self.scaler)
        res.append(train)
        for other in args:
            other = _meta(other, self.scaler)
            res.append(other)
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
        # todo 把该处理单独做函数
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

    def agg_X(self, X, use_n_steps):
        pass

    @classmethod
    def get_flattened_Xy(cls, alldatas, num, target, pred_n_steps, use_n_steps, drop_current):
        pass


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

        def meta(row: pd.Series):
            return row.dropna().values

        temp = pd.DataFrame(order_book_history.apply(lambda x: meta(x), axis=1, result_type="expand"))
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
    def _gen_clean_obh(dataloader):
        """

        :return:
        """
        order_book_history = dataloader.order_book_history
        current = dataloader.current
        assert len(order_book_history) > 0
        obh = order_book_history
        try:
            obh.index = pd.to_datetime(obh.index)
        except:
            obh = obh.T
            obh.index = pd.to_datetime(obh.index)
        obh = obh.sort_index(ascending=True)
        obh.columns = obh.columns.astype(float)
        obh_v = LobCleanObhPreprocessor.split_volume(obh)
        obh_p = LobCleanObhPreprocessor.split_price(obh)

        clean_obh = pd.concat([obh_p, obh_v, current], axis=1).ffill().bfill()
        clean_obh.index = pd.to_datetime(clean_obh.index)
        clean_obh = LobTimePreprocessor.del_untrade_time(clean_obh, cut_tail=True)
        clean_obh = LobTimePreprocessor.add_head_tail(clean_obh,
                                                      head_timestamp=important_times['continues_auction_am_start'],
                                                      tail_timestamp=important_times['continues_auction_pm_end'])

        return clean_obh

    @staticmethod
    def save_clean_obh(clean_obh, file_root, date, stk_name):
        clean_obh.to_csv(file_root + FILE_FMT_clean_obh.format(date, stk_name))

    @staticmethod
    def gen_and_save(dataloader, save_root, date: str, stk_name: str):
        clean_obh = LobCleanObhPreprocessor._gen_clean_obh(dataloader)
        LobCleanObhPreprocessor.save_clean_obh(clean_obh, save_root, date, stk_name)

    def run_batch(self):
        """
        批量处理
        :return:
        """
        pass


class LobTimePreprocessor(BasePreprocessor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self.freq = '200ms'

    @staticmethod
    def del_untrade_time(df, cut_tail=True):
        """

        :param df:
        :param cut_tail: 去掉尾盘3min竞价
        :return:
        """
        end_time = important_times['close_call_auction_end'] if not cut_tail else important_times[
            'continues_auction_pm_end']
        a = df.loc[important_times['continues_auction_am_start']:important_times['continues_auction_am_end']]
        b = df.loc[important_times['continues_auction_pm_start']:end_time]
        temp = pd.concat([a, b], axis=0)

        temp = temp.sort_index()
        return temp

    @staticmethod
    def add_head_tail(df, head_timestamp=pd.to_datetime(f'{date1} 09:30:00.000'),
                      tail_timestamp=pd.to_datetime(f'{date1} 14:57:00.000')):
        # df=df.dropna(how='any',axis=0)
        try:
            # print(df.index.dtype, df)
            assert df.index[0] >= head_timestamp and df.index[-1] <= tail_timestamp
        except Exception as e:
            print('add_head_tail', df.index[0], head_timestamp, df.index[-1], tail_timestamp)
            raise e
        df.loc[pd.to_datetime(tail_timestamp)] = df.iloc[-1]
        df.loc[pd.to_datetime(head_timestamp)] = df.iloc[0]
        df = df.sort_index()
        return df

    @staticmethod
    def split_by_trade_period(df):
        res = [df.loc[s:e] for s, e in ranges]
        res = [LobTimePreprocessor.add_head_tail(cobh, head_timestamp=pd.to_datetime(s),
                                                 tail_timestamp=pd.to_datetime(e)) for cobh, (s, e) in zip(res, ranges)]
        return res

    @staticmethod
    def change_freq(df, freq='200ms'):
        return df.asfreq(freq='10ms', method='ffill').asfreq(freq=freq)

    @staticmethod
    def align_price(df, prices):
        pass

    # @staticmethod
    # def split_change_freq(df,freq='200ms'):
    #     dfs=LobTimePreprocessor.split_by_trade_period(df)
    #     clean_obhs


class LobFeatureEngineering(object):
    """
    todo: 买入意愿因子《高频因子的现实与幻想》，需要利用9:30开盘后半小时内的数据构建该因子
    """

    def __init__(self):
        self.ap = {k: str(LobColTemplate('a', k, 'p')) for k in range(1, 11)}
        self.bp = {k: str(LobColTemplate('b', k, 'p')) for k in range(1, 11)}
        self.av = {k: str(LobColTemplate('a', k, 'v')) for k in range(1, 11)}
        self.bv = {k: str(LobColTemplate('b', k, 'v')) for k in range(1, 11)}
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

    def calc_depth_imbalance(self, df):
        """

        :return:

        .. [#] Cao, C., Hansch, O., & Wang, X. (2009). The information content of an open limit‐order book. Journal of Futures Markets: Futures, Options, and Other Derivative Products, 29(1), 16-41.
        """
        pass

    def calc_height_imbalance(self, df):
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

    def calc_gaps(self, df, events: pd.Series, level=5):
        """

        :param events: a Series of integers indicating the type of events
        :param df:
        :return:

        .. [#] Price jump prediction in Limit Order Book. Ban Zheng, Eric Moulines, Fr ́ed ́eric Abergel https://arxiv.org/pdf/1204.1381.pdf
        """
        assert len(df) == len(events)

        def meta_G_price(df, i, side: str):
            """price gap"""
            res = None
            if side == 'b':
                res = df[self.bp[i + 1]] - df[self.bp[i]]
            elif side == 'a':
                res = df[self.ap[i + 1]] - df[self.ap[i]]
            return res.rename(f"{side}{i + 1}{i}_p_gap")

        res = pd.DataFrame()
        for i in range(level - 1, 0):
            gap = meta_G_price(df, i, 'a')
            res = pd.concat([res, gap], axis=1)

        for i in range(1, level):
            gap = meta_G_price(df, i, 'b')
            res = pd.concat([res, gap], axis=1)
        return res

    def calc_event_dummies(self, events: pd.Series, level=5):
        """
        .. [#] Price jump prediction in Limit Order Book. Ban Zheng, Eric Moulines, Fr ́ed ́eric Abergel https://arxiv.org/pdf/1204.1381.pdf

        :param events: a Series of integers indicating the type of events
        :param df:
        :return:

        """
        vs = ["BLO", "ALO", "BMO", "AMO", "BTT", "ATT"]
        mapper = {k: vs[k - 1] for k in range(1, level + 2)}
        event_df = pd.get_dummies(events.apply(lambda x: mapper[x]))
        return event_df

    def calc_bid_ask_volume_ratio(self, df: pd.DataFrame, level=5):
        df1 = deepcopy(df)
        df1[[self.av[i] for i in range(1, level + 1)] + [self.bv[i] for i in range(1, level + 1)]] = np.exp(
            df1[[self.av[i] for i in range(1, level + 1)] + [self.bv[i] for i in range(1, level + 1)]])
        cum_b = 0
        cum_a = 0

        res = pd.DataFrame()
        for i in range(1, level + 1):
            cum_b += df1[self.bv[1]]
            cum_a += df1[self.av[1]]
            W_i = pd.Series(np.log(cum_b / cum_a), name=f'bid_ask_volume_ratio{i}')
            res = pd.concat([res, W_i], axis=1)

        return W_i

    def calc_window_avg_order_amount(self):
        """
        todo 一段时间窗口内的平均订单笔数
        :return:
        """
        pass

    def calc_window_avg_trade_amount(self):
        """
        todo 一段时间窗口内的平均成交笔数
        :return:
        """
        pass

    def calc_window_avg_order_vol(self):
        """
        todo 一段时间窗口内的平均订单量
        :return:
        """
        pass

    def calc_window_avg_trade_vol(self):
        """
        todo 一段时间窗口内的平均成交量
        :return:
        """
        pass

    def calc_window_weighted_avg_order_vol(self):
        """
        todo 一段时间窗口内的加权平均订单量
        :return:
        """
        pass

    def calc_window_weighted_avg_trade_vol(self):
        """
        todo 一段时间窗口内的加权平均成交量
        :return:
        """
        pass

    def calc_momentum(self):
        """
        todo 计算一些动量因子
        :return:
        """
        pass

    def generate(self, clean_obh):
        self.mp = self.calc_mid_price(clean_obh)
        self.waps = [self.calc_wap(clean_obh, level=i, cross=False) for i in range(1, 6)] + [
            self.calc_wap(clean_obh, level=i, cross=True) for i in range(1, 6)]
        self.spread = self.calc_spread(clean_obh)
        self.breadth_b, self.breadth_a = self.calc_price_breadth(clean_obh)
        self.rs = self.calc_relative_spread(clean_obh)
        self.st = self.calc_spread_tick(clean_obh, num_levels=5)
        self.voi = self.calc_volume_order_imbalance(clean_obh)
        self.bsp = self.calc_buy_sell_pressure(clean_obh, level1=1, level2=5, method='MID')

        # self.volatility=self.calc_realized_volatility() # 计算ret的累积volatility

        # self.di=self.calc_depth_imbalance(clean_obh_dict)
        # self.hi=self.calc_height_imbalance(clean_obh_dict)

        # self.gaps=self.calc_gaps(clean_obh_dict,events=None,level=5)
        # self.ed=self.calc_event_dummies(events=None,level=5)
        # self.bavr=self.calc_bid_ask_volume_ratio(clean_obh_dict, level=5)

        # # use this
        # self.features = pd.concat(
        #     self.waps + [self.mp, self.spread, self.breadth_b, self.breadth_a, self.rs, self.st, self.voi, self.bsp],
        #     axis=1)

        # delete below
        self.features = pd.concat(
            self.waps + [ self.spread, self.breadth_b, self.breadth_a, self.rs, self.st, self.voi, self.bsp],
            axis=1)
        return self.features


if __name__ == '__main__':
    pass