import logging

import numpy as np
import pandas as pd
import json
from dateutil import relativedelta
from math import sqrt
import datetime as dt  # import date
from datetime import datetime, time, timedelta
import time as t
import matplotlib.pyplot as plt
from typing import Optional, Union, List
from copy import deepcopy
from typing_extensions import (
    Literal,
)  # typing_extensions is used for using Literal from python 3.7
from backtest.preprocessors.preprocess import LobFeatureEngineering

'''
%load_ext autoreload
%autoreload 2

jupyter notebook:
%%prun -T ./profiling.txt -D .profiling_stats.txt
'''


def init():
    import pandas as pd
    import numpy as np
    from collections import defaultdict
    from copy import deepcopy
    import pickle
    from datetime import datetime, time, timedelta
    import logging
    import os
    from matplotlib import pyplot as plt
    # plt.rcParams['font.sans-serif'] = ['SimHei']  # windows用来正常显示中文标签
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # mac用来正常显示中文标签
    # plt.rcParams['figure.figsize'] = [12, 10]
    # plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


def login():
    # 登录同花顺账户
    with open('config.json', 'r') as openfile:
        account_info = json.load(openfile)
    # 输入用户的帐号和密码
    thsLogin = THS_iFinDLogin(account_info['account'], account_info['pwd'])
    # print(thsLogin)
    if thsLogin != 0:
        print('THS登录失败')
    else:
        print('THS登录成功')


def get_data(func, param, date: str, max_try=3):
    counter = 0
    temp = func('212001', param, date)
    while temp.errorcode != 0 and counter < max_try:
        print(temp.errmsg, 'retry', counter)
        t.sleep(0.5)
        temp = func('212001', param, date)
        counter += 1
    if temp.errorcode != 0:
        print('>' * 10, 'get_data error', '<' * 10)
    return temp.data


def get_THS_BD(codes, fields, params, max_try=3):
    counter = 0
    temp = THS_BD(codes, fields, params)
    while temp.errorcode != 0 and counter < max_try:
        print(temp.errmsg, 'retry', counter)
        t.sleep(0.5)
        temp = THS_BD(codes, fields, params)
        counter += 1
    data = temp.data
    if temp.errorcode != 0:
        print('>' * 10, 'get_THS_BD error', '<' * 10)
    return data


def get_THS_HQ(codes, fields, params, start_date, end_date, max_try=3):
    counter = 0
    temp = THS_HQ(codes, fields, params, start_date, end_date)
    while temp.errorcode != 0 and counter < max_try:
        print(temp.errmsg, 'retry', counter)
        t.sleep(0.5)
        temp = THS_HQ(codes, fields, params, start_date, end_date)
        counter += 1
    data = temp.data
    if temp.errorcode != 0:
        print('>' * 10, 'get_THS_HQ error', '<' * 10)
    return data


def get_THS_tradeday(start, end=None, freq='D', firstday=False):
    '''
    同花顺交易日历接口
    :params start:日历开始时间 YYYY-MM-DD
    :params end:日历结束时间,默认为今天 YYYY-MM-DD
    :params freq:日历的频率,默认为D
    '''
    if type(end) == type(None):
        end = dt.date.today().strftime("%Y-%m-%d")

    # 交易日历
    tradeCalender = THS_DateQuery('SSE'
                                  , f'dateType:0,period:{freq},dateFormat:0'
                                  , start
                                  , end
                                  )

    if tradeCalender['errorcode'] == 0:
        tradeday_list = tradeCalender['tables']['time']
        return np.array(tradeday_list)
    else:
        print('error')


# 收益率统计
def statistics(df, freq='D', risk_free_rate=0.03):
    print(f"净值序列长度：{len(df)}")
    df.index = pd.to_datetime(df.index)

    # 总收益
    return_total = df.iloc[-1] - 1
    return_total.rename('总收益', inplace=True)

    # 年化波动率
    if freq == 'D':
        volatility_annual = (df / df.shift(1) - 1).std() * sqrt(250)
    elif freq == 'M':
        volatility_annual = (df / df.shift(1) - 1).std() * sqrt(12)
    volatility_annual.rename('年化波动率', inplace=True)

    # 年化收益率
    delta = relativedelta.relativedelta(df.index[-1], df.index[0])
    length = delta.years + delta.months / 12
    return_annual = (return_total + 1) ** (1 / length) - 1

    return_annual.rename('年化收益率', inplace=True)

    # 夏普比率
    sharpeRatio = (return_annual - risk_free_rate) / volatility_annual
    sharpeRatio.rename('夏普比率', inplace=True)

    # 最大回撤
    temp = (np.maximum.accumulate(df) - df) / np.maximum.accumulate(df)
    maxDrawdown = (temp).max()
    maxDrawdown.rename('最大回撤', inplace=True)

    res = pd.concat([return_total, volatility_annual, return_annual, sharpeRatio, maxDrawdown], axis=1).T
    res.loc['最大回撤日期'] = pd.to_datetime(temp.idxmax()[0]).date()
    # 汇总成表
    return res


def do_MonteCarlo(res, iteratioin=1000, risk_free_rate=0.02) -> np.ndarray:
    best_weight = None
    best_sharpe = -1e7
    covariance = res.cov()
    for i in range(iteratioin):
        weights = np.random.random(len(res.columns))
        weights /= np.sum(weights)

        weighted_daily_return = (((res + 1).cumprod().iloc[-1] - 1) *
                                 weights).sum() / len(res.columns)
        if i == 0:
            pass
        weighted_std = np.sqrt(
            np.matmul(np.matmul(weights, covariance), weights.T))
        sharpe = (weighted_daily_return * 250 -
                  risk_free_rate) / (weighted_std * np.sqrt(250))
        if sharpe > best_sharpe:
            best_weight = weights
            best_sharpe = sharpe
    codes = res.columns
    if best_weight is None:
        print('do_MonteCarlo')
        print(res)
        print('res.columns', res.columns)
    return best_weight, best_sharpe, codes


def calc_weight(codes, now):
    '''
    马科维茨投资组合原理计算资产权重
    '''

    changeRatio = w.wsd(codes,
                        "pct_chg",
                        str((now - timedelta(60)).date()),
                        str(now),
                        "PriceAdj=B",
                        usedf=True)[1] / 100

    changeRatio = changeRatio.dropna(axis=1, how='all').fillna(0)
    #         if len(temp.columns)!=len(changeRatio.columns):

    best_weight, best_sharpe, codes = do_MonteCarlo(changeRatio,
                                                    iteratioin=1000)
    if best_weight is None:
        raise Exception('len(best_weight)!=len(changeRatio.columns)')
    return pd.Series(best_weight, index=codes)


# def calc_net_value(data,last_net_value):
#     '''
#     param: data: ths changeRatio
#     '''
#     data=data.fillna(0)/100+1
#     data.iloc[0]=data.iloc[0]*last_net_value
#     temp=data.cumprod()
#     return temp.mean(axis=1)

def calc_net_value(data, last_net_value, control_drawdown=False, drawdown_threshold=-0.1, level='portfolio',
                   weight=None):
    '''
    每个调仓周期内单独计算净值以及止损
    param: data, changeRatio
    param: last_net_value, 上一期末组合净值
    '''

    def weighted_net_value(data, weight):
        if data.isna().any().any() or weight.isna().any():
            print("warning !!!! calc_net_value weighted_net_value data,weight has nan")
        if weight is not None:
            net_value = data.apply(lambda x: np.matmul(x, np.array(weight).T), axis=1)
        else:
            net_value = data.mean(axis=1)

        return net_value

    stoploss = None  # 记录下跌幅度

    if len(data.columns) == 0:
        # 用nan填补没有持仓的日子，之后ffill
        print('calc_net_value: no holding codes')
        index = pd.to_datetime(pd.date_range(start=last_trade_date, end=trade_date, freq='D').date)
        return pd.DataFrame(data=[last_net_value] * len(index), index=index), stoploss

    data = data.fillna(0) / 100 + 1
    data.iloc[0] = data.iloc[0] * last_net_value
    temp = data.cumprod()

    if control_drawdown:
        if level == 'stock':
            test = deepcopy(temp)
            premaximum = test.cummax()
            stoploss = ((test - premaximum) / premaximum).cummin()
            if_stoploss = stoploss < drawdown_threshold
            for col in if_stoploss.columns:
                test.loc[if_stoploss.loc[:, col], col] = np.nan
            test = test.ffill()
            net_value = weighted_net_value(test, weight)

        if level == 'portfolio':
            net_value = weighted_net_value(temp, weight)
            test = deepcopy(net_value)
            premaximum = test.cummax()
            stoploss = ((test - premaximum) / premaximum).cummin()
            if_stoploss = stoploss < drawdown_threshold
            test.loc[if_stoploss] = np.nan
            net_value = test.ffill()
    else:
        net_value = weighted_net_value(temp, weight)

    net_value.index = pd.to_datetime(net_value.index).date.tolist()

    return net_value, stoploss


def get_data(func, param, date: str, max_try=3):
    counter = 0
    temp = func('212001', param, date)
    while temp.errorcode != 0 and counter < max_try:
        print(temp.errmsg, 'retry', counter)
        t.sleep(0.5)
        temp = func('212001', param, date)
        counter += 1
    if temp.errorcode != 0:
        print('>' * 10, 'get data error', '<' * 10)
    return temp.data


def calc_beta_alpha(data):
    # beta and alpha
    returns = data.resample('Y').last()
    returns.loc[pd.to_datetime('2012-12-31')] = 1
    returns.sort_index(inplace=True)
    returns = returns.pct_change().dropna()
    X = sm.add_constant(returns['benchmark'])
    # Fit a simple linear regression model of investment returns on benchmark returns
    model = sm.OLS(returns['portfolio'], X)
    results = model.fit()
    # Extract the alpha coefficient from the regression results
    alpha = results.params[0]
    beta = results.params[1]
    print(f'beta:{beta}, alpha:{alpha}')


def take_profit(changeRatio):
    # 个股级别止盈
    test = (changeRatio + 1).cumprod()
    if takeprofitMode == True:
        for col in test.columns:
            # DataFrame.idxmax(axis=0, skipna=True, numeric_only=False)
            # Return index of first occurrence of maximum over requested axis.
            index1 = (test[col] > 1.1).idxmax()  # max是指true
            index2 = (test[col] > 1.15).idxmax()
            index3 = (test[col] > 1.2).idxmax()
            if index1 != test[col].index[0]:
                num = test[col].loc[index1]
                test[col][test[col].index > index1] = test[col][
                                                          test[
                                                              col].index > index1] * 0.8 + num * 0.2  # 涨幅到达10%，卖掉20%仓位，因此之后的股价波动减小，仅为原始值的0.8，并且0.2的当前价格（即收益）成为定值，被包含在资产中
            # 上面已经将test之后的值都重新覆盖了（即已经止盈），因此之后的num是相对于当时的价格而言的，其大小为 1-真实持仓收缩比率
            # 但是持仓波动率则是与之前相关的，因为是卖出到指定总仓位
            if index2 != test[col].index[0]:  # 若该判断成立，则index1必然成立，即之前已经卖掉了20%的仓位
                num = test[col].loc[index2]
                test[col][test[col].index > index2] = test[col][
                                                          test[
                                                              col].index > index2] * 0.625 + num * 0.375  # 涨幅达到15%，卖掉总共50%的仓位。在已经卖出20%的基础上，持仓收益波动率需要进一步降低至0.5，而之前已经降低至0.8，因此此处需要再降低0.625,0.625*0.8=0.5。而0.375=1-0.625
            if index3 != test[col].index[0]:
                num = test[col].loc[index3]
                test[col][test[col].index > index3] = num


def get_basic_info(codes, fields='ths_stock_short_name_stock;ths_corp_nature_stock;ths_the_sw_industry_stock',
                   date=None):
    return get_THS_BD(codes, fields, f';{date};100,{date}').data


def plot_legend_outside(df: Union[pd.Series,pd.DataFrame,np.ndarray],
                        save_path=None,
                        baseline: Union[pd.Series,pd.DataFrame] = None,
                        linewidth=0.8,
                        xticks: Union[list,np.array] = None,
                        shrink: Union[bool,int] = 1000,
                        plot_uniform_timestamp=True,
                        title=None,
                        marker='',
                        linestyle='-',
                        markersize=3,
                        horizontal_line_values:Union[List[int],List[float],list]=None,
                        horizontal_line_marker='',
                        horizontal_line_linestyle='--',
                        horizontal_line_color='r',
                        ):
    """将legend放到图片外部

    Parameters
    ----------
    df :
    save_path :
        若非nan则会保存该图片
    baseline :
        基准benchmark
    linewidth :
    xticks : deprecated
        用来减少横轴显示的时间戳
    shrink : bool or int,
        if int, 将df缩减到shrink行。 if True, 缩放到最多10000行
    plot_uniform_timestamp : bool,
        将时间戳视为均匀的点，而不是按照时间戳的间隔来画图。设置为true可以跳过午盘。该参数能大幅提高画图效率
    linestyle : str,
        '-': solid line, '--': dashed line, '': no line
    marker : str,
        'o': circles, '.': dots, '': no marker
    horizontal_line_values : Union[List[int],List[float],list],
        if not none，则会在图中画指示性作用的dashed横线

    Returns
    -------

    """
    if isinstance(df, pd.Series): df = df.to_frame()
    df.index = df.index.astype(str)

    # 如果数据太多，直接舍弃部分df数据来画图，保证整幅图片总共数据点
    if shrink:  # True/int
        steps = min(len(df), 10000)
        if not isinstance(shrink, bool):
            steps = min(len(df), shrink)
        step = len(df) // steps
        df = df.iloc[::step]

    # for col in df.columns:
    #     plt.plot(df.index, df[col], label=col, linewidth=linewidth)

    plt.plot(df.index, df, label=df.columns, linewidth=linewidth, marker=marker, linestyle=linestyle,
             markersize=markersize)

    if baseline is not None:
        baseline_name = 'baseline'
        if isinstance(baseline, pd.DataFrame):
            baseline_name = baseline.columns[0]
        elif isinstance(baseline, pd.Series):
            baseline_name = baseline.name
        plt.plot(baseline, label=baseline_name, linewidth=linewidth)
    if xticks is not None:
        length = len(xticks)
        if length > 50:
            steps = 10
            step = length // steps
            plt.xticks(range(0, length, step), xticks[::step], rotation=70)
        else:
            plt.xticks(range(length), xticks)
    if plot_uniform_timestamp:
        xticks = df.index
        length = len(xticks)
        if length > 50:
            steps = 10
            step = length // steps
            plt.xticks(range(0, length, step), xticks[::step], rotation=70)
        else:
            plt.xticks(range(length), xticks)

    if title is not None:
        plt.title(title)
    plt.legend(bbox_to_anchor=(1.0, 1.0))

    if horizontal_line_values is not None:
        for y in horizontal_line_values:
            plt.axhline(y=y,linestyle=horizontal_line_linestyle,marker=horizontal_line_marker,color=horizontal_line_color)


    if save_path is not None:
        plt.savefig(save_path, dpi=320, bbox_inches='tight')
    plt.show()


def dataframe_subplot_legend_outside():
    temp = pd.concat([true_value, fit, IO2207C4450, IO2207P4450, IF2207], axis=1)
    delta = timedelta(seconds=window / 100)  # (window/100)s
    for t in jump_time:
        win_s = t - 3 * delta
        win_e = t + 3 * delta
        axs = temp.loc[win_s:win_e].plot(title=str(t), legend=False,
                                         subplots=[(['current_filled']), ([fitted_index_name]),
                                                   ('f_a1_p_filled', 'f_curr_filled', 'f_b1_p_filled'),
                                                   ('opt_C_a1_p_filled', 'opt_C_curr_filled', 'opt_C_b1_p_filled'),
                                                   ('opt_P_a1_p_filled', 'opt_P_curr_filled', 'opt_P_b1_p_filled')])
        plt.xticks([t, t - delta], labels=['jump', 'base'])
        for ax in axs:
            ax.legend(bbox_to_anchor=(1.0, 1.0))
        plt.show()


def read_hdf5():
    order_f = h5py.File(data_root + f'{date}.orders.XSHG.h5', 'r')
    code_dict = {'沪深300ETF': '510300.XSHG', '中国平安': '601318.XSHG'}
    symbol = code_dict['中国平安']

    order_dset = order_f[symbol]
    order_details = pd.DataFrame.from_records(order_dset[:], index='seq')
    order_details['timestamp'] = pd.to_datetime(order_details['timestamp'], format='%Y%m%d%H%M%S%f')
    order_details['last_traded_timestamp'] = pd.to_datetime(order_details['last_traded_timestamp'],
                                                            format='%Y%m%d%H%M%S%f')
    order_details['canceled_timestamp'] = pd.to_datetime(order_details['canceled_timestamp'], format='%Y%m%d%H%M%S%f')
    order_details['price'] = order_details['price'] / 10000
    order_details['filled_amount'] = order_details['filled_amount'] / 10000


def clip_inf(df: Union[pd.DataFrame, pd.Series]):
    # idx=np.isinf(df)
    # replace infinite value with the second extreme value
    df = df.replace(np.inf, df.replace([np.inf], np.nan).max().to_dict())
    df = df.replace(-np.inf, df.replace([-np.inf], np.nan).min().to_dict())
    return df


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

    if kwargs.get('n') is None:
        # freq = series.index.freq
        # logging.warning('AggDataPreprocessor.calc_realized_volatility: freq is none')
        freq = (series.index[1:] - series.index[:-1]).median()  # 根据数据自带的index来计算freq
        n = 252 * 4 * 60 * 60 / freq.seconds
    else:
        n = kwargs['n']
    # n = 1
    temp = series.dropna()
    res = pd.Series(data=[np.sqrt(np.matmul(temp.T, temp) / (len(temp) - 1) * n)], index=[series.index[-1]])
    return res


def calc_rv(df, window=10):
    """

    Parameters
    ----------
    df :
    window : int,
        rolling size

    Returns
    -------

    """
    close = df['close']
    ret = np.log(close).diff()
    rv = ret.rolling(window=window).apply(calc_realized_volatility)
    return rv


def calc_date2maturity(current: Union[str, datetime, np.datetime64], expiry: Union[str, datetime, np.datetime64],
                       trading_dates: Union[np.ndarray, List[str]] = None, including_right=False):
    """用于计算current距离expiry在中国交易日历中的天数

    Parameters
    ----------
    current : str,
        当前日期
    expiry : str,
        到期日
    trading_dates : np.ndarray,
        预先存成csv的数据。`trading_dates_cn_2000-01-01_2024-12-31.csv`
    including_right : bool,
        是否包括右边界。if True，expiry指代expiry当天15：00; if False, expiry指代expiry前一天15：00

    Returns
    -------

    """
    if isinstance(current, (datetime, np.datetime64)) or isinstance(expiry, (
            datetime, np.datetime64)): raise NotImplementedError()

    if trading_dates is None:
        # 历史和未来交易日信息
        data_root = '/Users/hongyifan/Desktop/work/internship/citic_futures/20231226bid-ask-spread/data/'
        start = '2000-01-01'
        end = '2024-12-31'
        market = 'cn'
        trading_dates = np.array(
            pd.read_csv(data_root + f'trading_dates_{market}_{start}_{end}.csv')['trading_dates'].tolist(), dtype=str)
        logging.warning(f"正在读取{data_root}下的交易日信息，数据截至{end}，请及时更新")
    if (not current in trading_dates) or (not expiry in trading_dates): raise ValueError('输入日期为非交易日')

    idx_curr = np.argwhere(trading_dates == current)[0][0]
    idx_exp = np.argwhere(trading_dates == expiry)[0][0]
    res = idx_exp - idx_curr
    if not including_right:
        res -= 1
    return res


def calc_time2maturity(current: Union[pd.DatetimeIndex, pd.Series], expiry: Union[pd.DatetimeIndex], days2maturity=None,
                       annualized=244):
    """

    Parameters
    ----------
    current :
    expiry :
    days2maturity :
    annualized : Optional, int,
        一年总共的交易日，用于将t2m年化

    Returns
    -------

    """
    if not isinstance(current, (list)):
        # 现阶段仅支持current和expiry都在同一天的计算
        assert days2maturity is not None
        # 计算time to maturity
        if days2maturity is not None:
            if len(set(expiry.date).difference(set(current.date))) >= 1:
                raise ValueError(
                    f'input has different dates {set(expiry.date).difference(set(current.date))}, conflict with param `days2maturity`')
        elif days2maturity is None:
            days2maturity = expiry.date - current.date  # 计算有多少天?

        # 计算time to maturity
        seconds2maturity = (expiry - current).total_seconds() + (
                expiry - current).microseconds / 1000000  # 计算距离maturity还有多少秒
        seconds2maturity = np.array(seconds2maturity)
        is_morning = seconds2maturity > 2 * 60 * 60
        seconds2maturity[is_morning] = seconds2maturity[is_morning] - 1.5 * 60 * 60

        time2maturity = seconds2maturity / (4 * 60 * 60) + days2maturity
        if annualized is not None:
            time2maturity /= annualized
        time2maturity = np.where(time2maturity < 1e-3, 1e-3,
                                 time2maturity)  # For small TTEs, use a small value (1e-3).即最后一小时的TTM都视为1h
        time2maturity = pd.Series(time2maturity, index=current, name='time2maturity')
        return time2maturity
    else:
        raise NotImplementedError()


def calc_effective_price(fe, df, method='wap', level=1) -> Union[pd.DataFrame, pd.Series]:
    """计算期权有效价格

    Parameters
    ----------
    fe : LobFeatureEngineering
    df :
    method :

    Returns
    -------

    """
    res = pd.DataFrame()
    if method == 'wap':
        for i in range(1, level + 1):
            res = pd.concat([res, fe.calc_wap(df, level=i, cross=True)], axis=1)
    elif method == 'cum_vol_wap':
        res = fe.calc_cum_vol_wap(df, cum_level=level, cross=True)
    elif method == 'cum_wap':
        res = fe.calc_cum_wap(df, level=level, cross=True)
    res.index = pd.to_datetime(res.index)

    return res


from py_vollib_vectorized.implied_volatility import vectorized_implied_volatility
from py_vollib_vectorized.greeks import delta, gamma, vega, theta, rho


def calc_iv_greeks(price, S, K, t, r, flag, q, calc_greeks=True, index=None):
    iv = vectorized_implied_volatility(price=price, S=S, K=K, t=t, r=r, flag=flag, q=q, on_error='ignore').replace(0,
                                                                                                                   np.nan)
    res = iv
    if calc_greeks:
        option_delta = delta(flag=flag, S=S, K=K, t=t, r=r, sigma=iv, q=q)
        option_gamma = gamma(flag=flag, S=S, K=K, t=t, r=r, sigma=iv, q=q)
        option_vega = vega(flag=flag, S=S, K=K, t=t, r=r, sigma=iv, q=q)
        option_theta = theta(flag=flag, S=S, K=K, t=t, r=r, sigma=iv, q=q)
        option_rho = rho(flag=flag, S=S, K=K, t=t, r=r, sigma=iv, q=q)
        res = pd.concat([iv, option_delta, option_gamma, option_vega, option_theta, option_rho], axis=1)
    if index is not None:
        res.index = index
    res.index = pd.to_datetime(res.index)
    return res


def get_class_name(obj):
    """获取某个对象的名称，常用于命名文件

    Parameters
    ----------
    obj :

    Returns
    -------

    """
    name = str(obj).split('.')[-1].split(' ')[0]
    return name


def get_expire_map(date_, expire_date_list):
    """获取某个日期的四个到期日，即当月、下月、当季、下季，前两者为近月，后两者为远月

    Parameters
    ----------
    date_ :
    expire_date_list : list,
        到期日的list，需要有序

    Returns
    -------

    """

    def get_month(date_str: str):
        return date_str.split('-')[1]

    quater_months = ['03', '06', '09', '12']
    expire_date_list = deepcopy(np.array(expire_date_list))
    after_today = expire_date_list[date_ <= expire_date_list]
    res = []
    counter = 1
    for i, expire_date in enumerate(after_today):
        if counter <= 2:
            res.append(expire_date)
            counter += 1
        elif counter >= 3 and counter <= 4:
            if get_month(expire_date) in quater_months:
                res.append(expire_date)
                counter += 1
        else:
            break
    expire_date_map = dict(zip(res, [1, 2, 3, 6]))
    assert len(expire_date_map) == 4
    return expire_date_map
