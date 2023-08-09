# -*- coding=utf-8 -*-
# @File     : config.py
# @Time     : 2023/8/2 18:57
# @Author   : EvanHong
# @Email    : 939778128@qq.com
# @Project  : 2023.06.08超高频上证50指数计算
# @Description:
from datetime import timedelta

import pandas as pd
import os

from sklearn.model_selection import ParameterGrid

code_dict = {
    # '贵州茅台': '600519.XSHG',
    # '宁德时代': '300750.XSHE',
    # '中国平安': '601318.XSHG',
    # '招商银行': '600036.XSHG',
    # '五粮液': '000858.XSHE',
    # '美的集团': '000333.XSHE',
    # '比亚迪': '002594.XSHE',
    # '兴业银行': '601166.XSHG',
    # '恒瑞医药': '600276.XSHG',
    # '紫金矿业': '601899.XSHG',
    # '浦发银行': '600000.XSHG',
    # '海通证券': '600837.XSHG',
    # '沪深300ETF': '510300.XSHG',
    # '涨停股测试': '600219.XSHG',
    '贵州茅台': '600519.XSHG',
    '宁德时代': '300750.XSHE',
    '招商银行': '600036.XSHG',
    '中国平安': '601318.XSHG',
    '隆基绿能': '601012.XSHG',
    '五粮液': '000858.XSHE',
    '比亚迪': '002594.XSHE',
    '美的集团': '000333.XSHE',
    '兴业银行': '601166.XSHG',
    '东方财富': '300059.XSHE',
}


root = 'D:/Work/INTERNSHIP/海通场内/2023.06.08超高频上证50指数计算/'
data_root = root + 'data/'
detail_data_root=data_root+'个股交易细节/'
res_root = root + 'res/'
model_root = root + 'models/'
scaler_root = root + 'scalers/'
y='2022'
m='06'
d='29'
date = f'{y}{m}{d}'
date1 = f'{y}-{m}-{d}'
start = pd.to_datetime(f'{date1} 09:30:00')
end = pd.to_datetime(f'{date1} 15:00:00.001')

FILE_FMT_order_book_history="{}_{}_order_book_history.csv" # 默认买卖各10档
FILE_FMT_price_history="{}_{}_price_history.csv"
FILE_FMT_clean_obh="{}_{}_clean_obh.csv" # 默认买卖各5档
FILE_FMT_my_trade_details="{}_{}_my_trade_details.csv"
FILE_FMT_vol_tov="{}_{}_vol_tov.csv"
FILE_FMT_model="{}_period{}_automl.pkl"
FILE_FMT_scaler= "{}_scaler_{}_{}.pkl"




important_times = {
    'open_call_auction_start': pd.to_datetime(f'{date1} 09:15:00.000000'),
    'open_call_auction_end': pd.to_datetime(f'{date1} 09:25:00.000000'),
    'continues_auction_am_start': pd.to_datetime(f'{date1} 09:30:00.000000'),
    'continues_auction_am_end': pd.to_datetime(f'{date1} 11:30:00.000000'),
    'continues_auction_pm_start': pd.to_datetime(f'{date1} 13:00:00.000000'),
    'continues_auction_pm_end': pd.to_datetime(f'{date1} 14:57:00.000000'),
    'close_call_auction_start': pd.to_datetime(f'{date1} 14:57:00.000000'),
    'close_call_auction_end': pd.to_datetime(f'{date1} 15:00:00.000000'), }

ranges = [(pd.to_datetime(f'{date1} 09:30:00.000'),
           pd.to_datetime(f'{date1} 10:30:00.000') - timedelta(milliseconds=10)),
          (pd.to_datetime(f'{date1} 10:30:00.000'),
           pd.to_datetime(f'{date1} 11:30:00.000') - timedelta(milliseconds=10)),
          (pd.to_datetime(f'{date1} 13:00:00.000'),
           pd.to_datetime(f'{date1} 14:00:00.000') - timedelta(milliseconds=10)),
          (pd.to_datetime(f'{date1} 14:00:00.000'),
           pd.to_datetime(f'{date1} 14:57:00.000') - timedelta(milliseconds=10))]


# prediction

freq='200ms'
pred_n_steps = 200 # 预测40s，即200个steps
use_n_steps = 50 # 利用use_n_steps个steps的数据去预测pred_n_steps之后的涨跌幅
drop_current = False # 是否将当前股价作为因子输入给模型
# num = param['num']
# target = param['target']  # ret,current
# pred_n_steps = param['pred_n_steps']
# use_n_steps = param['use_n_steps']
# drop_current = param['drop_current']

# print('params are',param)