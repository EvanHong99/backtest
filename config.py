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
    # 沪深300
    # '贵州茅台': '600519.XSHG',
    # '宁德时代': '300750.XSHE',
    # '招商银行': '600036.XSHG',
    # '中国平安': '601318.XSHG',
    # '隆基绿能': '601012.XSHG',
    # '五粮液': '000858.XSHE',
    # '比亚迪': '002594.XSHE',
    # '美的集团': '000333.XSHE',
    # '兴业银行': '601166.XSHG',
    # '东方财富': '300059.XSHE',

    # 上证50
    # '贵州茅台':'600519.XSHG',
    # '中国平安': '601318.XSHG',
    '招商银行': '600036.XSHG',
    '兴业银行': '601166.XSHG',
    '中信证券': '600030.XSHG',
    '紫金矿业': '601899.XSHG',
    '长江电力': '600900.XSHG',
    '恒瑞医药': '600276.XSHG',
    '万华化学': '600309.XSHG',
    '伊利股份': '600887.XSHG',
    '隆基绿能': '601012.XSHG',
    '工商银行': '601398.XSHG',
    '药明康德': '603259.XSHG',
    '中国建筑': '601668.XSHG',
    '中国中免': '601888.XSHG',
    '中国石化': '600028.XSHG',
    '山西汾酒': '600809.XSHG',
    '农业银行': '601288.XSHG',
    '三一重工': '600031.XSHG',
    '保利发展': '600048.XSHG',
    '中国联通': '600050.XSHG',
    '国电南瑞': '600406.XSHG',
    '通威股份': '600438.XSHG',
    '中国神华': '601088.XSHG',
    '海尔智家': '600690.XSHG',
    '中国石油': '601857.XSHG',
    '中国电信': '601728.XSHG',
    '片仔癀': '600436.XSHG',
    '韦尔股份': '603501.XSHG',
    '特变电工': '600089.XSHG',
    '中国中铁': '601390.XSHG',
    '海天味业': '603288.XSHG',
    '三峡能源': '600905.XSHG',
    '兆易创新': '603986.XSHG',
    '金山办公': '688111.XSHG',
    '上汽集团': '600104.XSHG',
    '华友钴业': '603799.XSHG',
    '中远海控': '601919.XSHG',
    '陕西煤业': '601225.XSHG',
    '北方稀土': '600111.XSHG',
    '中国人寿': '601628.XSHG',
    '航发动力': '600893.XSHG',
    '包钢股份': '600010.XSHG',
    '中国电建': '601669.XSHG',
    '天合光能': '688599.XSHG',
    '闻泰科技': '600745.XSHG',
    '复星医药': '600196.XSHG',
    '长城汽车': '601633.XSHG',
    '中信建投': '601066.XSHG',
    '合盛硅业': '603260.XSHG',

    # 测试
    # '恒瑞医药': '600276.XSHG',
    # '紫金矿业': '601899.XSHG',
    # '浦发银行': '600000.XSHG',
    # '海通证券': '600837.XSHG',
    # '沪深300ETF': '510300.XSHG',
    # '涨停股测试': '600219.XSHG',
}
stk_name_dict = {v: k for k, v in code_dict.items()}

root = 'D:/Work/INTERNSHIP/海通场内/2023.06.08超高频上证50指数计算/'
data_root = root + 'data/'
detail_data_root = data_root + '个股交易细节/'
res_root = root + 'res/'
model_root = root + 'models/'
scaler_root = root + 'scalers/'

FILE_FMT_order_book_history = "{}_{}_order_book_history.csv"  # 默认买卖各10档
FILE_FMT_price_history = "{}_{}_price_history.csv"
FILE_FMT_clean_obh = "{}_{}_clean_obh.csv"  # 默认买卖各5档
FILE_FMT_my_trade_details = "{}_{}_my_trade_details.csv"
FILE_FMT_vol_tov = "{}_{}_vol_tov.csv"
FILE_FMT_model = "{}_period{}_automl.pkl"
FILE_FMT_scaler = "{}_scaler_{}_{}.pkl"
FILE_FMT_events = "{}_{}_events.csv"

y = None
m = None
d = None
date = None
date1 = None
start = None
end = None
important_times = None
ranges = None


# prediction settings
freq = '200ms'
pred_n_steps = 200  # 预测40s，即200个steps
use_n_steps = 50  # 利用use_n_steps个steps的数据去预测pred_n_steps之后的涨跌幅
drop_current = False  # 是否将当前股价作为因子输入给模型
use_level = 5
# target = param['target']  # ret,current

# init
# update_date(yy='2022', mm='06', dd='29')
