# -*- coding=utf-8 -*-
# @Project  : caitong_securities
# @FilePath : 2023.10.23股指日内短期预测/backtest
# @File     : basics.py
# @Time     : 2023/11/13 14:07
# @Author   : EvanHong
# @Email    : 939778128@qq.com
# @Description:
from datetime import datetime, time

def time_plus_timedelta(time, timedelta):
    start = datetime(
        2000, 1, 1,
        hour=time.hour, minute=time.minute, second=time.second)
    end = start + timedelta
    return end.time()

def check_middle_hours(df,start:str='11:29:50'):
    date=df.index.date[0]
    print(df.loc[f"{date} {start}":])
