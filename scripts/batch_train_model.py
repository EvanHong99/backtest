# -*- coding=utf-8 -*-
# @File     : batch_train_model.py
# @Time     : 2023/8/11 16:33
# @Author   : EvanHong
# @Email    : 939778128@qq.com
# @Project  : 2023.06.08超高频上证50指数计算
# @Description: 训练模型，并保存相关的scaler和model

from backtest.backtester import LobBackTester
from broker import Broker
from config import *
import config
from datafeed import LobDataFeed
from observer import LobObserver
from statistics import LobStatistics
from strategies import LobStrategy
from preprocess import AggDataPreprocessor
from support import Target, update_date





if __name__ == '__main__':
    # todo 每个小时内是否需要打乱
    # train scaler for each stk
    # train on the first two days
    train_ratio = 2 / 3
    train_len = int(len(self.alldatas) * train_ratio)
    train_date = list(self.alldatas.keys())[:train_len]

    f_dict= {}
    for r,d,f in os.walk(detail_data_root):
        print(f)
        for filename in f:
            if filename=='placeholder':continue
            parts=filename.split('_')
            yyyy=parts[0][:4]
            mm = parts[0][4:6]
            dd = parts[0][-2:]
            stk_name=parts[1]
            f_dict[stk_name]=(yyyy, mm, dd)
    f_dict=sorted(list(set(f_dict)))

    for yyyy,mm,dd,stk_name in f_dict:
        update_date(yyyy,mm, dd)
        datafeed = LobDataFeed()
        for num in range(4):
            feature=datafeed.load_feature(detail_data_root,config.date,stk_name,num)




