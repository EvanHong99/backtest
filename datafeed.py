# -*- coding=utf-8 -*-
# @File     : ret.py
# @Time     : 2023/8/2 12:22
# @Author   : EvanHong
# @Email    : 939778128@qq.com
# @Project  : 2023.06.08超高频上证50指数计算
# @Description:
from copy import deepcopy

import numpy as np
import pandas as pd

from config import *
from support import LobColTemplate
from preprocess import LobTimePreprocessor,LobCleanObhPreprocessor
import pickle


class BaseDataFeed(object):
    def __init__(self, *args, **kwargs):
        self.args = args
        for key, value in kwargs.items():
            self.__setattr__(key, value)

class LobModelFeed(BaseDataFeed):
    def __init__(self,model_root,stk_name, *args, **kwargs):
        # load models
        super().__init__(*args, **kwargs)
        self.models = []
        for num in range(4):
            model = pickle.load(open(model_root + FILE_FMT_model.format(stk_name,num), 'rb'))
            print(f'best model for period {num}', model.model.estimator)
            self.models.append(deepcopy(model))

class LobDataFeed(BaseDataFeed):
    def __init__(self, file_root, date: str, stk_name: str,*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.file_root=file_root
        self.date= date.replace('-', '')
        self.stk_name=stk_name

        self.vol_tov = None
        self.current = None
        self.order_book_history = None
        self.clean_obh = None

    def load_basic(self):

        self.order_book_history = pd.read_csv(self.file_root + FILE_FMT_order_book_history.format(self.date,self.stk_name), index_col=0,
                                              header=0)
        self.order_book_history.columns = self.order_book_history.columns.astype(float)
        self.order_book_history.index = pd.to_datetime(self.order_book_history.index)
        # print(self.order_book_history)
        self.order_book_history = LobTimePreprocessor.del_untrade_time(self.order_book_history, cut_tail=False)
        self.order_book_history = LobTimePreprocessor.add_head_tail(self.order_book_history,
                                                                    head_timestamp=important_times[
                                                                        'continues_auction_am_start'],
                                                                    tail_timestamp=important_times[
                                                                        'close_call_auction_end'])

        self.current = pd.read_csv(self.file_root + FILE_FMT_price_history.format(self.date,self.stk_name)).set_index('timestamp')[
            'current'].rename('current').groupby(level=0).last()
        self.current.index = pd.to_datetime(self.current.index)
        return self.order_book_history,self.current

    def load_clean_obh(self, snapshot_window=5):
        self.clean_obh = pd.read_csv(self.file_root + FILE_FMT_clean_obh.format(self.date,self.stk_name), index_col=0)
        self.clean_obh.index = pd.to_datetime(self.clean_obh.index)

        # 可能需要删除部分列
        cols=[str(LobColTemplate('a',i,'p')) for i in range(snapshot_window,0,-1)]
        cols+=[str(LobColTemplate('b',i,'p')) for i in range(1,snapshot_window+1)]
        cols+=[str(LobColTemplate('a',i,'v')) for i in range(snapshot_window,0,-1)]
        cols+=[str(LobColTemplate('b',i,'v')) for i in range(1,snapshot_window+1)]
        cols.append('current')

        self.clean_obh=self.clean_obh.loc[:,cols]
        return self.clean_obh

    def load_vol_tov(self):
        self.vol_tov=pd.read_csv(self.file_root+FILE_FMT_vol_tov.format(self.date,self.stk_name),index_col='timestamp')
        self.vol_tov.index = pd.to_datetime(self.vol_tov.index)

        return self.vol_tov

def test_LobCleanObhPreprocessor():
    stk_name="贵州茅台"
    datafeed=LobDataFeed(detail_data_root,date,stk_name=stk_name)
    datafeed.load_basic()
    cobh_pp=LobCleanObhPreprocessor(datafeed,snapshot_window=5)
    cobh_pp._gen_clean_obh(datafeed)
    cobh_pp.gen_and_save(detail_data_root,date,stk_name=stk_name)

if __name__ == '__main__':
    pass

