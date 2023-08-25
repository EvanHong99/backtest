# -*- coding=utf-8 -*-
# @File     : ret.py
# @Time     : 2023/8/2 12:22
# @Author   : EvanHong
# @Email    : 939778128@qq.com
# @Project  : 2023.06.08超高频上证50指数计算
# @Description:
import logging
from copy import deepcopy
import flaml
import flaml.automl.state

from config import *
import config
from support import *
from preprocessors.preprocess import LobTimePreprocessor
import pickle
import h5py
import pandas as pd

class BaseDataFeed(object):
    def __init__(self, *args, **kwargs):
        self.args = args
        for key, value in kwargs.items():
            self.__setattr__(key, value)


class LobModelFeed(BaseDataFeed):
    def __init__(self, model_root, stk_name,model_class, *args, **kwargs):
        # load models
        super().__init__(*args, **kwargs)
        self.models = []
        for num in range(4):
            pickle.load(open(model_root + FILE_FMT_model.format(stk_name, num, model_class), 'rb'))
            try:
                model = pickle.load(open(model_root + FILE_FMT_model.format(stk_name, num,model_class), 'rb'))
                print(f'best model for period {num}', model.model.estimator)
                self.models.append(deepcopy(model))
            except Exception as e:
                logging.error(model_root + FILE_FMT_model.format(stk_name, num,model_class)+" not exists")
                print(str(e.__context__))

    @staticmethod
    def load_model(model_root, stk_name,num,model_class,verbose=False):
        try:
            model = pickle.load(open(model_root + FILE_FMT_model.format(stk_name, num,model_class), 'rb'))
            if verbose:print(f'best model for period {num}', model.model.estimator)
        except:
            logging.error(model_root + FILE_FMT_model.format(stk_name, num,model_class) + " not exists")

class LobDataFeed(BaseDataFeed):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.file_root = None
        self.date = None
        self.stk_name = None

        self.trade_details = None
        self.order_details = None
        self.vol_tov = None
        self.current = None
        self.order_book_history = None
        self.clean_obh = None

    def get_order_details(self, data_root, date, symbol):
        """
        通过h5文件提取出对应symbol的数据
        :param order_f:
        :param symbol:
        :return:
        """
        if symbol.startswith('6'):
            order_f = h5py.File(data_root + f'{date}.orders.XSHG.h5', 'r')
        else:
            order_f = h5py.File(data_root + f'{date}.orders.XSHE.h5', 'r')
        order_dset = order_f[symbol]
        order_details = pd.DataFrame.from_records(order_dset[:])
        order_details = order_details.set_index('seq', drop=False)
        order_details['timestamp'] = pd.to_datetime(order_details['timestamp'], format='%Y%m%d%H%M%S%f')
        order_details['last_traded_timestamp'] = pd.to_datetime(
            order_details['last_traded_timestamp'].replace(0, np.nan),
            format='%Y%m%d%H%M%S%f')
        order_details['canceled_timestamp'] = pd.to_datetime(order_details['canceled_timestamp'].replace(0, np.nan),
                                                             format='%Y%m%d%H%M%S%f')
        order_details['price'] = order_details['price'] / 10000
        order_details['filled_amount'] = order_details['filled_amount'] / 10000

        temp = order_details.loc[order_details['type'] == OrderTypeInt.bop.value]
        if len(temp) > 0:
            # raise Exception('contents best of party (bop) price order')
            print('contents best of party (bop) price order')
        return order_details

    def get_trade_details(self, data_root, date, symbol):
        """
        通过h5文件提取出对应symbol的数据
        :param trade_f:
        :param symbol:
        :return:
        """
        try:
            if symbol.startswith('6'):
                trade_f = h5py.File(data_root + f'{date}.trades.XSHG.h5', 'r')
            else:
                trade_f = h5py.File(data_root + f'{date}.trades.XSHE.h5', 'r')
            trade_dset = trade_f[symbol]
            trade_details = pd.DataFrame.from_records(trade_dset[:])
            trade_details = trade_details.set_index('seq', drop=False)
            trade_details['timestamp'] = pd.to_datetime(trade_details['timestamp'], format='%Y%m%d%H%M%S%f')
            trade_details['price'] = trade_details['price'] / 10000
            return trade_details
        except:
            print(f"KeyError: Unable to open object (object '{symbol}' doesn't exist)")
            return None

    def load_details(self, data_root, date, symbol):
        self.trade_details = self.get_trade_details(data_root=data_root, date=date, symbol=symbol)
        self.order_details = self.get_order_details(data_root=data_root, date=date, symbol=symbol)
        self.trade_details = LobTimePreprocessor.del_untrade_time(self.trade_details, cut_tail=False)
        self.trade_details = LobTimePreprocessor.add_head_tail(self.trade_details,
                                                               head_timestamp=config.important_times[
                                                                   'continues_auction_am_start'],
                                                               tail_timestamp=config.important_times[
                                                                   'close_call_auction_end'])
        self.order_details = LobTimePreprocessor.del_untrade_time(self.order_details, cut_tail=False)
        self.order_details = LobTimePreprocessor.add_head_tail(self.order_details,
                                                               head_timestamp=config.important_times[
                                                                   'continues_auction_am_start'],
                                                               tail_timestamp=config.important_times[
                                                                   'close_call_auction_end'])

        return self.trade_details, self.order_details

    def load_events(self, file_root, date, stk_name):
        self.events = pd.read_csv(file_root + FILE_FMT_events.format(date, stk_name), index_col=0)
        self.events.index = pd.to_datetime(self.events.index)
        return self.events

    def load_basic(self, file_root, date, stk_name):

        self.order_book_history = pd.read_csv(file_root + FILE_FMT_order_book_history.format(date, stk_name),
                                              index_col=0,
                                              header=0)
        self.order_book_history.columns = self.order_book_history.columns.astype(float)
        self.order_book_history.index = pd.to_datetime(self.order_book_history.index)
        # print(self.order_book_history)
        self.order_book_history = LobTimePreprocessor.del_untrade_time(self.order_book_history, cut_tail=False)
        self.order_book_history = LobTimePreprocessor.add_head_tail(self.order_book_history,
                                                                    head_timestamp=config.important_times[
                                                                        'continues_auction_am_start'],
                                                                    tail_timestamp=config.important_times[
                                                                        'close_call_auction_end'])

        self.current = \
        pd.read_csv(file_root + FILE_FMT_price_history.format(date, stk_name)).set_index('timestamp')[
            'current'].rename('current').groupby(level=0).last()
        self.current.index = pd.to_datetime(self.current.index)
        return self.order_book_history, self.current

    def load_clean_obh(self, file_root, date, stk_name, snapshot_window=5,use_cols:list=None):

        if use_cols is not None:
            cols=use_cols
        else:
            # 删除部分high level列
            cols = [str(LobColTemplate('a', i, 'p')) for i in range(snapshot_window, 0, -1)]
            cols += [str(LobColTemplate('b', i, 'p')) for i in range(1, snapshot_window + 1)]
            cols += [str(LobColTemplate('a', i, 'v')) for i in range(snapshot_window, 0, -1)]
            cols += [str(LobColTemplate('b', i, 'v')) for i in range(1, snapshot_window + 1)]
            cols.append('current')

        self.clean_obh = pd.read_csv(file_root + FILE_FMT_clean_obh.format(date, stk_name), index_col=0)
        self.clean_obh.index = pd.to_datetime(self.clean_obh.index)
        self.clean_obh = self.clean_obh.loc[:, cols]
        return self.clean_obh

    def load_vol_tov(self, file_root, date, stk_name):
        self.vol_tov = pd.read_csv(file_root + FILE_FMT_vol_tov.format(date, stk_name), index_col='timestamp')
        self.vol_tov.index = pd.to_datetime(self.vol_tov.index)

        return self.vol_tov

    def load_feature(self, file_root, date, stk_name,num):
        self.feature = pd.read_csv(file_root + FILE_FMT_feature.format(date, stk_name,num), index_col=0,header=0)
        self.feature.index = pd.to_datetime(self.feature.index)

        return self.feature

if __name__ == '__main__':
    pass
