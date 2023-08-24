# -*- coding=utf-8 -*-
# @File     : order_book_reconstruction.py
# @Time     : 2023/8/11 9:02
# @Author   : EvanHong
# @Email    : 939778128@qq.com
# @Project  : 2023.06.08超高频上证50指数计算
# @Description:

import os
import sys
path = os.path.join(os.path.dirname(__file__), os.pardir)
sys.path.append(path)

import logging
from copy import deepcopy
from typing import Union

import numpy as np
import pandas as pd
from sortedcontainers import SortedDict
from tqdm import tqdm

from config import *
import config
from preprocessors.preprocess import LobCleanObhPreprocessor
from support import OrderTypeInt, OrderSideInt


class OrderTypeIntError(TypeError):
    def __init__(self, *arg):
        self.args = arg


class Trade(object):

    def __init__(self, seq, bid_seq, ask_seq, quantity, price, timestamp):
        self.seq = seq
        self.bid_seq = bid_seq
        self.ask_seq = ask_seq
        self.quantity = quantity
        self.price = price
        self.timestamp = timestamp

    def __str__(self):
        return str(self.__dict__)


class Order(object):
    def __init__(self, seq, timestamp, side, type_, quantity, price, filled_quantity=0, filled_amount=0,
                 last_traded_timestamp=pd.NaT, canceled_timestamp=pd.NaT, canceled_seq=0):
        self.seq = seq
        self.timestamp = timestamp
        self.side = side
        self.type = type_
        self.quantity = quantity
        self.price = price
        self.filled_quantity = filled_quantity
        self.can_filled = self.quantity - self.filled_quantity
        self.filled_amount = filled_amount
        self.last_traded_timestamp = last_traded_timestamp
        self.canceled_timestamp = canceled_timestamp
        self.canceled_seq = canceled_seq
        self.filled = False
        self.canceled = False

    def __str__(self):
        return "seq-{}, {}: {} @ {}. side={},type={},filled_quantity={},filled_amount={},\n last_traded_timestamp={},canceled_timestamp={},canceled_seq={}\n".format(
            self.seq, self.timestamp, self.quantity, self.price, OrderSideInt(self.side).name,
            OrderTypeInt(self.type).name, self.filled_quantity, self.filled_amount, self.last_traded_timestamp,
            self.canceled_timestamp, self.canceled_seq)

    def fill_quantity(self, timestamp, quantity, price):
        """全部/部分fill放在orderbook对象中处理，进而保证函数功能的原子性"""
        if self.filled:
            raise Exception('already filled ' + self.__str__())
        if self.canceled:
            raise Exception('already canceled ' + self.__str__())  # 限价单则限制价格
        #         if self.type==OrderTypeInt['limit'].value:
        #             assert self.price==price

        assert quantity <= self.can_filled
        self.last_traded_timestamp = timestamp
        self.filled_quantity += quantity
        self.filled_amount += price * quantity
        self.can_filled -= quantity
        # 市价单平均价格会变化
        if self.type == OrderTypeInt['market'].value:
            self.price = self.filled_amount / self.filled_quantity  # 平均价格
        self._set_filled()
        return quantity

    def _set_filled(self):
        if self.quantity == self.filled_quantity:
            self.filled = True

    def is_filled(self):
        if self.quantity == self.filled_quantity:
            self.filled = True
            return True
        else:
            return False

    # def can_filled(self):
    #     """quantity to fill"""
    #     return self.quantity - self.filled_quantity

    def cancel(self, canceled_seq, canceled_timestamp):
        self.canceled_seq = canceled_seq
        self.canceled_timestamp = canceled_timestamp
        self.canceled = True

    def to_dict(self):
        return self.__dict__


class OrderBook(object):
    """
    Attributes
    ----------
    book_bid: SortedDict, {price : OrderedDict of Orders}


    Note
    ----
    有以下特点：

    1. 如果一开始没有full fill，那么之后fill会使得seq自增，但是并不改变原始order的seq，只是在它的基础上更改了quantity和amount、price，并使得last_traded_timestamp非nan。
        `order_details.loc[np.logical_and(order_details['price']*order_details['filled_quantity']!=order_details['filled_amount'],order_details['price']!=0)]`，即没有fill的市价单，并没有太多样本，仅2000-，所有样本的last traded time都是not nan，可能是当时没有完全fill的，所以导致价格\*数量不平；
        `order_details.loc[~order_details['last_traded_timestamp'].isna()]`有6w+，包含限价、市价单，但限价单价格保持限定的价格。

    2. 市价单<==>order价格为0

    3. `['09:25:00.000000','09:30:00.000000'),['11:30:00.000000','13:00:00.000000'),['15:00:00.000000':)`没有任何委托

    4. 吃完了五档价格，继续吃对手，直到全部成交
        ```python
        order_details.loc[14943]
        trade_details.loc[trade_details['ask_seq']==14943,'quantity'].sum()
        ```

    """

    def __init__(self, stk_name, symbol, snapshot_window=10,snapshot_ABtest=False):
        self.stk_name = stk_name
        self.symbol = symbol
        self.exchange = 'XSHG' if self.symbol.endswith('XSHG') else 'XSHE'
        self.snapshot_window = snapshot_window
        self.snapshot_ABtest=snapshot_ABtest
        # 做的简单的话只需要存交易的股数
        self.current = -1
        self.close_p = -1
        self.open_p = -1
        self.book_bid = SortedDict()  # 买，需要用 .peekitem(-1) 去索引
        self.book_ask = SortedDict()  # 卖
        self.pv_dict_bid = SortedDict()  # price-volume
        self.pv_dict_ask = SortedDict()  # price-volume
        self.order_queue_dict = {}  # 有序dict，根据插入顺序排序
        self.trade_counter = 0
        self.trade_queue = []  # 有序list，根据插入顺序排序

        self.cum_volume = 0  # 累积交易量
        self.cum_money = 0  # 累积交易额
        self.history = pd.DataFrame(
            columns=['timestamp', 'current', 'volume', 'money'])  # (上一个timestamp,timestamp]期间发生的交易情况
        # self.price_history=pd.DataFrame(columns=['seq','timestamp','current'])
        self.price_history_dict = {}
        self.price_history = None
        self.price_history_idx = 0
        self.order_book_history_dict = {}  # timestamp发生交易后盘口数据
        self.order_book_history_dict1 = {}  # timestamp发生交易后盘口数据
        self.order_book_history = None

        self.open_call_auction_matched = False
        self.close_call_auction_matched = False

        # 记录最优价格
        self.last_best_bid = -1
        self.last_worst_bid = -1
        self.last_best_ask = -1
        self.last_worst_ask = -1

    def get_best_bid(self):
        if len(self.book_bid) > 0:
            self.last_best_bid = self.book_bid.peekitem(-1)[0]
            return self.last_best_bid
        else:
            return -1

    def get_worst_bid(self):
        if len(self.book_bid) > 0:
            self.last_worst_bid = self.book_bid.peekitem(0)[0]
            return self.last_worst_bid
        else:
            return -1

    def get_best_ask(self):
        if len(self.book_ask) > 0:
            self.last_best_ask = self.book_ask.peekitem(0)[0]
            return self.last_best_ask
        else:
            return -1

    def get_worst_ask(self):
        if len(self.book_ask) > 0:
            self.last_worst_ask = self.book_ask.peekitem(-1)[0]
            return self.last_worst_ask
        else:
            return -1

    # testit
    def create_order(self):
        """下单"""
        pass

    def append_to_book(self, order: Order):
        """
        在book_bid、book_ask加入订单，仅在集合竞价阶段需要用到，否则直接操作两个books即可

        .. note: 必然不为市价单，因为市价单仅限于连续竞价阶段，如果对手方订单簿没有订单，则市价单会撤单，而非进入订单簿
        """
        try:
            assert order.type != OrderTypeInt.market.value  # 必然不为市价单，只能是限价单或己方最优
        except:
            logging.warning(f'order type is not limit but still append it into order book {str(order)}')
            # order.type = OrderTypeInt.limit.value
            # order.price = self.get_best_ask() if order.side == OrderSideInt.ask.value else self.get_best_bid()
            # logging.warning(f'transform it to limit order {str(order)}')

        # order book
        # 浅拷贝可以实现对dict的更改
        if order.side == OrderSideInt.bid.value:
            book = self.book_bid
            pv_dict = self.pv_dict_bid
        else:
            book = self.book_ask
            pv_dict = self.pv_dict_ask

        temp = book.get(order.price)
        if temp is None:
            book[order.price] = {order.seq: order}
            pv_dict[order.price] = order.can_filled
        else:
            book[order.price][order.seq] = order
            pv_dict[order.price] += order.can_filled

        # order queue
        self.order_queue_dict[order.seq] = order

    def append_to_trade(self, order, counter_order, filled, price, timestamp=None):
        if timestamp is None: timestamp = order.timestamp
        if order.side == OrderSideInt.bid.value:
            self.trade_queue.append(
                Trade(self.trade_counter, order.seq, counter_order.seq, filled, price, timestamp=timestamp))
            self.trade_counter += 1
        else:
            self.trade_queue.append(
                Trade(self.trade_counter, counter_order.seq, order.seq, filled, price, timestamp=timestamp))
            self.trade_counter += 1

    def sum_volume(self, order_dict):
        s = 0
        for seq, order in order_dict.items():
            s += order.can_filled
        return s

    def calc_p_call_auction(self, is_open):
        """
        计算集合竞价申报撮合。价格优先，时间优先

        3.5.2 集合竞价时，成交价格的确定原则为： （一）可实现最大成交量的价格； （二）高于该价格的买入申报与低于该价格的卖出申报全部成交的价格； （三）与该价格相同的买方或卖方至少有一方全部成交的价格。 两个以上申报价格符合上述条件的，使未成交量最小的申报价格为成交价格；仍有两个以上使未成交量最小的申报价格符合上述条件的，其中间价为成交价格。 集合竞价的所有交易以同一价格成交。

        refs: [1](https://caifuhao.eastmoney.com/news/20211019210936002828900)
        """

        # todo如果没有任何match，以前一天收盘作为开盘价
        ...

        # ask_v=pd.DataFrame(ret={k:self.sum_volume(v) for k,v in self.book_ask.items()},index=['ask_v']).T.sort_index(ascending=False)
        # bid_v=pd.DataFrame(ret={k:self.sum_volume(v) for k,v in self.book_bid.items()},index=['bid_v']).T.sort_index(ascending=False)
        # vol_table=pd.concat([ask_v,bid_v],axis=1).fillna(0).sort_index(ascending=False)
        # ask_cum_v=vol_table['ask_v'].sort_index(ascending=True).cumsum().sort_index(ascending=False).rename('ask_cum_v')
        # bid_cum_v=vol_table['bid_v'].cumsum().rename('bid_cum_v')
        # vol_table=pd.concat([vol_table,ask_cum_v,bid_cum_v],axis=1)
        # # 既有买又有卖的价格是可行的开盘价
        # avaliable_prices=np.logical_and(vol_table['ask_v']!=0,vol_table['bid_v']!=0)
        # avaliable_prices=avaliable_prices.loc[avaliable_prices]
        # temp=vol_table[['ask_cum_v','bid_cum_v']].loc[avaliable_prices.index].min(axis=1)
        # open_p=temp.loc[temp==temp.max()].index.tolist()
        # max_tradable_vol=temp.loc[open_p]
        # if len(open_p)>1:
        #     raise Exception('多个符合要求价格')
        #     #todo 两个以上申报价格符合上述条件的，使未成交量最小的申报价格为成交价格；仍有两个以上使未成交量最小的申报价格符合上述条件的，其中间价为成交价格。
        # else:
        #     open_p=open_p[0]
        #     max_tradable_vol=max_tradable_vol.values[0]
        #
        # return open_p,max_tradable_vol
        status_bkp = deepcopy(self.__dict__)
        price = self.match_call_auction(is_open=is_open)
        self.__dict__ = status_bkp
        return price

    def match_call_auction(self, is_open=True):
        """
        todo 整合trade进来

        :param is_open:
        :return:
        """
        proc_sequence = {}
        last_bid_p = -1
        last_ask_p = -1
        if is_open:
            timestamp = pd.to_datetime(config.important_times['open_call_auction_end']) + timedelta(milliseconds=10)
        else:
            timestamp = pd.to_datetime(config.important_times['close_call_auction_start']) + timedelta(milliseconds=10)

        while self.get_best_bid() >= self.get_best_ask():
            last_bid_p = self.get_best_bid()
            last_ask_p = self.get_best_ask()

            # 一方无订单
            if last_ask_p == -1 and last_bid_p == -1:
                return (self.last_best_bid + self.last_best_ask) / 2
            elif last_ask_p == -1:
                return self.last_best_bid
            elif last_bid_p == -1:
                return self.last_best_ask

            # 寻找可交易的order
            order_bids_dict = self.book_bid[last_bid_p]  # 最高买价
            order_bid = order_bids_dict[list(order_bids_dict.keys())[0]]  # 时间优先
            order_asks_dict = self.book_ask[last_ask_p]  # 最低卖价
            order_ask = order_asks_dict[list(order_asks_dict.keys())[0]]  # 时间优先

            # 更新order信息
            price = self.open_p if is_open else self.close_p
            can_filled = min(order_bid.quantity - order_bid.filled_quantity,
                             order_ask.quantity - order_ask.filled_quantity)
            order_bid.fill_quantity(timestamp, can_filled, price)
            order_ask.fill_quantity(timestamp, can_filled, price)

            # 更新collections
            self._sub_pv_dict(self.pv_dict_bid, order_bid.price, can_filled)
            self._sub_pv_dict(self.pv_dict_ask, order_ask.price, can_filled)
            # assert self.pv_dict_bid.get(order_bid.price) is not None and self.pv_dict_bid.get(order_bid.price)>=can_filled
            # assert self.pv_dict_ask.get(order_ask.price) is not None and self.pv_dict_ask.get(order_ask.price)>=can_filled
            # self.pv_dict_bid[order_bid.price] -= can_filled
            # self.pv_dict_ask[order_ask.price] -= can_filled
            # if self.pv_dict_bid[order_bid.price] == 0: self.pv_dict_bid.pop(order_bid.price)
            # if self.pv_dict_ask[order_ask.price] == 0: self.pv_dict_ask.pop(order_ask.price)
            self.maintain_collections(order_bid, is_counter_order=True)
            self.maintain_collections(order_ask, is_counter_order=True)
            self.append_to_trade(order_bid, order_ask, can_filled, price,
                                 pd.to_datetime(config.important_times['open_call_auction_end']) + timedelta(
                                     milliseconds=10))

        # 计算开盘价
        # todo:两个以上申报价格符合上述条件的，使未成交量最小的申报价格为成交价格；这一条好像没有作用
        if self.get_best_ask() != last_ask_p and self.get_best_bid() != last_bid_p:  # 仍有两个以上使未成交量最小的申报价格符合上述条件的，其中间价为成交价格。
            new_p = (self.get_best_ask() + self.get_best_bid()) / 2
        elif self.get_best_ask() != last_ask_p:
            new_p = self.get_best_bid()
        elif self.get_best_bid() != last_bid_p:
            new_p = self.get_best_ask()
        else:
            raise Exception('match_call_auction error')

        return new_p
    
    @staticmethod
    def _sub_pv_dict(pv_dict:Union[SortedDict], price, sub_quantity):
        """
        从pv_dict中删除某个订单的unfilled quantity
        Parameters
        ----------
        pv_dict
        sub_quantity:
            要删除的数量

        Returns
        -------

        """
        assert (pv_dict.get(price) is not None) and (pv_dict[price] >= sub_quantity)
        pv_dict[price]-=sub_quantity
        if pv_dict[price] == 0: pv_dict.pop(price)

    def maintain_collections(self, order, is_counter_order):
        """
        order_queue_dict, 
        :param order: 
        :param is_counter_order: 
        :return: 
        """
        self.order_queue_dict[order.seq] = order
        if is_counter_order:  # 只有对手方需要维护已经存储的信息，新的order可以直接在外部loop实现多笔交易
            assert (order.type == OrderTypeInt.limit.value) or (order.type == OrderTypeInt.bop.value)
            if order.is_filled():
                if order.side == OrderSideInt.ask.value:  # counter order is an ask
                    order_queue_dict = self.book_ask[order.price]
                    order_queue_dict.pop(order.seq)
                    if len(order_queue_dict) == 0:
                        self.book_ask.pop(order.price)
                else:
                    order_queue_dict = self.book_bid[order.price]
                    order_queue_dict.pop(order.seq)
                    if len(order_queue_dict) == 0:
                        self.book_bid.pop(order.price)
            else:
                if order.side == OrderSideInt.ask.value:  # counter order is an ask
                    self.book_ask[order.price][order.seq] = order  # 更新信息
                else:
                    self.book_bid[order.price][order.seq] = order  # 更新信息

        else:
            # 一般在外面做append_to_book的逻辑，因为一个order可以在while中连续吃对手单
            pass

    def proc_bop_order(self, order):
        assert order.type == OrderTypeInt.bop.value
        logging.warning(f"proc_limit_order")
        # 直接加入订单簿即可，因为不可能成交
        order.price = self.get_best_ask() if order.side == OrderSideInt.ask.value else self.get_best_bid()
        self.append_to_book(order)

    def proc_limit_order(self, order, mkt_lmt=False):
        assert order.type == OrderTypeInt.limit.value or mkt_lmt
        side = order.side
        price = order.price
        if side == OrderSideInt.bid.value:  # 撮合
            while price >= self.get_best_ask() and not order.is_filled():
                best_ask_p = self.get_best_ask()
                if best_ask_p == -1:  # 没有对手订单
                    break
                order_queue_dict = self.book_ask[best_ask_p]
                counter_order = order_queue_dict[list(order_queue_dict.keys())[0]]
                self.current = self.trade(order, counter_order)

            if not order.is_filled():
                self.append_to_book(order)

        elif side == OrderSideInt.ask.value:  # 撮合
            while price <= self.get_best_bid() and not order.is_filled():
                best_bid_p = self.get_best_bid()
                if best_bid_p == -1:  # 没有对手订单
                    break
                order_queue_dict = self.book_bid[best_bid_p]
                counter_order = order_queue_dict[list(order_queue_dict.keys())[0]]
                self.current = self.trade(order, counter_order)

            if not order.is_filled():
                self.append_to_book(order)

    def proc_market_order(self, order):
        """
        todo 大市价单转为限价单

        .. 上交所：
            3.3.5 市价申报内容应当包含投资者能够接受的最高买价（以下简称买入保护限价）或者最低卖价（以下简称卖出保护限价）。 本所交易系统处理前款规定的市价申报时，买入申报的成交价格和转为限价申报的申报价格不高于买入保护限价，卖出申报的成交价格和转为限价申报的申报价格不低于卖出保护限价。

        .. 深交所：
            3.3.4
            本所 可以 接受下列类型的市价申报：
            （一）对手方最优价格申报；
            （二）本方最优价格申报；
            （三）最优五档即时成交剩余撤销申报；
            （四）即时成交剩余撤销申报；
            （五）全额成交或撤销申报；
            （六）本所规定的其他类型。
            对手方最优价格申报，以申报进入交易主机时集中申报簿中
            对手方队列的最优价格为其申报价格。
            本方最优价格申报，以申报进入交易主机时集中申报簿中本
            方队列的最优价格为其申报价格。
            最优五档即时成交剩余撤销申报，以对手方价格为成交价，
            与申报进入交易主机时集中申报簿中对手方最优五个价位的申
            报队列依次成交，未成交部分自动撤销。
            即时成交剩余撤销申报，以对手方价格为成交价，与申报进
            入交易主机时集中申报簿中对手方所有申报队列依次成交，未成
            交部分自动撤销。
            全额成交或撤
            销申报，以对手方价格为成交价，如与申报进
            入交易主机时集中申报簿中对手方所有申报队列依次成交能够使其完全成交的，则依次成交，否则申报全部自动撤销。
        """
        assert order.type == OrderTypeInt.market.value
        counter = 0
        side = order.side
        if side == OrderSideInt.bid.value:
            while not order.is_filled():
                best_ask_p = self.get_best_ask()
                if best_ask_p == -1:  # 没有对手订单
                    break
                order_queue_dict = self.book_ask[best_ask_p]
                counter_order = order_queue_dict[list(order_queue_dict.keys())[0]]  # 取ordered dict第一个
                self.current = self.trade(order, counter_order)

                # 对手方完全成交
                if counter_order.is_filled():
                    counter += 1  # 吃完了一档

            # if not order.is_filled():
            #     self.append_to_book(order)

        elif side == OrderSideInt.ask.value:
            while not order.is_filled():
                best_bid_p = self.get_best_bid()
                if best_bid_p == -1:  # 没有对手订单
                    break
                order_queue_dict = self.book_bid[best_bid_p]
                counter_order = order_queue_dict[list(order_queue_dict.keys())[0]]
                self.current = self.trade(order, counter_order)

                # 对手方完全成交
                if counter_order.is_filled():
                    counter += 1  # 吃完了一档

            # if not order.is_filled():
            #     self.append_to_book(order)

    def __proc_market_order_bkp(self, order):
        """
        todo 大市价单转为限价单。

        .. [x]: 深交所
            3.3.4
            本所 可以 接受下列类型的市价申报：
            （一）对手方最优价格申报；
            （二）本方最优价格申报；
            （三）最优五档即时成交剩余撤销申报；
            （四）即时成交剩余撤销申报；
            （五）全额成交或撤销申报；
            （六）本所规定的其他类型。
            对手方最优价格申报，以申报进入交易主机时集中申报簿中
            对手方队列的最优价格为其申报价格。
            本方最优价格申报，以申报进入交易主机时集中申报簿中本
            方队列的最优价格为其申报价格。
            最优五档即时成交剩余撤销申报，以对手方价格为成交价，
            与申报进入交易主机时集中申报簿中对手方最优五个价位的申
            报队列依次成交，未成交部分自动撤销。
            即时成交剩余撤销申报，以对手方价格为成交价，与申报进
            入交易主机时集中申报簿中对手方所有申报队列依次成交，未成
            交部分自动撤销。
            全额成交或撤
            销申报，以对手方价格为成交价，如与申报进
            入交易主机时集中申报簿中对手方所有申报队列依次成交能够使其完全成交的，则依次成交，否则申报全部自动撤销。
        """
        assert order.type == OrderTypeInt.market.value
        # if self.exchange=='XSHG':
        counter = 0
        side = order.side
        if side == OrderSideInt.bid.value:
            while not order.is_filled():
                best_ask_p = self.get_best_ask()
                if best_ask_p == -1:  # 没有对手订单
                    break
                order_queue_dict = self.book_ask[best_ask_p]
                counter_order = order_queue_dict[list(order_queue_dict.keys())[0]]  # 取ordered dict第一个
                self.current = self.trade(order, counter_order)

                # 对手方完全成交
                if counter_order.is_filled():
                    counter += 1  # 吃完了一档

            if not order.is_filled():
                self.append_to_book(order)

        elif side == OrderSideInt.ask.value:
            while not order.is_filled():
                best_bid_p = self.get_best_bid()
                if best_bid_p == -1:  # 没有对手订单
                    break
                order_queue_dict = self.book_bid[best_bid_p]
                counter_order = order_queue_dict[list(order_queue_dict.keys())[0]]
                self.current = self.trade(order, counter_order)

                # 对手方完全成交
                if counter_order.is_filled():
                    counter += 1  # 吃完了一档

            if not order.is_filled():
                self.append_to_book(order)
        # elif self.exchange=='XSHE':
        #     counter = 0
        #     side = order.side
        #     if side == OrderSideInt.bid.value:
        #         order.price=self.get_best_ask()
        #         # order.type=OrderTypeInt.limit.value
        #         self.proc_limit_order(order,mkt_lmt=True)
        #
        #
        #     elif side == OrderSideInt.ask.value:
        #         order.price=self.get_best_bid()
        #         # order.type=OrderTypeInt.limit.value
        #         self.proc_limit_order(order,mkt_lmt=True)

    # testit
    def trade(self, order: Order, counter_order: Order, price=None):
        """
        进行单次交易，并且维护各个dict/list
        不需要关心买单还是卖单，也不需要管是市价单还是限价单（因为都是按照对手价成交），是否可进行交易的逻辑放在外面
        counter_order: 必然为限价单，否则价格会因为市价单而变成0
        """
        assert counter_order.type == OrderTypeInt.limit.value
        assert counter_order.seq < order.seq

        can_filled = order.can_filled
        can_filled_counter = counter_order.can_filled
        filled = min(can_filled, can_filled_counter)
        timestamp = order.timestamp
        price = counter_order.price if price is None else price
        current = price
        # 更新order
        order.fill_quantity(timestamp, filled, price)
        counter_order.fill_quantity(timestamp, filled, price)

        # 更新order相关的所有collections。需要注意order是市价单的情况
        pv_dict=self.pv_dict_bid if counter_order.side==OrderSideInt.bid.value else self.pv_dict_ask # 仅需关注counter order，因为order还未加入到pv_dict
        self._sub_pv_dict(pv_dict,counter_order.price,filled)
        # assert pv_dict.get(counter_order.price) is not None and pv_dict.get(counter_order.price) >= filled
        # pv_dict[counter_order.price] -= filled
        # if pv_dict[counter_order.price] == 0: pv_dict.pop(counter_order.price)
        self.maintain_collections(order, is_counter_order=False)
        self.maintain_collections(counter_order, is_counter_order=True)

        # 更新trade记录
        self.append_to_trade(order, counter_order, filled, price, timestamp)

        return current

    # testit
    def match_continues_auction(self, order):
        """
        .. [#]: 上交所
            3.5.1 证券竞价交易按价格优先、时间优先的原则撮合成交。 成交时价格优先的原则为：较高价格买入申报优先于较低价格买入申报，较低价格卖出申报优先于较高价格卖出申报。 成交时时间优先的原则为：买卖方向、价格相同的，先申报者优先于后申报者。先后顺序按交易主机接受申报的时间确定。
            3.5.3 连续竞价时，成交价格的确定原则为： （一）最高买入申报价格与最低卖出申报价格相同，以该价格为成交价格； （二）买入申报价格高于即时揭示的最低卖出申报价格的，以即时揭示的最低卖出申报价格为成交价格； （三）卖出申报价格低于即时揭示的最高买入申报价格的，以即时揭示的最高买入申报价格为成交价格。
            3.3.4 根据市场需要，本所可以接受下列方式的市价申报：
                （一）最优5档即时成交剩余撤销申报，即该申报在对手方实时最优5个价位内以对手方价格为成交价逐次成交，剩余未成交部分自动撤销；
                （二）最优5档即时成交剩余转限价申报，即该申报在对手方实时5个最优价位内以对手方价格为成交价逐次成交，剩余未成交部分按本方申报最新成交价转为限价申报；如该申报无成交的，按本方最优报价转为限价申报；如无本方申报的，该申报撤销；
                （三）本方最优价格申报（BOP, best of party），即该申报以其进入交易主机时，集中申报簿中本方最优报价为其申报价格。本方最优价格申报进入交易主机时，集中申报簿中本方无申报的，申报自动撤销；
                （四）对手方最优价格申报（BOC, best of counterparty) ·，即该申报以其进入交易主机时，集中申报簿中对手方最优报价为其申报价格。对手方最优价格申报进入交易主机时，集中申报簿中对手方无申报的，申报自动撤销；

        .. [#]: 深交所
            3.3.4 本所可以接受下列类型的市价申报：
            （一）对手方最优价格申报；
            （二）本方最优价格申报；
            （三）最优五档即时成交剩余撤销申报；
            （四）即时成交剩余撤销申报；
            （五）全额成交或撤销申报；
            （六）本所规定的其他类型。
            对手方最优价格申报，以申报进入交易主机时集中申报簿中对手方队列的最优价格为其申报价格。
            本方最优价格申报，以申报进入交易主机时集中申报簿中本方队列的最优价格为其申报价格。
            最优五档即时成交剩余撤销申报，以对手方价格为成交价，与申报进入交易主机时集中申报簿中对手方最优五个价位的申报队列依次成交，未成交部分自动撤销。
            即时成交剩余撤销申报，以对手方价格为成交价，与申报进入交易主机时集中申报簿中对手方所有申报队列依次成交，未成
            交部分自动撤销。
            全额成交或撤销申报，以对手方价格为成交价，如与申报进入交易主机时集中申报簿中对手方所有申报队列依次成交能够使其完全成交的，则依次成交，否则申报全部自动撤销。
            3.3.5 市价申报只适用于有价格涨跌幅限制证券连续竞价期间的交易。其他交易时间，交易主机不接受市价申报。
            3.3.6 本方最优价格申报进入交易主机时，集中申报簿中本方无申报的，申报自动撤销。
            其他市价申报类型进入交易主机时，集中申报簿中对手方无申报的，申报自动撤销。
        """
        if order.type == OrderTypeInt.limit.value:
            self.proc_limit_order(order)
        elif order.type == OrderTypeInt.bop.value:
            self.proc_bop_order(order)
        elif order.type == OrderTypeInt.market.value:
            self.proc_market_order(order)
        else:
            raise OrderTypeIntError('reconstruct OrderTypeInt error')

    def __snapshot(self, timestamp, window):
        """
        盘口快照，并存储于history

        Note
        ----
        仅用于校验

        Parameters
        ----------
        timestamp
        window:
            window档盘口数据

        Returns
        -------

        """
        # logging.warning(str(self.__snapshot)+" is deprecated",DeprecationWarning)
        window_ask = min(len(self.book_ask), window)
        window_bid = min(len(self.book_bid), window)
        p_v = {k: self.sum_volume(v) for k, v in self.book_bid.items()[-window_bid:]}
        p_v.update({k: self.sum_volume(v) for k, v in self.book_ask.items()[:window_ask]})
        self.order_book_history_dict1[timestamp] = p_v
        # return this_order_book

    def snapshot(self, timestamp, window):
        """
        盘口快照，并存储于history
        window: window档盘口数据
        """
        # logging.warning(str(self.__snapshot)+" is deprecated",DeprecationWarning)
        window_bid = min(len(self.pv_dict_bid.keys()), window)
        window_ask = min(len(self.pv_dict_ask.keys()), window)
        p_v = {k: v for k, v in self.pv_dict_bid.items()[-window_bid:]}
        p_v.update({k: v for k, v in self.pv_dict_ask.items()[:window_ask]})
        self.order_book_history_dict[timestamp] = p_v

    def get_trade_details(self, last_n=0):
        if last_n == 0:
            last_n = len(self.trade_queue)
        return pd.DataFrame([x.__dict__ for x in self.trade_queue[-last_n:]])

    def get_best_order_book(self):
        best_book_ask = pd.DataFrame([v.__dict__ for k, v in self.book_ask[self.get_best_ask()].items()])
        best_book_bid = pd.DataFrame([v.__dict__ for k, v in self.book_bid[self.get_best_bid()].items()])
        return best_book_bid, best_book_ask

    def check_snapshot(self,timestamp,window):
        """
        订单簿价格、数量匹配，已生成的order_book_history相同
        :return:
        """
        assert ((np.array(list(self.book_bid.keys()))-np.array(list(self.pv_dict_bid.keys())))==0).all() 
        assert ((np.array(list(self.book_ask.keys()))-np.array(list(self.pv_dict_ask.keys())))==0).all() 

        # self.__snapshot(timestamp,window)
        # self.snapshot(timestamp,window)

        snap=list(self.order_book_history_dict.values())[-1]
        snap1=list(self.order_book_history_dict1.values())[-1]
        assert (np.array(list(snap.keys()))-np.array(list(snap1.keys()))==0).all()
        assert (np.array(list(snap.values()))-np.array(list(snap1.values()))==0).all()


    def check_trade_details(self, trade_details, last_n=0):
        """对比trade details"""

        self.my_trade_details = self.get_trade_details(last_n=last_n)
        total_len = len(self.trade_queue)
        start = total_len - last_n if last_n != 0 else 0
        cols = ['bid_seq', 'ask_seq', 'quantity', 'price']
        my_td = self.my_trade_details.loc[:, cols]
        true_td = trade_details.iloc[start:total_len].loc[:, cols]
        try:
            assert not ((my_td.values - true_td.values) != 0).any()
            if last_n == 0: self.my_trade_details = self.sync_seq_trade_details(trade_details, self.my_trade_details)
            return self.my_trade_details
        except:
            # print(td1)
            # print('-' * 20)
            # print(td2)
            true_td.columns = [col + '_true' for col in true_td.columns]
            my_td.index = true_td.index
            first_mismatch_idx = self._first_mismatch_idx(true_td, my_td)
            window_size = 5
            self._mismatch_trade = pd.concat([true_td, my_td], axis=1).iloc[
                                   first_mismatch_idx - window_size:first_mismatch_idx + window_size]
            self._dict_to_df()
            self._mismatch_order = self.order_details.loc[
                self._mismatch_trade[['bid_seq_true', 'ask_seq_true', 'bid_seq', 'ask_seq']].iloc[window_size].values].T
            print(self._mismatch_trade)
            print(self._mismatch_order)
            raise Exception('check_trade_details mismatch')

    def _first_mismatch_idx(self, true_trade_details, my_trade_details):
        first_mismatch_idx = pd.DataFrame((my_trade_details.values - true_trade_details.values) != 0).any(
            axis=1).idxmax()
        return first_mismatch_idx

    def sync_seq_trade_details(self, true_trade_details, my_trade_details):
        """利用真实trade details的seq赋值给我构建的trade details"""
        my_trade_details['seq'] = true_trade_details['seq']
        return my_trade_details

    def _dict_to_df(self):
        """
        将类中存储的dict形式信息转为dataframe
        :return:
        """
        # 最后将order_book_history的列（价格），按照降序排列

        self.order_book_history = pd.DataFrame(self.order_book_history_dict)
        self.order_book_history = self.order_book_history.sort_index(ascending=False).T
        # self.order_book_history1 = pd.DataFrame(self.order_book_history_dict1)
        # self.order_book_history1 = self.order_book_history1.sort_index(ascending=False).T


        self.price_history = pd.DataFrame(self.price_history_dict)
        self.price_history = self.price_history.T
        self.price_history = self.price_history.sort_index(ascending=True)

        self.clean_obh = self._gen_clean_obh()

    def _gen_clean_obh(self):
        """

        :return:
        """
        assert len(self.order_book_history) > 0
        obh = self.order_book_history
        try:
            obh.index = pd.to_datetime(obh.index)
        except:
            obh = obh.T
            obh.index = pd.to_datetime(obh.index)
        obh = obh.sort_index(ascending=True)
        obh.columns = obh.columns.astype(float)
        obh_v = LobCleanObhPreprocessor.split_volume(obh)
        obh_p = LobCleanObhPreprocessor.split_price(obh)
        clean_obh = pd.concat([obh_p, obh_v], axis=1).ffill()
        return clean_obh

    def _exception_cleanup(self):
        self._dict_to_df()
        self.my_trade_details = self.check_trade_details(self.my_trade_details)

    def calc_vol_tov(self, trade_details):
        if trade_details.index.name != 'timestamp':
            trade_details = trade_details.set_index('timestamp', drop=False)
        trade_details.index = pd.to_datetime(trade_details.index)
        volume = trade_details.groupby(level=0)['quantity'].sum().rename('volume')
        cum_vol = volume.cumsum().rename('cum_vol')
        turnover = (trade_details['quantity'] * trade_details['price']).groupby(level=0).sum().rename('turnover')
        cum_turnover = turnover.cumsum().rename('cum_turnover')
        vol_tov = pd.concat([volume, cum_vol, turnover, cum_turnover], axis=1)
        # res = LobTimePreprocessor.del_untrade_time(vol_tov, cut_tail=True)
        # if len(vol_tov.loc[:config.important_times['continues_auction_am_start']]) >= 1:
        #     head = vol_tov.loc[:config.important_times['continues_auction_am_start']].iloc[-1]
        #     head['volume'] = head['turnover'] = 0
        #     head = head.rename(config.important_times['continues_auction_am_start'])
        #     res = pd.concat([head.to_frame().T, res]).head(10)
        return vol_tov

    # testit
    def reconstruct(self, order_details, trade_details):
        order = None
        DEBUG = False
        self.order_details = order_details
        self.trade_details = trade_details

        # 主程序main
        canceled_order_details = order_details.loc[order_details['canceled_seq'] != 0].set_index('canceled_seq',
                                                                                                 drop=False).sort_index()
        idx_order_details = set(order_details.index.tolist())
        idx_canceled_order_details = set(canceled_order_details.index.tolist())
        idx_trade_details = set(trade_details.index.tolist())
        self.idx_canceled_order_details = idx_canceled_order_details

        for seq in tqdm(range(max(order_details.index.max(), canceled_order_details.index.max(),
                                  trade_details.index.max()) + 1)):
            if DEBUG and seq >= 100000:
                break

            ################ 主动撤单 ################
            if seq in idx_canceled_order_details:
                order_info = canceled_order_details.loc[seq]
                order = Order(order_info['seq'], order_info['timestamp'], order_info['side'], order_info['type'],
                              order_info['quantity'], order_info['price'],
                              filled_quantity=order_info['filled_quantity'], filled_amount=order_info['filled_amount'],
                              last_traded_timestamp=order_info['last_traded_timestamp'],
                              canceled_timestamp=order_info['canceled_timestamp'],
                              canceled_seq=order_info['canceled_seq'])
                self.order = order

                # logging.warning(f"cancel order {str(order)}")
                try:
                    if order.side == OrderSideInt.bid.value:  # 买
                        self.book_bid[order.price].pop(order.seq)
                        if len(self.book_bid[order.price]) == 0:
                            self.book_bid.pop(order.price)
                            
                        # self.pv_dict_bid[order.price]-=
                        self._sub_pv_dict(self.pv_dict_bid,order.price,order.can_filled)
                    if order.side == OrderSideInt.ask.value:  # 卖
                        self.book_ask[order.price].pop(order.seq)
                        if len(self.book_ask[order.price]) == 0:
                            self.book_ask.pop(order.price)
                        self._sub_pv_dict(self.pv_dict_ask,order.price,order.can_filled)

                except Exception as e:
                    self._exception_cleanup()
                    raise e
                # finally func
                ...
                continue
            ######################################

            ################ 下单 ################
            elif seq in idx_order_details:
                order_info = order_details.loc[seq]
            else:
                # order_info仍旧可能是none，因为有些成交不需要重新生成order，而seq是根据成交来的。
                # 即此刻没有新的order，但是trade产生了新的seq
                order_info = None
                assert seq in idx_trade_details
                continue

            # 有新的order，不能通过details直接设置撤单等信息，都得是全新的
            timestamp = order_info['timestamp']
            order = Order(order_info['seq'], order_info['timestamp'], order_info['side'], order_info['type'],
                          order_info['quantity'], order_info['price'], filled_quantity=0, filled_amount=0,
                          last_traded_timestamp=pd.NaT, canceled_timestamp=pd.NaT, canceled_seq=0)
            self.order_queue_dict[order.seq] = order
            self.order = order

            # 开盘集合竞价（左闭右开）
            # 上交所：只能是申报限价单：3.3.6 市价申报只适用于连续竞价期间的交易，本所另有规定的除外。
            # 深交所：3.3.6 本方最优价格申报进入交易主机时，集中申报簿中本方无申报的，申报自动撤销。
            if timestamp >= config.important_times['open_call_auction_start'] and timestamp < config.important_times[
                'open_call_auction_end']:
                if order.type == OrderTypeInt.bop.value:  # 处理本方最优订单
                    if order.side == OrderSideInt.bid.value and self.get_best_bid() != -1:
                        order.price = self.get_best_bid()
                        logging.warning(f"bop order")
                    elif order.side == OrderSideInt.ask.value and self.get_best_ask() != -1:
                        order.price = self.get_best_ask()
                        logging.warning(f"bop order")
                    else:
                        pass  # 不加入订单簿
                        # raise NotImplementedError("集合竞价阶段")
                self.append_to_book(order)
                continue

            # 9:25-9:30
            elif timestamp >= config.important_times['open_call_auction_end'] and timestamp < config.important_times[
                'continues_auction_am_start']:
                # self.append_to_book(order)
                continue

            # 连续竞价（左闭右开）
            elif timestamp >= config.important_times['continues_auction_am_start'] and timestamp < \
                    config.important_times[
                        'continues_auction_pm_end']:
                # if DEBUG: return None
                # 开盘集合竞价尚未撮合
                if not self.open_call_auction_matched:
                    print(f'开盘集合竞价撮合')
                    self.open_p = self.calc_p_call_auction(is_open=True)
                    self.match_call_auction(is_open=True)  # 进行按序重构交易
                    self.current = self.open_p
                    self.my_trade_details = self.check_trade_details(trade_details)
                    self.open_call_auction_matched = True
                    print('开盘集合竞价撮合结束，开盘价为', self.open_p)
                    # 这里不能continue，因为该if是新的开盘后order触发的，因此还需要处理该order

                # 逐笔申报，逐笔挂单/成交；价格优先，时间优先
                self.match_continues_auction(order)

                # log
                if self.snapshot_ABtest:
                    self.__snapshot(order.timestamp, window=self.snapshot_window)
                self.snapshot(order.timestamp, window=self.snapshot_window)
                self.price_history_dict[self.price_history_idx] = {'seq': seq, 'timestamp': timestamp,
                                                                   'current': self.current}
                self.price_history_idx += 1

                if seq % 10000 == 0:
                    self.check_trade_details(trade_details, last_n=50)
                    if self.snapshot_ABtest:
                        self.check_snapshot(order.timestamp,window=self.snapshot_window)
                continue

            # 收盘集合竞价（左闭右开）
            elif timestamp >= config.important_times['close_call_auction_start'] and timestamp < config.important_times[
                'close_call_auction_end']:
                self.append_to_book(order)
                continue

            else:
                raise Exception('reconstruct:' + str(timestamp) + 'out of circumstance')
            ######################################

        ################### 收盘竞价 ###################
        try:
            if not self.close_call_auction_matched:  # 开盘集合竞价尚未撮合
                print(f'收盘集合竞价撮合')
                self.close_p = self.calc_p_call_auction(is_open=False)
                self.match_call_auction(is_open=False)  # 进行按序重构交易
                self.current = self.close_p
                self.my_trade_details = self.check_trade_details(trade_details)
                self.close_call_auction_matched = True
                print('收盘集合竞价撮合结束，收盘价为', self.close_p)
        except Exception as e:
            print(order)
            raise e
        ######################################

        self._dict_to_df()
        self.my_trade_details = self.check_trade_details(trade_details)
        self.vol_tov = self.calc_vol_tov(self.my_trade_details)
