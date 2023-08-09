# -*- coding=utf-8 -*-
# @File     : broker.py
# @Time     : 2023/8/2 12:22
# @Author   : EvanHong
# @Email    : 939778128@qq.com
# @Project  : 2023.06.08超高频上证50指数计算
# @Description:
from typing import Callable, Dict, List, Union, Sequence, Tuple, Type, Union

from collections import defaultdict
from copy import deepcopy

import numpy as np
import pandas as pd

from support import *
from config import *
from datafeed import LobDataFeed


class OrderTypeIntError(TypeError):
    def __init__(self, *arg):
        self.args = arg


class Trade(object):
    def __init__(self,
                 seq,
                 bid_seq,
                 ask_seq,
                 quantity,
                 price,
                 timestamp=pd.to_datetime(f'{date1} 09:25:00.000000')):
        self.seq = seq
        self.bid_seq = bid_seq
        self.ask_seq = ask_seq
        self.quantity = quantity
        self.price = price
        self.timestamp = timestamp

    def __str__(self):
        return str(self.__dict__)


class Order(object):
    def __init__(self,
                 seq,
                 timestamp,
                 side,
                 type_,
                 quantity,
                 price,
                 filled_quantity=0,
                 filled_amount=0,
                 last_traded_timestamp=pd.NaT,
                 canceled_timestamp=pd.NaT,
                 canceled_seq=0,
                 stop_price=None,
                 sl_price=None,
                 tp_price=None,
                 trade_seq=None,
                 ):
        self.seq = seq
        self.timestamp = timestamp
        self.side = side
        self.type = type_
        self.quantity = quantity
        self.price = price
        self.filled_quantity = filled_quantity
        self.filled_amount = filled_amount
        self.last_traded_timestamp = last_traded_timestamp
        self.canceled_timestamp = canceled_timestamp
        self.canceled_seq = canceled_seq
        self.filled = False
        self.canceled = False

        self.__stop_price = stop_price  # Order stop price for stop-limit/stop-market order, otherwise None if no stop was set, or the stop price has already been hit.
        self.__sl_price = sl_price  # A stop-loss price at which, if set, a new contingent stop-market order will be placed upon the Trade following this order's execution. See also Trade.sl.
        self.__tp_price = tp_price  # A take-profit price at which, if set, a new contingent limit order will be placed upon the Trade following this order's execution. See also Trade.tp.
        self.__parent_trade = trade_seq
        # self.__tag = tag

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

        assert quantity <= self.can_filled()
        self.last_traded_timestamp = timestamp
        self.filled_quantity += quantity
        self.filled_amount += price * quantity
        # 市价单平均价格会变化
        if self.type == OrderTypeInt['market'].value:
            self.price = self.filled_amount / self.filled_quantity  # 平均价格
        self._set_filled()

    def _set_filled(self):
        if self.quantity == self.filled_quantity:
            self.filled = True

    def is_filled(self):
        if self.quantity == self.filled_quantity:
            self.filled = True
            return True
        else:
            return False

    def can_filled(self):
        """quantity to fill"""
        return self.quantity - self.filled_quantity

    def cancel(self, canceled_seq, canceled_timestamp):
        self.canceled_seq = canceled_seq
        self.canceled_timestamp = canceled_timestamp
        self.canceled = True

    def to_dict(self):
        return self.__dict__


class Transaction(object):
    def __init__(self):
        pass


class Position:
    """
    only for assets like stocks or options, but not for cash
    """

    # def __init__(self, broker: 'Broker'):
    #     self.__broker = broker

    def __init__(self, kind: str = 'stock'):
        self.kind = kind

    def __bool__(self):
        return self.size != 0

    def add(self, trade):
        # self.stk_name=stk_name
        # self.volume=volume # long if positive, short if negative
        # self.price=price
        pass

    @property
    def size(self) -> float:
        """Position size in units of asset. Negative if position is short."""
        return sum(trade.size for trade in self.__broker.trades)

    @property
    def pl(self) -> float:
        """Profit (positive) or loss (negative) of the current position in _cash units."""
        return sum(trade.pl for trade in self.__broker.trades)

    @property
    def pl_pct(self) -> float:
        """Profit (positive) or loss (negative) of the current position in percent."""
        weights = np.abs([trade.size for trade in self.__broker.trades])
        weights = weights / weights.sum()
        pl_pcts = np.array([trade.pl_pct for trade in self.__broker.trades])
        return (pl_pcts * weights).sum()

    @property
    def is_long(self) -> bool:
        """True if the position is long (position size is positive)."""
        return self.size > 0

    @property
    def is_short(self) -> bool:
        """True if the position is short (position size is negative)."""
        return self.size < 0

    def close(self, portion: float = 1.):
        """
        Close portion of position by closing `portion` of each active trade. See `Trade.close`.
        """
        for trade in self.__broker.trades:
            trade.close(portion)

    def __repr__(self):
        return f'<Position: {self.size} ({len(self.__broker.trades)} trades)>'


class BaseBroker(object):
    def __init__(self, *args, **kwargs):
        self.args = args
        for key, value in kwargs.items():
            self.__setattr__(key, value)


class Broker(BaseBroker):
    def __init__(self,
                 data: Union[dict, defaultdict] = None,
                 cash=1e6,
                 commission=1e-3,
                 exclusive_orders=[],
                 *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.data = data

        self._cash = cash
        self._commission = commission
        self.usage_limit_pct = 0.01
        self.max_qty = 2e4
        self.verbose = False
        self.cash_from_short = 0
        self.slippage = 0

        # self.transactions = pd.DataFrame(['timestamp','type','pair_codes', 'qty','price','status']).set_index('timestamp')
        # self.positions=pd.DataFrame(columns=['pair_codes', 'qty']).set_index('pair_codes') #shares
        self.seq = 0
        self.transactions = {}
        self.positions = defaultdict(int)
        self.net_value = pd.DataFrame(columns=['timestamp', 'netvalue']).set_index('timestamp')
        self.order_pending = []

        self.orders: Dict[Order] = {}
        self.trades: Dict[Trade] = {}
        self.closed_orders: Dict[Order] = {}
        self.closed_trades: Dict[Trade] = {}
        self.position = Position(kind='stock')

        # self._leverage = 1 / margin
        # self._trade_on_close = trade_on_close
        # self._hedging = hedging
        self._exclusive_orders = exclusive_orders

    def __repr__(self):
        return f'<Broker: {self._cash:.0f}{self.position.pl:+.1f} ({len(self.trades)} trades)>'

    def load_data(self, clean_obh_dict: Union[dict, defaultdict] = None):
        """

        :param clean_obh_dict: dict, {date:{stk_name:ret <pd.DataFrame>}}
        :param signals: dict, {date:all_signals}
        :return:
        """
        self.clean_obh_dict = clean_obh_dict

    def execute(self, signal):
        """

        :param signal: 指令，即signal df中的信息
        :return:
        """
        side = signal['side']
        type = signal['type']
        price_limit = signal['price_limit']
        volume = signal['volume']
        stk_name = signal['stk_name']
        timestamp = signal['timestamp']
        action = signal['action']
        seq = signal['seq']

        if type == OrderTypeInt.market.value:  # 市价单，todo 最优五档？？？
            if side == 1:  # for long order
                pass
            elif side == -1:  # for short order
                pass
            elif side == 0:
                pass
            else:
                raise NotImplementedError("side not implemented")

    def batch_execute(self, signals, date, stk_names):
        revenue_dict = defaultdict(dict)  # dict of dict
        ret_dict = defaultdict(dict)  # dict of dict
        aligned_signals_dict = defaultdict(dict)  # dict of dict
        sig_dates = signals['timestamp'].apply(lambda x: str(x.date()))
        sig_stk_names = signals['stk_name'].unique()
        # for sig_date in sorted(sig_dates.unique().tolist()):
        #     for sig_stk_name in sig_stk_names:
        for sig_date in [date]:
            for sig_stk_name in stk_names:
                _signals = signals.loc[np.logical_and(sig_date == sig_dates, signals['stk_name'] == sig_stk_name)]

                # 只有side才是决定是否开仓的真正信号，此处的open close用于标识开平仓时间
                _open_signals = _signals.loc[_signals['action'] == 'open']
                _close_signals = _signals.loc[_signals['action'] == 'close']
                _aligned_signals = pd.merge(_open_signals, _close_signals, how='inner', left_on='seq', right_on='seq',
                                            suffixes=('_open', '_close'))

                _aligned_signals_long = _aligned_signals.loc[_aligned_signals['side_open'] == 1]
                _aligned_signals_short = _aligned_signals.loc[_aligned_signals['side_open'] == -1]
                _aligned_signals_hold = _aligned_signals.loc[_aligned_signals['side_open'] == 0]

                temp = self.clean_obh_dict[sig_date][sig_stk_name]
                b1p = temp[str(LobColTemplate('b', 1, 'p'))]
                b1v = temp[str(LobColTemplate('b', 1, 'v'))]
                a1p = temp[str(LobColTemplate('a', 1, 'p'))]
                a1v = temp[str(LobColTemplate('a', 1, 'v'))]
                current = temp[str(LobColTemplate().current)]

                # 计算收益，如果为正则代表收益为正
                long_time=_aligned_signals_long['timestamp_open'].values
                short_time=_aligned_signals_short['timestamp_open'].values
                hold_time=_aligned_signals_hold['timestamp_open'].values

                long_revenue = b1p.loc[_aligned_signals_long['timestamp_close']].values - a1p.loc[
                    _aligned_signals_long['timestamp_open']].values
                long_ret = b1p.loc[_aligned_signals_long['timestamp_close']].values / a1p.loc[
                    _aligned_signals_long['timestamp_open']].values - 1
                short_revenue = b1p.loc[_aligned_signals_short['timestamp_open']].values - a1p.loc[
                    _aligned_signals_short['timestamp_close']].values
                short_ret = b1p.loc[_aligned_signals_short['timestamp_open']].values / a1p.loc[
                    _aligned_signals_short['timestamp_close']].values - 1
                long_revenue=pd.Series(long_revenue,index=long_time)
                long_ret=pd.Series(long_ret,index=long_time)
                short_revenue=pd.Series(short_revenue,index=short_time)
                short_ret=pd.Series(short_ret,index=short_time)
                hold_revenue = pd.Series(np.zeros_like(_aligned_signals_hold['timestamp_open'],dtype=float),
                                         index=hold_time)
                hold_ret = pd.Series(np.zeros_like(_aligned_signals_hold['timestamp_open'],dtype=float),
                                     index=hold_time)

                revenue = pd.concat([long_revenue, short_revenue,hold_revenue], axis=0)
                ret = pd.concat([long_ret, short_ret,hold_ret], axis=0)
                _aligned_signals = pd.concat([_aligned_signals_long, _aligned_signals_short,_aligned_signals_hold], axis=0)

                revenue_dict[sig_date][sig_stk_name] = revenue
                ret_dict[sig_date][sig_stk_name] = ret
                aligned_signals_dict[sig_date][sig_stk_name] = _aligned_signals

        return revenue_dict, ret_dict, aligned_signals_dict

    def update_netvalue(self, date):
        # netvalue=self._cash+self.cash_from_short
        netvalue = self._cash
        prices = self.data.loc[date]
        for k, v in self.positions.items():
            pair_codes = k
            qty = v
            # print(prices[pair_codes],qty)
            price = prices[pair_codes]
            netvalue += price * qty
        print(f'netvalue {date} {netvalue}')
        self.net_value = pd.concat(
            [self.net_value,
             pd.DataFrame(data={'timestamp': date, 'netvalue': netvalue}, index=[0]).set_index('timestamp')])

    def calc_net_value(self, data, last_net_value, control_drawdown=False, drawdown_threshold=-0.1, level='portfolio',
                       weight=None):
        """
        每个调仓周期内单独计算净值以及止损
        param: ret, changeRatio
        param: last_net_value, 上一期末组合净值
        """

        def weighted_net_value(data, weight):
            if data.isna().any().any() or weight.isna().any():
                print("warning !!!! calc_net_value weighted_net_value ret,weight has nan")
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

    def get_sign(self, type):
        if type == 'long':
            return 1
        elif type == 'short':
            return -1

    def open_position(self, timestamp, symbol, qty, type):
        sign = self.get_sign(type)
        price = self.data.loc[timestamp][symbol]
        if self._cash - sign * qty * price < 0:
            print('no more _cash')
            return 'fail'

        self._cash -= sign * qty * price
        self.positions[symbol] += sign * qty
        if self.positions[symbol] == 0:
            del self.positions[symbol]
        new = {'timestamp': timestamp, 'type': type, 'pair_codes': symbol, 'qty': qty, 'price': price,
               'status': 'sucess'}
        self.transactions.append(new)
        print(f"open {new}")
        return 'sucess'

    def close_position(self, timestamp, symbol, qty, type):
        if symbol not in self.positions.keys():
            print('No position to close', symbol)
            return 'fail'

        sign = self.get_sign(type)
        price = self.data.loc[timestamp][symbol]
        if type == 'long':
            if self.positions[symbol] < qty:
                print('Not enough shares to close long')
                return 'fail'

        elif type == 'short':
            if abs(self.positions[symbol]) < qty:
                print('Not enough shares to close short')
                return 'fail'

        self._cash += sign * qty * price
        self.positions[symbol] -= sign * qty
        if self.positions[symbol] == 0:
            del self.positions[symbol]
        new = {'timestamp': timestamp, 'type': type, 'pair_codes': symbol, 'qty': qty, 'price': price,
               'status': 'sucess'}
        self.transactions.append(new)
        print(f"close {new}")
        return 'sucess'

    def run(self):
        """
        todo 手续费，中间价，时间拉长，看论文

        :return:
        """
        # for ret in self.ret:
        #     signals = self.strategies.generate_signals(ret)
        #     orders = self.broker.execute_signals(signals)
        #     self.observer.record(orders, self.ret, self.broker)
        # self.statistics.report(self.observer)

        for date in self.data.index:
            triggers = self.signals.loc[date]
            prices = self.data.loc[date]

            # close pended
            for symbol in self.order_pending:
                if symbol in self.positions.keys():
                    q = self.positions[symbol]
                    res = self.close_position(date, symbol, q, 'long')
                    if res == 'fail':
                        self.order_pending.append(symbol)
                    else:
                        self.order_pending.remove(symbol)

            # close long
            close_long_list = triggers.loc[triggers['close_long'] == 1, 'pair_codes']
            if len(close_long_list) > 0:
                # cash_buy1stk=self._cash*self.usage_limit_pct/len(close_short_list)
                # qty=cash_buy1stk/prices.loc[close_short_list]
                # print(qty)
                for symbol in close_long_list:
                    if symbol in self.positions.keys():
                        q = self.positions[symbol]
                        res = self.close_position(date, symbol, q, 'long')
                        if res == 'fail':
                            self.order_pending.append(symbol)

            # open short
            open_short_list = triggers.loc[triggers['open_short'] == 1, 'pair_codes']
            if len(open_short_list) > 0:
                cash_buy1stk = self._cash * self.usage_limit_pct / len(open_short_list)
                qty = abs(cash_buy1stk / prices.loc[open_short_list])
                # print(qty)
                for symbol in open_short_list:
                    q = qty[symbol]
                    q = min(self.max_qty, q)
                    self.open_position(date, symbol, q, 'short')

            # open long
            open_long_list = triggers.loc[triggers['open_long'] == 1, 'pair_codes']
            if len(open_long_list) > 0:
                cash_buy1stk = self._cash * self.usage_limit_pct / len(open_long_list)
                qty = abs(cash_buy1stk / prices.loc[open_long_list])
                # print(qty)
                for symbol in open_long_list:
                    q = qty[symbol]
                    q = min(self.max_qty, q)
                    self.open_position(date, symbol, q, 'long')

            # close short
            close_short_list = triggers.loc[triggers['close_short'] == 1, 'pair_codes']
            if len(close_short_list) > 0:
                # cash_buy1stk=self._cash*self.usage_limit_pct/len(close_short_list)
                # qty=cash_buy1stk/prices.loc[close_short_list]
                # print(qty)
                for symbol in close_short_list:
                    if symbol in self.positions.keys():
                        q = self.positions[symbol]
                        self.close_position(date, symbol, q, 'short')

            # update info
            self.update_netvalue(date)

            if self.verbose:
                print(self._cash, self.transactions, self.positions, self.net_value)
