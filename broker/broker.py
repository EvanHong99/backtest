# -*- coding=utf-8 -*-
# @File     : broker.py
# @Time     : 2023/8/2 12:22
# @Author   : EvanHong
# @Email    : 939778128@qq.com
# @Project  : 2023.06.08超高频上证50指数计算
# @Description:
import logging
from typing import Dict, List, Union

from collections import defaultdict

from backtest.support import *
from backtest.config import *
from backtest.broker.orders import Order
from backtest.broker.trades import Trade
from backtest.recorders.transactions import Transaction
from backtest.recorders.position import Position
from backtest.recorders.portfolio import *
from backtest.datafeeds.datafeed import PandasOHLCDataFeed


# class OrderTypeIntError(TypeError):
#     def __init__(self, *arg):
#         self.args = arg


# class Trade(object):
#     def __init__(self,
#                  seq,
#                  bid_seq,
#                  ask_seq,
#                  quantity,
#                  price,
#                  timestamp):
#         self.seq = seq
#         self.bid_seq = bid_seq
#         self.ask_seq = ask_seq
#         self.quantity = quantity
#         self.price = price
#         self.timestamp = timestamp
#
#     def __str__(self):
#         return str(self.__dict__)

# class Order(object):
#     def __init__(self, seq, timestamp, side, type_, quantity, price, filled_quantity=0, filled_amount=0,
#                  last_traded_timestamp=pd.NaT, canceled_timestamp=pd.NaT, canceled_seq=0):
#         """
#
#         Parameters
#         ----------
#         seq :
#         timestamp :
#         side :
#         type_ :
#         quantity :
#         price :
#         filled_quantity :
#         filled_amount :
#         last_traded_timestamp :
#         canceled_timestamp :
#         canceled_seq :
#         """
#         self.seq = seq
#         self.timestamp = timestamp
#         self.side = side
#         self.type = type_
#         self.quantity = quantity
#         self.price = price
#         self.filled_quantity = filled_quantity
#         self.can_filled = self.quantity - self.filled_quantity
#         self.filled_amount = filled_amount
#         self.last_traded_timestamp = last_traded_timestamp
#         self.canceled_timestamp = canceled_timestamp
#         self.canceled_seq = canceled_seq
#         self.filled = False
#         self.canceled = False
#
#     def __str__(self):
#         return "seq-{}, {}: {} @ {}. side={},type={},filled_quantity={},filled_amount={},\n last_traded_timestamp={},canceled_timestamp={},canceled_seq={}\n".format(
#             self.seq, self.timestamp, self.quantity, self.price, OrderSideInt(self.side).name,
#             OrderTypeInt(self.type).name, self.filled_quantity, self.filled_amount, self.last_traded_timestamp,
#             self.canceled_timestamp, self.canceled_seq)
#
#     def fill_quantity(self, timestamp, quantity, price):
#         """
#         全部/部分fill放在orderbook对象中处理，进而保证函数功能的原子性
#
#         Parameters
#         ----------
#         timestamp :
#         quantity :
#         price :
#
#         Returns
#         -------
#
#         """
#         if self.filled:
#             raise Exception('already filled ' + self.__str__())
#         if self.canceled:
#             raise Exception('already canceled ' + self.__str__())  # 限价单则限制价格
#         #         if self.type==OrderTypeInt['limit'].value:
#         #             assert self.price==price
#
#         assert quantity <= self.can_filled
#         self.last_traded_timestamp = timestamp
#         self.filled_quantity += quantity
#         self.filled_amount += price * quantity
#         self.can_filled -= quantity
#         # 市价单平均价格会变化
#         if self.type == OrderTypeInt['market'].value:
#             self.price = self.filled_amount / self.filled_quantity  # 平均价格
#         self._set_filled()
#         return quantity
#
#     def _set_filled(self):
#         if self.quantity == self.filled_quantity:
#             self.filled = True
#
#     def is_filled(self):
#         if self.quantity == self.filled_quantity:
#             self.filled = True
#             return True
#         else:
#             return False
#
#     # def can_filled(self):
#     #     """quantity to fill"""
#     #     return self.quantity - self.filled_quantity
#
#     def cancel(self, canceled_seq, canceled_timestamp):
#         self.canceled_seq = canceled_seq
#         self.canceled_timestamp = canceled_timestamp
#         self.canceled = True
#
#     def to_dict(self):
#         return self.__dict__

# class Transaction(object):
#     def __init__(self):
#         pass


# class Position:
#     """
#     only for assets like stocks or options, but not for cash
#     """
#
#     # def __init__(self, broker: 'Broker'):
#     #     self.__broker = broker
#
#     def __init__(self, kind: str = 'stock'):
#         self.kind = kind
#
#     def __bool__(self):
#         return self.size != 0
#
#     def add(self, trade):
#         # self.stk_name=stk_name
#         # self.volume=volume # long if positive, short if negative
#         # self.price=price
#         pass
#
#     @property
#     def size(self) -> float:
#         """Position size in units of asset. Negative if position is short."""
#         return sum(trade.size for trade in self.__broker.trades)
#
#     @property
#     def pl(self) -> float:
#         """Profit (positive) or loss (negative) of the current position in _cash units."""
#         return sum(trade.pl for trade in self.__broker.trades)
#
#     @property
#     def pl_pct(self) -> float:
#         """Profit (positive) or loss (negative) of the current position in percent."""
#         weights = np.abs([trade.size for trade in self.__broker.trades])
#         weights = weights / weights.sum()
#         pl_pcts = np.array([trade.pl_pct for trade in self.__broker.trades])
#         return (pl_pcts * weights).sum()
#
#     @property
#     def is_long(self) -> bool:
#         """True if the position is long (position size is positive)."""
#         return self.size > 0
#
#     @property
#     def is_short(self) -> bool:
#         """True if the position is short (position size is negative)."""
#         return self.size < 0
#
#     def close(self, portion: float = 1.):
#         """
#         Close portion of position by closing `portion` of each active trade. See `Trade.close`.
#         """
#         for trade in self.__broker.trades:
#             trade.close(portion)
#
#     def __repr__(self):
#         return f'<Position: {self.size} ({len(self.__broker.trades)} trades)>'


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

    def load_data(self, clean_obh_dict: Union[dict, defaultdict,pd.DataFrame,PandasOHLCDataFeed] = None):
        """

        :param clean_obh_dict: dict, {date:{stk_name:ret <pd.DataFrame>}}
        :return:
        """
        self.clean_obh_dict = clean_obh_dict
        if (self.clean_obh_dict is None) or (self.clean_obh_dict == 0):
            logging.warning("self.clean_obh_dict has no data", UserWarning)

    def execute(self, signal):
        """
        执行单个指令
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

        if type == OrderTypeInt.market.value:  # 市价单，fixme 最优五档？？？
            if side == 1:  # for long order
                pass
            elif side == -1:  # for short order
                pass
            elif side == 0:
                pass
            else:
                raise NotImplementedError("side not implemented")

    def _meta_batch_execute(self, _signals: pd.DataFrame, date: str, stk_name: str):
        """
        批量处理信号，但仅针对单日单只个股信号
        Parameters
        ----------
        _signals : pd.DataFrame
            单日单只个股信号
        date
        stk_name

        Returns
        -------

        """
        # 只有side才是决定是否开仓的真正信号，此处的open close用于标识开平仓时间
        _open_signals = _signals.loc[_signals['action'] == 'open']
        _close_signals = _signals.loc[_signals['action'] == 'close']
        _aligned_signals = pd.merge(_open_signals, _close_signals, how='inner', left_on='seq', right_on='seq',
                                    suffixes=('_open', '_close'))

        _aligned_signals_long = _aligned_signals.loc[_aligned_signals['side_open'] == 1]
        _aligned_signals_short = _aligned_signals.loc[_aligned_signals['side_open'] == -1]
        _aligned_signals_hold = _aligned_signals.loc[_aligned_signals['side_open'] == 0]

        temp = self.clean_obh_dict[date][stk_name].asfreq(freq=min_freq, method='ffill')
        b1p = temp[str(LobColTemplate('b', 1, 'p'))]
        b1v = temp[str(LobColTemplate('b', 1, 'v'))]
        a1p = temp[str(LobColTemplate('a', 1, 'p'))]
        a1v = temp[str(LobColTemplate('a', 1, 'v'))]
        current = temp[str(LobColTemplate().current)]

        # 计算收益，如果为正则代表收益为正
        long_open_time = _aligned_signals_long['timestamp_open'].values
        short_open_time = _aligned_signals_short['timestamp_open'].values
        hold_open_time = _aligned_signals_hold['timestamp_open'].values
        long_close_time = _aligned_signals_long['timestamp_close'].values
        short_close_time = _aligned_signals_short['timestamp_close'].values
        hold_close_time = _aligned_signals_hold['timestamp_close'].values

        long_revenue = b1p.loc[long_close_time].values - a1p.loc[long_open_time].values
        long_ret = b1p.loc[long_close_time].values / a1p.loc[long_open_time].values - 1
        short_revenue = b1p.loc[short_open_time].values - a1p.loc[short_close_time].values
        short_ret = b1p.loc[short_open_time].values / a1p.loc[short_close_time].values - 1

        long_revenue = pd.Series(long_revenue, index=long_open_time)
        long_ret = pd.Series(long_ret, index=long_open_time)
        short_revenue = pd.Series(short_revenue, index=short_open_time)
        short_ret = pd.Series(short_ret, index=short_open_time)
        hold_revenue = pd.Series(np.zeros_like(hold_open_time, dtype=float), index=hold_open_time)
        hold_ret = pd.Series(np.zeros_like(hold_open_time, dtype=float), index=hold_open_time)

        revenue = pd.concat([long_revenue, short_revenue, hold_revenue], axis=0)
        ret = pd.concat([long_ret, short_ret, hold_ret], axis=0)
        _aligned_signals = pd.concat([_aligned_signals_long, _aligned_signals_short, _aligned_signals_hold], axis=0)

        return revenue, ret, _aligned_signals

    def batch_execute(self, signals: pd.DataFrame, use_dates: List[str] = None, use_stk_names: List[str] = None):
        """
        批量执行指令，无法画出净值曲线

        ----
        需要提前使用Broker.load_data()函数买卖量价信息历史（至少1档）存于Broker.clean_obh_dict <{date:{stk_name:ret <pd.DataFrame>}}>

        Parameters
        ----------
        signals : pd.DataFrame
            columes: [timestamp	side	type	price_limit	volume	stk_name	action	seq]
            stk_name+seq用于唯一识别一次交易，可以有open和close两种actions，side==1<->long，side==-1<->short，side==0<->no action（优先级高于action）
            type为订单类型，服从米筐订单类型定义，如limit order / market order，具体查看support.OrderTypeInt。todo: 更加通用化的type表示形式
        use_dates: list of str
            回测日期
        use_stk_names: list of str
            回测股票简称（中文），如`"贵州茅台"`

        Returns
        -------
        revenue_dict:dict, {stk_name: revenue}
            收益
        ret_dict:dict, {stk_name: revenue}
            收益率
        aligned_signals_dict: deprecated
        
        Examples
        --------
        >>> self.clean_ohb_dict['2022-06-29']['贵州茅台']
                a5_p	a4_p	a3_p	a2_p	a1_p	b1_p	b2_p	b3_p	b4_p	b5_p	a5_v	a4_v	a3_v	a2_v	a1_v	b1_v	b2_v	b3_v	b4_v	b5_v	current
        9:30:00.00	1947.87	1947.44	1947	1946	1945	1942.7	1942.6	1942	1941	1940.95	100	100	400	300	900	2290	100	100	100	100	1945
        9:30:00.01	1947.87	1947.44	1947	1946	1945	1942.7	1942.6	1942	1941	1940.95	100	100	400	300	900	2290	100	100	100	100	1945
        9:30:00.02	1947.87	1947.44	1947	1946	1945	1943	1942.7	1942.6	1942	1941	100	100	400	700	900	900	2290	100	100	100	1943
        9:30:00.03	1947.87	1947.44	1947	1946	1945	1943	1942.7	1942.6	1942	1941	100	100	400	700	900	900	2290	100	100	100	1943
        >>> signals
        datetime	side	type	price_limit volume  stk_name       action	   seq
        2022/6/29 9:34	1	77	    0	        100     贵州茅台        open        0
        2022/6/29 9:34	-1	77	    0	        100     中信证券        open        0
        2022/6/29 9:35	-1	77	    0	        100     贵州茅台        open        1
        2022/6/29 9:35	1	77	    0	        100     贵州茅台        close       0
        2022/6/29 9:35	1	77	    0	        100     中信证券        open        1
        2022/6/29 9:35	-1	77	    0	        100     中信证券        close       0
        2022/6/29 9:36	-1	77	    0	        100     贵州茅台        close       1
        2022/6/29 9:36	1	77	    0	        100     中信证券        close       1
        """
        if len(signals) == 0: raise ValueError("has no signals")
        revenue_dict = defaultdict(dict)  # dict of dict
        ret_dict = defaultdict(dict)  # dict of dict
        aligned_signals_dict = defaultdict(dict)  # dict of dict
        signals['date']=signals['datetime'].apply(lambda x: str(x.date()))
        sig_dates = sorted(signals['date'].unique())
        sig_stk_names = sorted(signals['stk_name'].unique())

        for sig_date in sig_dates:
            for sig_stk_name in sig_stk_names:
                # 筛选出某一天的signals
                if use_dates is not None and sig_date not in use_dates: continue
                if use_stk_names is not None and sig_stk_name not in use_stk_names: continue
                print(f"broker processes {sig_date} {sig_stk_name}")

                _signals = signals.loc[np.logical_and(sig_date == signals['date'], signals['stk_name'] == sig_stk_name)]

                revenue, ret, _aligned_signals = self._meta_batch_execute(_signals, sig_date, sig_stk_name)

                revenue_dict[sig_date][sig_stk_name] = revenue
                ret_dict[sig_date][sig_stk_name] = ret
                aligned_signals_dict[sig_date][sig_stk_name] = _aligned_signals

        return revenue_dict, ret_dict, aligned_signals_dict

    ############## 旧代码 deprecated ################
    # def update_netvalue(self, date):
    #     # netvalue=self._cash+self.cash_from_short
    #     netvalue = self._cash
    #     prices = self.data.loc[date]
    #     for k, v in self.positions.items():
    #         pair_codes = k
    #         qty = v
    #         # print(prices[pair_codes],qty)
    #         price = prices[pair_codes]
    #         netvalue += price * qty
    #     print(f'netvalue {date} {netvalue}')
    #     self.net_value = pd.concat(
    #         [self.net_value,
    #          pd.DataFrame(data={'timestamp': date, 'netvalue': netvalue}, index=[0]).set_index('timestamp')])
    #
    # def calc_net_value(self, data, last_net_value, control_drawdown=False, drawdown_threshold=-0.1, level='portfolio',
    #                    weight=None):
    #     """
    #     每个调仓周期内单独计算净值以及止损
    #     param: ret, changeRatio
    #     param: last_net_value, 上一期末组合净值
    #     """
    #
    #     def weighted_net_value(data, weight):
    #         if data.isna().any().any() or weight.isna().any():
    #             print("warning !!!! calc_net_value weighted_net_value ret,weight has nan")
    #         if weight is not None:
    #             net_value = data.apply(lambda x: np.matmul(x, np.array(weight).T), axis=1)
    #         else:
    #             net_value = data.mean(axis=1)
    #
    #         return net_value
    #
    #     stoploss = None  # 记录下跌幅度
    #
    #     if len(data.columns) == 0:
    #         # 用nan填补没有持仓的日子，之后ffill
    #         print('calc_net_value: no holding codes')
    #         index = pd.to_datetime(pd.date_range(start=last_trade_date, end=trade_date, freq='D').date)
    #         return pd.DataFrame(data=[last_net_value] * len(index), index=index), stoploss
    #
    #     data = data.fillna(0) / 100 + 1
    #     data.iloc[0] = data.iloc[0] * last_net_value
    #     temp = data.cumprod()
    #
    #     if control_drawdown:
    #         if level == 'stock':
    #             test = deepcopy(temp)
    #             premaximum = test.cummax()
    #             stoploss = ((test - premaximum) / premaximum).cummin()
    #             if_stoploss = stoploss < drawdown_threshold
    #             for col in if_stoploss.columns:
    #                 test.loc[if_stoploss.loc[:, col], col] = np.nan
    #             test = test.ffill()
    #             net_value = weighted_net_value(test, weight)
    #
    #         if level == 'portfolio':
    #             net_value = weighted_net_value(temp, weight)
    #             test = deepcopy(net_value)
    #             premaximum = test.cummax()
    #             stoploss = ((test - premaximum) / premaximum).cummin()
    #             if_stoploss = stoploss < drawdown_threshold
    #             test.loc[if_stoploss] = np.nan
    #             net_value = test.ffill()
    #     else:
    #         net_value = weighted_net_value(temp, weight)
    #
    #     net_value.index = pd.to_datetime(net_value.index).date.tolist()
    #
    #     return net_value, stoploss
    #
    # def get_sign(self, type):
    #     if type == 'long':
    #         return 1
    #     elif type == 'short':
    #         return -1
    #
    # def open_position(self, timestamp, symbol, qty, type):
    #     sign = self.get_sign(type)
    #     price = self.data.loc[timestamp][symbol]
    #     if self._cash - sign * qty * price < 0:
    #         print('no more _cash')
    #         return 'fail'
    #
    #     self._cash -= sign * qty * price
    #     self.positions[symbol] += sign * qty
    #     if self.positions[symbol] == 0:
    #         del self.positions[symbol]
    #     new = {'timestamp': timestamp, 'type': type, 'pair_codes': symbol, 'qty': qty, 'price': price,
    #            'status': 'sucess'}
    #     self.transactions.append(new)
    #     print(f"open {new}")
    #     return 'sucess'
    #
    # def close_position(self, timestamp, symbol, qty, type):
    #     if symbol not in self.positions.keys():
    #         print('No position to close', symbol)
    #         return 'fail'
    #
    #     sign = self.get_sign(type)
    #     price = self.data.loc[timestamp][symbol]
    #     if type == 'long':
    #         if self.positions[symbol] < qty:
    #             print('Not enough shares to close long')
    #             return 'fail'
    #
    #     elif type == 'short':
    #         if abs(self.positions[symbol]) < qty:
    #             print('Not enough shares to close short')
    #             return 'fail'
    #
    #     self._cash += sign * qty * price
    #     self.positions[symbol] -= sign * qty
    #     if self.positions[symbol] == 0:
    #         del self.positions[symbol]
    #     new = {'timestamp': timestamp, 'type': type, 'pair_codes': symbol, 'qty': qty, 'price': price,
    #            'status': 'sucess'}
    #     self.transactions.append(new)
    #     print(f"close {new}")
    #     return 'sucess'
    #
    # def run(self):
    #     """
    #
    #     :return:
    #     """
    #     # for ret in self.ret:
    #     #     signals = self.strategies.generate_signals(ret)
    #     #     orders = self.broker.execute_signals(signals)
    #     #     self.observer.record(orders, self.ret, self.broker)
    #     # self.statistics.report(self.observer)
    #
    #     for date in self.data.index:
    #         triggers = self.signals.loc[date]
    #         prices = self.data.loc[date]
    #
    #         # close pended
    #         for symbol in self.order_pending:
    #             if symbol in self.positions.keys():
    #                 q = self.positions[symbol]
    #                 res = self.close_position(date, symbol, q, 'long')
    #                 if res == 'fail':
    #                     self.order_pending.append(symbol)
    #                 else:
    #                     self.order_pending.remove(symbol)
    #
    #         # close long
    #         close_long_list = triggers.loc[triggers['close_long'] == 1, 'pair_codes']
    #         if len(close_long_list) > 0:
    #             # cash_buy1stk=self._cash*self.usage_limit_pct/len(close_short_list)
    #             # qty=cash_buy1stk/prices.loc[close_short_list]
    #             # print(qty)
    #             for symbol in close_long_list:
    #                 if symbol in self.positions.keys():
    #                     q = self.positions[symbol]
    #                     res = self.close_position(date, symbol, q, 'long')
    #                     if res == 'fail':
    #                         self.order_pending.append(symbol)
    #
    #         # open short
    #         open_short_list = triggers.loc[triggers['open_short'] == 1, 'pair_codes']
    #         if len(open_short_list) > 0:
    #             cash_buy1stk = self._cash * self.usage_limit_pct / len(open_short_list)
    #             qty = abs(cash_buy1stk / prices.loc[open_short_list])
    #             # print(qty)
    #             for symbol in open_short_list:
    #                 q = qty[symbol]
    #                 q = min(self.max_qty, q)
    #                 self.open_position(date, symbol, q, 'short')
    #
    #         # open long
    #         open_long_list = triggers.loc[triggers['open_long'] == 1, 'pair_codes']
    #         if len(open_long_list) > 0:
    #             cash_buy1stk = self._cash * self.usage_limit_pct / len(open_long_list)
    #             qty = abs(cash_buy1stk / prices.loc[open_long_list])
    #             # print(qty)
    #             for symbol in open_long_list:
    #                 q = qty[symbol]
    #                 q = min(self.max_qty, q)
    #                 self.open_position(date, symbol, q, 'long')
    #
    #         # close short
    #         close_short_list = triggers.loc[triggers['close_short'] == 1, 'pair_codes']
    #         if len(close_short_list) > 0:
    #             # cash_buy1stk=self._cash*self.usage_limit_pct/len(close_short_list)
    #             # qty=cash_buy1stk/prices.loc[close_short_list]
    #             # print(qty)
    #             for symbol in close_short_list:
    #                 if symbol in self.positions.keys():
    #                     q = self.positions[symbol]
    #                     self.close_position(date, symbol, q, 'short')
    #
    #         # update info
    #         self.update_netvalue(date)
    #
    #         if self.verbose:
    #             print(self._cash, self.transactions, self.positions, self.net_value)

    ############## 旧代码 deprecated ################


class StockBroker(BaseBroker):
    def __init__(self,
                 data: Union[dict, defaultdict] = None,
                 cash=1e6,
                 commission=1e-3,
                 exclusive_orders=[],
                 *args, **kwargs):
        """

        Parameters
        ----------
        data :
        cash :
        commission :
        exclusive_orders :
            一律不接受相关标的的order
        args :
        kwargs :
        """

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

    def load_data(self, clean_obh_dict: Union[dict, defaultdict,PandasOHLCDataFeed] = None):
        """

        :param clean_obh_dict: dict, {date:{stk_name:ret <pd.DataFrame>}}
        :return:
        """
        self.clean_obh_dict = clean_obh_dict
        if (self.clean_obh_dict is None) or (self.clean_obh_dict == 0):
            logging.warning("self.clean_obh_dict has no data", UserWarning)

    def execute(self, signal):
        """
        执行单个指令
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

        if type == OrderTypeInt.market.value:  # 市价单，fixme 最优五档？？？
            if side == 1:  # for long order
                pass
            elif side == -1:  # for short order
                pass
            elif side == 0:
                pass
            else:
                raise NotImplementedError("side not implemented")

    def _meta_batch_execute(self, _signals: pd.DataFrame, date: str, stk_name: str):
        """
        批量处理信号，但仅针对单日单只个股信号
        Parameters
        ----------
        _signals : pd.DataFrame
            单日单只个股信号
        date
        stk_name

        Returns
        -------

        """
        # 只有side才是决定是否开仓的真正信号，此处的open close用于标识开平仓时间
        _open_signals = _signals.loc[_signals['action'] == 'open']
        _close_signals = _signals.loc[_signals['action'] == 'close']
        _aligned_signals = pd.merge(_open_signals, _close_signals, how='inner', left_on='seq', right_on='seq',
                                    suffixes=('_open', '_close'))

        _aligned_signals_long = _aligned_signals.loc[_aligned_signals['side_open'] == 1]
        _aligned_signals_short = _aligned_signals.loc[_aligned_signals['side_open'] == -1]
        _aligned_signals_hold = _aligned_signals.loc[_aligned_signals['side_open'] == 0]

        temp = self.clean_obh_dict[date][stk_name].asfreq(freq=min_freq, method='ffill')
        # b1p = temp[str(LobColTemplate('b', 1, 'p'))]
        # b1v = temp[str(LobColTemplate('b', 1, 'v'))]
        # a1p = temp[str(LobColTemplate('a', 1, 'p'))]
        # a1v = temp[str(LobColTemplate('a', 1, 'v'))]
        current = temp[str(LobColTemplate().current)]

        # 计算收益，如果为正则代表收益为正
        long_open_time = _aligned_signals_long['datetime_open'].values
        short_open_time = _aligned_signals_short['datetime_open'].values
        hold_open_time = _aligned_signals_hold['datetime_open'].values
        long_close_time = _aligned_signals_long['datetime_close'].values
        short_close_time = _aligned_signals_short['datetime_close'].values
        hold_close_time = _aligned_signals_hold['datetime_close'].values

        long_revenue = current.loc[long_close_time].values - current.loc[long_open_time].values
        long_ret = np.log(current.loc[long_close_time].values / current.loc[long_open_time].values)
        short_revenue = current.loc[short_open_time].values - current.loc[short_close_time].values
        short_ret = np.log(current.loc[short_open_time].values / current.loc[short_close_time].values)

        long_revenue = pd.Series(long_revenue, index=long_open_time)
        long_ret = pd.Series(long_ret, index=long_open_time)
        short_revenue = pd.Series(short_revenue, index=short_open_time)
        short_ret = pd.Series(short_ret, index=short_open_time)
        hold_revenue = pd.Series(np.zeros_like(hold_open_time, dtype=float), index=hold_open_time)
        hold_ret = pd.Series(np.zeros_like(hold_open_time, dtype=float), index=hold_open_time)

        revenue = pd.concat([long_revenue, short_revenue, hold_revenue], axis=0)
        ret = pd.concat([long_ret, short_ret, hold_ret], axis=0)
        _aligned_signals = pd.concat([_aligned_signals_long, _aligned_signals_short, _aligned_signals_hold], axis=0)

        return revenue, ret, _aligned_signals

    def batch_execute(self, signals: pd.DataFrame, use_dates: List[str] = None, use_stk_names: List[str] = None):
        """
        批量执行指令，无法画出净值曲线

        ----
        需要提前使用Broker.load_data()函数买卖量价信息历史（至少1档）存于Broker.clean_obh_dict <{date:{stk_name:ret <pd.DataFrame>}}>

        Parameters
        ----------
        signals : pd.DataFrame
            columes: [timestamp	side	type	price_limit	volume	stk_name	action	seq]
            stk_name+seq用于唯一识别一次交易，可以有open和close两种actions，side==1<->long，side==-1<->short，side==0<->no action（优先级高于action）
            type为订单类型，服从米筐订单类型定义，如limit order / market order，具体查看support.OrderTypeInt。todo: 更加通用化的type表示形式
        use_dates: list of str
            回测日期
        use_stk_names: list of str
            回测股票简称（中文），如`"贵州茅台"`

        Returns
        -------
        revenue_dict:dict, {stk_name: revenue}
            收益
        ret_dict:dict, {stk_name: revenue}
            收益率
        aligned_signals_dict: deprecated

        Examples
        --------
        >>> self.clean_ohb_dict['2022-06-29']['贵州茅台']
                a5_p	a4_p	a3_p	a2_p	a1_p	b1_p	b2_p	b3_p	b4_p	b5_p	a5_v	a4_v	a3_v	a2_v	a1_v	b1_v	b2_v	b3_v	b4_v	b5_v	current
        9:30:00.00	1947.87	1947.44	1947	1946	1945	1942.7	1942.6	1942	1941	1940.95	100	100	400	300	900	2290	100	100	100	100	1945
        9:30:00.01	1947.87	1947.44	1947	1946	1945	1942.7	1942.6	1942	1941	1940.95	100	100	400	300	900	2290	100	100	100	100	1945
        9:30:00.02	1947.87	1947.44	1947	1946	1945	1943	1942.7	1942.6	1942	1941	100	100	400	700	900	900	2290	100	100	100	1943
        9:30:00.03	1947.87	1947.44	1947	1946	1945	1943	1942.7	1942.6	1942	1941	100	100	400	700	900	900	2290	100	100	100	1943
        >>> signals
        datetime	side	type	price_limit volume  stk_name       action	   seq
        2022/6/29 9:34	1	77	    0	        100     贵州茅台        open        0
        2022/6/29 9:34	-1	77	    0	        100     中信证券        open        0
        2022/6/29 9:35	-1	77	    0	        100     贵州茅台        open        1
        2022/6/29 9:35	1	77	    0	        100     贵州茅台        close       0
        2022/6/29 9:35	1	77	    0	        100     中信证券        open        1
        2022/6/29 9:35	-1	77	    0	        100     中信证券        close       0
        2022/6/29 9:36	-1	77	    0	        100     贵州茅台        close       1
        2022/6/29 9:36	1	77	    0	        100     中信证券        close       1
        """
        if len(signals) == 0: raise ValueError("has no signals")
        revenue_dict = defaultdict(dict)  # dict of dict
        ret_dict = defaultdict(dict)  # dict of dict
        aligned_signals_dict = defaultdict(dict)  # dict of dict
        signals['date'] = signals['datetime'].apply(lambda x: str(x.date()))
        sig_dates = sorted(signals['date'].unique())
        sig_stk_names = sorted(signals['stk_name'].unique())

        for sig_date in sig_dates:
            for sig_stk_name in sig_stk_names:
                # 筛选出某一天的signals
                if use_dates is not None and sig_date not in use_dates: continue
                if use_stk_names is not None and sig_stk_name not in use_stk_names: continue
                print(f"broker processes {sig_date} {sig_stk_name}")

                _signals = signals.loc[np.logical_and(sig_date == signals['date'], signals['stk_name'] == sig_stk_name)]

                revenue, ret, _aligned_signals = self._meta_batch_execute(_signals, sig_date, sig_stk_name)

                revenue_dict[sig_date][sig_stk_name] = revenue
                ret_dict[sig_date][sig_stk_name] = ret
                aligned_signals_dict[sig_date][sig_stk_name] = _aligned_signals

        return revenue_dict, ret_dict, aligned_signals_dict
