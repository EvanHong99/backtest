# -*- coding=utf-8 -*-
# @Project  : caitong_securities
# @FilePath : 2023.10.23股指日内短期预测/backtest/recorders
# @File     : position.py
# @Time     : 2023/11/27 00:02
# @Author   : EvanHong
# @Email    : 939778128@qq.com
# @Description:


class Position(object):
    """
    only for assets like stocks or options, but not for cash
    """

    # def __init__(self, broker: 'Broker'):
    #     self.__broker = broker

    def __init__(self, kind: str = 'stock'):
        self.kind = kind
        self.__broker=None

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
