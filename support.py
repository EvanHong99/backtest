# -*- coding=utf-8 -*-
# @File     : support.py
# @Time     : 2023/8/2 19:05
# @Author   : EvanHong
# @Email    : 939778128@qq.com
# @Project  : 2023.06.08超高频上证50指数计算
# @Description:

from enum import Enum

class OrderSideInt(Enum):
    bid = 66  # 买
    ask = 83  # 卖


class OrderTypeInt(Enum):
    limit = 76  # 限价单
    market = 77  # 市价单
    best_own = 85  # 本⽅最优
    bop = 85  # 本⽅最优

class Target(Enum):
    ret = 0  # ret
    mid_p_ret = 1  # 预测中间价的ret

class LobColTemplate(object):
    """
    column template
    """

    def __init__(self, side: str=None, level: int=None, target: str=None):
        self.side = side
        self.level = level
        self.target = target
        self.current='current'

    def __str__(self):
        return f"{self.side}{self.level}_{self.target}"
