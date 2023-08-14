# -*- coding=utf-8 -*-
# @File     : split_data.py
# @Time     : 2023/8/14 22:11
# @Author   : EvanHong
# @Email    : 939778128@qq.com
# @Project  : 2023.06.08超高频上证50指数计算
# @Description: 将不同股票数据都整合到一起并split，从而为模型训练提供足量的数据

from sklearn.model_selection import train_test_split
from config import *
import config
from support import *



