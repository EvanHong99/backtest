# -*- coding=utf-8 -*-
# @File     : __init__.py.py
# @Time     : 2023/7/28 14:54
# @Author   : EvanHong
# @Email    : 939778128@qq.com
# @Project  : 2023.06.08超高频上证50指数计算
# @Description:
from support import *
import json
import config
from support import __check_obedience

if __name__ == '__main__':
    # with open(config.root+'backtest/complete_status.json', 'r', encoding='gb2312') as fr:
    #     config.complete_status = json.load(fr)
    # print("load_status",config.complete_status)
    # __check_obedience(config.complete_status)
    load_status()
    save_status()
