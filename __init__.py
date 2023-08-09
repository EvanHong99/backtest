# -*- coding=utf-8 -*-
# @File     : __init__.py.py
# @Time     : 2023/8/2 11:11
# @Author   : EvanHong
# @Email    : 939778128@qq.com
# @Project  : 2023.06.08超高频上证50指数计算
# @Description:


class BaseObject(object):
    def __init__(self,*args,**kwargs):
        self.args=args
        for k,v in kwargs.items():
            self.__setattr__(k,v)