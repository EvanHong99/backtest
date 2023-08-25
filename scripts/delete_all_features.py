# -*- coding=utf-8 -*-
# @File     : delete_all_features.py
# @Time     : 2023/8/24 10:53
# @Author   : EvanHong
# @Email    : 939778128@qq.com
# @Project  : 2023.06.08超高频上证50指数计算
# @Description:

from config import *
import config

import os

if __name__ == '__main__':
    for r,d,f in os.walk(root+"data/个股交易细节"):
        res=[]
        for ff in f:
            if 'feature' in ff:
                # res.append((ff.split('_')[0],ff.split('_')[1]))
                os.remove(root+"data/个股交易细节/"+ff)
        # f=sorted(list(set(res)))

    for r,d,f in os.walk(root+"data/个股交易细节"):
        res=[]
        for ff in f:
            if 'feature' in ff:
                res.append((ff.split('_')[0],ff.split('_')[1]))
        f=sorted(list(set(res)))

    print(f)