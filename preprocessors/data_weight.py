# -*- coding=utf-8 -*-
# @File     : observer.py
# @Time     : 2023/8/2 12:23
# @Author   : EvanHong
# @Email    : 939778128@qq.com
# @Project  : 2023.06.08超高频上证50指数计算
# @Description:
import numpy as np
import pandas as pd

def tar_weight(_y_train, bins=20):
    """
    用于计算不平衡样本的权重，仅限于连续型y的权重
    Parameters
    ----------
    _y_train
    bins

    Returns
    -------

    """
    nrows = len(_y_train)
    hist, bin_edge = np.histogram(_y_train, bins=bins)
    hist[
        hist == 0] += 1  # 避免出现RuntimeWarning: divide by zero encountered in divide  bin_w = np.sqrt(nrows / hist)
    bin_idx = np.digitize(_y_train, bin_edge,
                          right=False) - 1  # series中元素是第几个bin，可能会导致溢出，因为有些值（比如0）刚好在bin边界上，所以要把这些归到第一个bin（即现在的bin_idx==1处）
    bin_idx[bin_idx == 0] += 1
    bin_idx -= 1
    bin_w = np.sqrt(nrows / hist)
    series_w = np.array([bin_w[i] for i in bin_idx])
    series_w = nrows / sum(series_w) * series_w
    series_w = pd.Series(series_w, index=_y_train.index, name='weight')
    return series_w
