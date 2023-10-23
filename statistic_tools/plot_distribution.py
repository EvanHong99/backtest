# -*- coding=utf-8 -*-
# @File     : plot_distribution.py
# @Time     : 2023/8/24 17:06
# @Author   : EvanHong
# @Email    : 939778128@qq.com
# @Project  : 2023.06.08超高频上证50指数计算
# @Description:

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm

def plot_distribution(series:pd.Series,save_path=None):
    t=sns.displot(series, label="Skewness: %.2f" % (series.skew()), kind="hist", log_scale=False) # kind=[kde,ecdf]
    ax=t.figure.axes
    xaixs=np.arange(series.min(), series.max(), 1e-5)
    mu, std = norm.fit(series)
    ax[0].plot(xaixs,norm.pdf(xaixs,mu,std),color='red',label='normal')
    ax[0].legend(loc='best')
    # t.legend.remove()
    if save_path is not None:
        t.savefig(save_path)
    plt.show()
    
    
def to_bins(series:pd.Series,bins):
    """
    常用于将收益率序列划分到不同的label中

    Parameters
    ----------
    series :
    bins :

    Returns
    -------

    """
    # todo 将bins作为list，然后用n个标准差来划分bins
    assert bins % 2 == 1
    hist, bin_edge = np.histogram(series, bins=bins)
    hist[hist == 0] += 1  # 避免出现RuntimeWarning: divide by zero encountered in divide  bin_w = np.sqrt(nrows / hist)
    bin_idx = np.digitize(series, bin_edge,
                          right=False) - 1  # series中元素是第几个bin，可能会导致溢出，因为有些值（比如0）刚好在bin边界上，所以要把这些归到第一个bin（即现在的bin_idx==1处）
    bin_idx[bin_idx == 0] += 1
    bin_idx -= 1
    bin_value = list(range(-(bins // 2), bins // 2 + 1, 1))
    series_value = np.array([bin_value[i] for i in bin_idx])
    series_value = pd.Series(series_value, index=series.index, name=series.name)
    return series_value