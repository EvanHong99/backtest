# -*- coding=utf-8 -*-
# @File     : plot_distribution.py
# @Time     : 2023/8/28 14:05
# @Author   : EvanHong
# @Email    : 939778128@qq.com
# @Project  : 2023.06.08超高频上证50指数计算
# @Description:
from copy import deepcopy

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from config import res_root
import seaborn as sns
from scipy.stats import norm

stk_name='贵州茅台'
mid_path='波动率预测/preds/'
y_true=pd.read_csv(res_root+mid_path+f'all_y_true_{stk_name}_vol.csv',index_col=0,header=0)
y_true.columns=['y_true']
idx=(y_true==0).values.flatten()
y_true=y_true[~idx]
y_true_origin=deepcopy(y_true)

t=sns.displot(y_true,label="Skewness: %.2f"%(y_true.skew()) , kind="hist",log_scale=False) # kind=[kde,ecdf]
ax=t.figure.axes
xaixs=np.arange(0,y_true.max().values[0],1e-5)
mu, std = norm.fit(y_true)
ax[0].plot(xaixs,norm.pdf(xaixs,mu,std),color='red',label='normal')
ax[0].legend(loc='best')
t.legend.remove()
t.savefig(res_root+mid_path+'y_distribution.png')

y_true=np.log(y_true_origin)
t=sns.displot(y_true,label="Skewness: %.2f"%(y_true.skew()) , kind="hist",log_scale=False) # kind=[kde,ecdf]
ax=t.figure.axes
xaixs=np.arange(0,y_true.max().values[0],1e-5)
mu, std = norm.fit(y_true)
ax[0].plot(xaixs,norm.pdf(xaixs,mu,std),color='red',label='normal')
ax[0].legend(loc='best')
t.legend.remove()
t.savefig(res_root+mid_path+'y_distribution_log.png')

y_true=np.sqrt(y_true_origin)
t=sns.displot(y_true,label="Skewness: %.2f"%(y_true.skew()) , kind="hist",log_scale=False) # kind=[kde,ecdf]
ax=t.figure.axes
xaixs=np.arange(0,y_true.max().values[0],1e-5)
mu, std = norm.fit(y_true)
ax[0].plot(xaixs,norm.pdf(xaixs,mu,std),color='red',label='normal')
ax[0].legend(loc='best')
t.legend.remove()
t.savefig(res_root+mid_path+'y_distribution_sqrt.png')

y_true=np.float_power(y_true,1/3)
t=sns.displot(y_true,label="Skewness: %.2f"%(y_true.skew()) , kind="hist",log_scale=False) # kind=[kde,ecdf]
ax=t.figure.axes
xaixs=np.arange(0,y_true.max().values[0],1e-5)
mu, std = norm.fit(y_true)
ax[0].plot(xaixs,norm.pdf(xaixs,mu,std),color='red',label='normal')
ax[0].legend(loc='best')
t.legend.remove()
t.savefig(res_root+mid_path+'y_distribution_cube_root.png')


def tar_weight(_y_train, bins=20):
    nrows = len(_y_train)
    hist, bin_edge = np.histogram(_y_train, bins=bins)
    hist[hist == 0] += 1  # 避免出现RuntimeWarning: divide by zero encountered in divide  bin_w = np.sqrt(nrows / hist)
    bin_idx = np.digitize(_y_train, bin_edge,
                          right=False) - 1  # series中元素是第几个bin，可能会导致溢出，因为有些值（比如0）刚好在bin边界上，所以要把这些归到第一个bin（即现在的bin_idx==1处）
    bin_idx[bin_idx == 0] += 1
    bin_idx -= 1
    bin_w = np.sqrt(nrows / hist)
    series_w = np.array([bin_w[i] for i in bin_idx])
    series_w = nrows / sum(series_w) * series_w
    series_w = pd.Series(series_w, index=_y_train.index, name='weight')
    return series_w