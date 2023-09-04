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
mid_path='preds/'
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