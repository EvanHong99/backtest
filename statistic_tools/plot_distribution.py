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
    
