# -*- coding=utf-8 -*-
# @File     : plot_pred_vol.py
# @Time     : 2023/8/24 17:06
# @Author   : EvanHong
# @Email    : 939778128@qq.com
# @Project  : 2023.06.08超高频上证50指数计算
# @Description:
import pandas as pd
import matplotlib.pyplot as plt

y_true=pd.read_csv('./res/all_y_true_vol.csv',index_col=0,header=0)
y_true.columns=['y_true']
y_pred=pd.read_csv('./res/all_y_pred_TabularPredictor_vol.csv',index_col=0,header=0)
y_pred.columns=['y_pred']

length=int(len(y_true)/4)
i=0
pd.concat([y_true,y_pred],axis=1).iloc[length*(i):length*(i+1)].plot()
plt.show()