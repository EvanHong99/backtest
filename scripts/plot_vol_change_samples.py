# -*- coding=utf-8 -*-
# @File     : plot_vol_change_samples.py
# @Time     : 2023/9/15 11:08
# @Author   : EvanHong
# @Email    : 939778128@qq.com
# @Project  : 2023.06.08超高频上证50指数计算
# @Description:
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import config
from support import load_concat_data
from datetime import timedelta
from preprocessors.preprocess import AggDataPreprocessor

price_df=pd.read_csv(config.data_root+'tick_贵州茅台_2023-04-01_2023-08-28.csv',header=0,index_col=0)
price_df.index=pd.to_datetime(price_df.index)
wap1=(price_df['a1_p']*price_df['b1_v']+price_df['b1_p']*price_df['a1_v'])/(price_df['a1_v']+price_df['b1_v'])
current=price_df['current']
y_pred=pd.read_csv("D:/Work/INTERNSHIP/海通场内/2023.06.08超高频上证50指数计算/res/波动率方向预测/preds/vol_chg_medium_quality_1.0_600.0_600.0/all_y_pred_贵州茅台_TabularPredictor_vol_chg.csv",header=0,index_col=0)
y_pred.index=pd.to_datetime(y_pred.index)

y_train_dict1 = load_concat_data('y_train_dict', mid_path='concat_tick_data/')
y_test_dict1 = load_concat_data('y_test_dict', mid_path='concat_tick_data/')
df1=pd.Series()
for i in range(4):
    df1=pd.concat([df1,y_train_dict1['贵州茅台'][i]],axis=0)

mid_path = f'concat_tick_data_{config.target}_10min_10min/'
X_train_dict = load_concat_data('X_train_dict', mid_path=mid_path)
X_test_dict = load_concat_data('X_test_dict', mid_path=mid_path)
y_train_dict = load_concat_data('y_train_dict', mid_path=mid_path)
y_test_dict = load_concat_data('y_test_dict', mid_path=mid_path)


df=pd.Series()
for i in range(4):
    df=pd.concat([df,y_train_dict['贵州茅台'][i]],axis=0)

# 画正样本日内频数分数图
df.loc[df==1].reset_index()['index'].apply(lambda x: str(x)[-8:]).hist()
plt.savefig(config.res_root+'hist_of_positive.png')
plt.figure()

# 画波动变化的示例
wap1.index=pd.to_datetime(wap1.index)
tar_series=wap1
index = y_pred.loc[(y_pred == 1).values].index
for idx in range(100):
    idx_time=index[idx]
    frames=[idx_time-timedelta(seconds=1500),idx_time-timedelta(seconds=1200),idx_time-timedelta(seconds=600),idx_time,idx_time+timedelta(seconds=300)]
    temp=tar_series.loc[frames[0]:frames[4]]
    X_std=AggDataPreprocessor.calc_realized_volatility(np.log(tar_series.loc[frames[1]:frames[2]]).diff().dropna()).values[0]
    y_std=AggDataPreprocessor.calc_realized_volatility(np.log(tar_series.loc[frames[2]:frames[3]]).diff().dropna()).values[0]
    if y_std>X_std*1.03:
        temp.plot()
        plt.title(f'X_std{np.round(X_std,decimals=7)}_y_std{np.round(y_std,decimals=7)}')
        plt.axvline(x=frames[1], color='r')
        plt.axvline(x=frames[2], color='r', linestyle='--')
        plt.axvline(x=frames[3], color='r')
        plt.show()

