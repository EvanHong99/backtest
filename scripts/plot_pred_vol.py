# -*- coding=utf-8 -*-
# @File     : plot_pred_vol.py
# @Time     : 2023/8/24 17:06
# @Author   : EvanHong
# @Email    : 939778128@qq.com
# @Project  : 2023.06.08超高频上证50指数计算
# @Description:
import pandas as pd
import matplotlib.pyplot as plt
from config import res_root

stk_name = '贵州茅台'
mid_path = '波动率预测/preds/skew_holdout0.2/'
y_true = pd.read_csv(res_root + mid_path + f'all_y_test_{stk_name}_vol.csv', index_col=0, header=0)
y_true.columns = ['y_true']
y_pred = pd.read_csv(res_root + mid_path + f'all_y_pred_{stk_name}_TabularPredictor_vol.csv', index_col=0, header=0)
y_pred.columns = ['y_pred']

length = int(len(y_true) / 4)
period = 0
pd.concat([y_true, y_pred], axis=1).plot()
# pd.concat([y_true, y_pred], axis=1).iloc[length * period:length * (period + 1)].plot()
# plt.show()
plt.savefig(res_root + mid_path + f'pred_{stk_name}.png', dpi=640, bbox_inches='tight')
