# -*- coding=utf-8 -*-
# @File     : batch_train_model_lobster.py
# @Time     : 2023/8/21 10:34
# @Author   : EvanHong
# @Email    : 939778128@qq.com
# @Project  : 2023.06.08超高频上证50指数计算
# @Description:


import os
import sys

path = os.path.join(os.path.dirname(__file__), os.pardir)
sys.path.append(path)

import logging
import pickle
from collections import defaultdict
from copy import deepcopy
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV, RidgeCV
from sklearn.utils import shuffle

from flaml import AutoML
from autogluon.tabular import TabularDataset, TabularPredictor

from config import *
import config
from support import *
from datafeeds.datafeed import LobDataFeed
from support import update_date, LobColTemplate, save_model, get_model_name
from preprocessors.preprocess import AggDataPreprocessor, LobTimePreprocessor, BaseDataPreprocessor
from statistic_tools.statistics import LobStatistics


if __name__ == '__main__':

    skip = 0
    limit = 5
    load_status()
    # 读取已处理数据文件名从而确认有多少数据可用于后续计算
    f_dict = defaultdict(list)
    for r, d, f in os.walk(detail_data_root):
        if len(f) == 0: logging.error("find no file")
        print("find files", f)
        for filename in f:
            if filename in ['placeholder']: continue
            if 'feature' not in filename: continue
            parts = filename.split('_')
            yyyy = parts[0][:4]
            mm = parts[0][4:6]
            dd = parts[0][-2:]
            stk_name = parts[1]
            if stk_name not in code_dict.keys():continue
            f_dict[stk_name].append((yyyy, mm, dd))
    # 去重并保留数据覆盖3天的股票，为了前两天scaler去scale第三天的数据
    f_dict = {k: sorted(list(set(v))) for k, v in list(f_dict.items())[skip:limit]}
    # f_dict1 = deepcopy(f_dict)
    # for k, v in f_dict1.items():
    #     if len(v) < 3:
    #         f_dict.pop(k)
    #         logging.warning(f"incompletely constructed data {k}")
    #         assert k not in config.complete_status['scalers']
    print(f"waiting list {list(f_dict.keys())}")

    # 读数据
    data_dict = defaultdict(lambda: defaultdict(list))  # data_dict={date:{stkname:[data0,data1,data2,data3]}
    tar_dict = defaultdict(dict)  # data_dict={date:{stkname:tar_data}
    datafeed = LobDataFeed()
    for stk_name, features in f_dict.items():
        for yyyy, mm, dd in features:
            try:
                for num in range(4):
                    update_date(yyyy, mm, dd)
                    feature = datafeed.load_feature(detail_data_root, config.date, stk_name, num)
                    data_dict[config.date][stk_name].append(feature.dropna(how='all'))
                    # print(f"new feature {stk_name} {num}")
            except FileNotFoundError as e:
                print("missing feature", stk_name, yyyy, mm, dd)
                continue

            # target
            tar = None
            temp = datafeed.load_clean_obh(detail_data_root, config.date, stk_name, snapshot_window=use_level,
                                           use_cols=[str(LobColTemplate('a', 1, 'p')),
                                                     str(LobColTemplate('b', 1, 'p'))])
            temp = temp.groupby(level=0).last().resample('10ms').last().sort_index().ffill().bfill()
            shift_rows = int(pred_timedelta / min_timedelta)  # 预测 pred_timedelta 之后的涨跌幅
            # todo 以一段时间的平均ret作为target
            if config.target == Target.mid_p_ret.name:
                tar = (temp[str(LobColTemplate('a', 1, 'p'))] + temp[str(LobColTemplate('b', 1, 'p'))]) / 2
                tar = np.log(tar / tar.shift(shift_rows))  # log ret
            elif config.target == Target.ret.name:
                tar = temp[LobColTemplate().current]
                tar = np.log(tar / tar.shift(shift_rows))  # log ret
            elif config.target == Target.vol.name:
                # 波动率
                ...
            tar = LobTimePreprocessor().del_untrade_time(tar, cut_tail=True)  # 不能忘
            tar_dict[config.date][stk_name] = tar.rename(config.target)
            print("load", detail_data_root, stk_name, config.date)

    # train_test_split and concat X,y in different dates
    # 法1：train test按日划分，日内数据仍旧shuffle
    train_ratio = 2 / 3
    train_len = int(len(data_dict.keys()) * train_ratio)
    dp = AggDataPreprocessor()
    # train data
    X_train_dict = defaultdict(lambda: defaultdict(pd.DataFrame))
    y_train_dict = defaultdict(lambda: defaultdict(pd.Series))
    for ymd, stk_data in list(data_dict.items())[: train_len]:
        for stk_name, features in stk_data.items():
            for num, feature in enumerate(features):
                X, y = dp.align_Xy(feature, tar_dict[ymd][stk_name], pred_timedelta=pred_timedelta)  # 最重要的是对齐X y
                X_train_dict[stk_name][num] = pd.concat([X_train_dict[stk_name][num], X], axis=0)
                y_train_dict[stk_name][num] = pd.concat([y_train_dict[stk_name][num].rename(config.target), y], axis=0)
    # test data
    # note: do not delete
    #  todo:该阶段（train scaler and model）不需要test的数据，但在之后完整的backtest时仍旧需要用到该段代码
    X_test_dict = defaultdict(lambda: defaultdict(pd.DataFrame))
    y_test_dict = defaultdict(lambda: defaultdict(pd.Series))
    for ymd, stk_data in list(data_dict.items())[train_len:]:
        for stk_name, features in stk_data.items():
            for num, feature in enumerate(features):
                X, y = dp.align_Xy(feature, tar_dict[ymd][stk_name], pred_timedelta=pred_timedelta)  # 最重要的是对齐X y
                X_test_dict[stk_name][num] = pd.concat([X_test_dict[stk_name][num], X], axis=0)
                y_test_dict[stk_name][num] = pd.concat([y_test_dict[stk_name][num].rename(config.target), y], axis=0)

    # 法2：shuffle后按比例划分
    ...

    # scale X data and save scaler
    scaler_dict = defaultdict(dict)
    for stk_name in X_train_dict.keys():
        for num in range(4):
            dp = AggDataPreprocessor()
            X_train_dict[stk_name][num] = dp.std_scale(X_train_dict[stk_name][num], refit=True)[0]
            dp.save_scaler(scaler_root, FILE_FMT_scaler.format(stk_name, num, '_'))
        config.complete_status['scalers'].append(stk_name)
        save_status()
    for stk_name in X_test_dict.keys():
        for num in range(4):
            dp = AggDataPreprocessor()
            X_test_dict[stk_name][num] = dp.std_scale(X_test_dict[stk_name][num], refit=True)[0]
            dp.save_scaler(scaler_root, FILE_FMT_scaler.format(stk_name, num, '_'))
        config.complete_status['scalers'].append(stk_name)
        save_status()

    # gather different stk data todo: 不同股票不同模型
    all_X_train = defaultdict(pd.DataFrame)
    all_y_train = defaultdict(pd.Series)
    all_X_test = defaultdict(pd.DataFrame)
    all_y_test = defaultdict(pd.Series)
    for stk_name in X_train_dict.keys():
        for num in range(4):
            all_X_train[num] = pd.concat([all_X_train[num], X_train_dict[stk_name][num]], axis=0)
            all_y_train[num] = pd.concat([all_y_train[num].rename(config.target), y_train_dict[stk_name][num]], axis=0)
    for stk_name in X_test_dict.keys():
        for num in range(4):
            all_X_test[num] = pd.concat([all_X_test[num], X_test_dict[stk_name][num]], axis=0)
            all_y_test[num] = pd.concat([all_y_test[num].rename(config.target), y_test_dict[stk_name][num]], axis=0)

    # train & save model
    USE_LINEAER = False
    USE_FLAML = False
    USE_AUTOGLUON = True
    stat = pd.DataFrame()
    all_y_pred = pd.Series()
    models = {}
    lob_stat = LobStatistics()
    for num in range(4):
        X_train, y_train = shuffle(all_X_train[num], all_y_train[num])
        X_test, y_test = shuffle(all_X_test[num], all_y_test[num])

        # # linear
        # model = RidgeCV(cv=5, gcv_mode='auto')
        # # model = LassoCV(cv=5, n_jobs=1, random_state=0,tol=1e-5,precompute=True,selection='random',max_iter=3000)
        # model.fit(X_train, y_train)
        # print(f"linear coef: {model.coef_}")

        # flaml
        # automl_settings = {
        #     "time_budget": 120,  # in seconds
        #     "metric": 'rmse',
        #     "task": 'regression',
        #     "log_file_name": model_root + f"flaml_training{num}.log",
        #     "verbose": 2,  # int, default=3 | Controls the verbosity, higher means more messages.
        # }
        # model = AutoML()
        # model.fit(X_train, y_train, **automl_settings)
        # print(f'best model for period {num}', model.model.estimator)  # Print the best model

        # autogluon
        if USE_AUTOGLUON:
            _y_train = deepcopy(y_train)
            _y_test = deepcopy(y_test)
            _y_train.index = X_train.index
            _y_test.index = X_test.index
            _train_data = pd.concat([X_train, _y_train], axis=1)
            _test_data = pd.concat([X_test, _y_test], axis=1)
            # _train_data = TabularDataset(_train_data)
            # _test_data = TabularDataset(_test_data)
            model = TabularPredictor(label=str(config.target),
                                     problem_type='regression',
                                     eval_metric='rmse',
                                     log_to_file=True,
                                     log_file_path=model_root + f"autogluon_training{num}.log",
                                     verbosity=2,
                                     path=model_root)

            model.fit(_train_data, time_limit=60, fit_weighted_ensemble=True)

        # predict
        if USE_LINEAER or USE_FLAML:
            y_pred1 = model.predict(X_train)
            stat = pd.concat([stat, lob_stat.stat_pred_error(y_train, y_pred1, name=f'train_stat_{num}')], axis=1)
            y_pred = model.predict(X_test)
            stat = pd.concat([stat, lob_stat.stat_pred_error(y_test, y_pred, name=f'test_stat_{num}')], axis=1)
            all_y_pred = pd.concat([all_y_pred, pd.Series(y_pred)], axis=0, ignore_index=True)
        if USE_AUTOGLUON:
            y_pred1 = model.predict(_train_data)
            stat = pd.concat([stat, lob_stat.stat_pred_error(y_train, y_pred1, name=f'autogluon_train_stat_{num}')], axis=1)
            y_pred = model.predict(_test_data)
            stat = pd.concat([stat, lob_stat.stat_pred_error(y_test, y_pred, name=f'autogluon_test_stat_{num}')], axis=1)
            all_y_pred = pd.concat([all_y_pred, pd.Series(y_pred)], axis=0, ignore_index=True)
            eval = model.evaluate(_test_data, silent=True)
            lboard = model.leaderboard(_test_data, silent=True)
            results = model.fit_summary(show_plot=False)
            print(eval)
            print(lboard)
            print(results)

        # save
        save_model(model_root, FILE_FMT_model.format('lobster_general', num, get_model_name(model)), model)

    print(stat)
    print(all_y_pred)
    stat.to_csv(res_root + f"lobster_stat_{get_model_name(model)}.csv")
    all_y_pred.to_csv(res_root + f"lobster_all_y_pred_{get_model_name(model)}.csv")
