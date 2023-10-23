# -*- coding=utf-8 -*-
# @File     : batch_train_model.py
# @Time     : 2023/8/11 16:33
# @Author   : EvanHong
# @Email    : 939778128@qq.com
# @Project  : 2023.06.08超高频上证50指数计算
# @Description: 训练模型，并保存相关的scaler和model
import os
import sys

import numpy as np
import pandas as pd

import config
from tqdm import tqdm

path = os.path.join(os.path.dirname(__file__), os.pardir)
sys.path.append(path)

from collections import defaultdict
from copy import deepcopy
from typing import Union
from scipy.special import xlogy

from sklearn.linear_model import LassoCV
from sklearn.utils import shuffle
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score, fbeta_score
from sklearn.preprocessing import Binarizer

from autogluon.core.metrics import make_scorer
from autogluon.tabular import TabularPredictor
from autogluon.common import space
from autogluon.tabular.configs.hyperparameter_configs import get_hyperparameter_config
from autogluon.multimodal import MultiModalPredictor

from config import *
from support import *
from datafeeds.datafeed import LobDataFeed
from support import update_date, LobColTemplate, save_model, get_model_name
from preprocessors.preprocess import AggDataPreprocessor, LobTimePreprocessor
from statistic_tools.statistics import LobStatistics
from preprocessors.data_weight import tar_weight
from metrics.focal_loss import focal_loss,fbeta_focal_score

metric_dict = {}
metric_dict['fbeta_score'] = make_scorer(name='fbeta_score',
                                         score_func=fbeta_score,
                                         optimum=1,
                                         greater_is_better=True,
                                         needs_proba=False,
                                         needs_threshold=False,
                                         needs_quantile=False,
                                         metric_kwargs={'beta': 0.5, # beta = 2 makes recall twice as important as precision
                                                        'average': 'binary',  #
                                                        'zero_division': 0}
                                         )
metric_dict['focal_loss'] = make_scorer(name='focal_loss',
                                         score_func=focal_loss,
                                         optimum=0,
                                         greater_is_better=False,
                                         needs_proba=True,
                                         needs_threshold=False,
                                         needs_quantile=False,
                                         metric_kwargs={'gamma': 2,
                                                        'alpha': None,
                                                        'size_average': True}
                                         )
metric_dict['focal_loss_noweight'] = make_scorer(name='focal_loss',
                                         score_func=focal_loss,
                                         optimum=0,
                                         greater_is_better=True,
                                         needs_proba=True,
                                         needs_threshold=False,
                                         needs_quantile=False,
                                         metric_kwargs={'gamma': 5,
                                                        'alpha': 0.8,
                                                        'size_average': True}
                                         )
metric_dict['fbeta_focal_score'] = make_scorer(name='fbeta_focal_score',
                                              score_func=fbeta_focal_score,
                                              optimum=1,
                                              greater_is_better=True,
                                              needs_proba=True,
                                              needs_threshold=False,
                                              needs_quantile=False,
                                              metric_kwargs={'gamma': 5,
                                                        'alpha': 0.8,
                                                        'size_average': True,
                                                        'beta':0.5,
                                                        'zeta':1}
                                              )

if __name__ == '__main__':
    load_status(is_tick=True)
    mid_path = f'concat_tick_data_{config.target}_10min_10min/'
    all_data_dict = load_concat_data('all_data_dict', mid_path=mid_path)

    '''
    skip = 0
    limit = 1
    # 读取已处理数据文件名从而确认有多少数据可用于后续计算
    f_dict = defaultdict(list)
    counter = defaultdict(lambda: defaultdict(int))
    for r, d, f in os.walk(data_root + 'tick_data/'):
        if len(f) == 0: logging.error("find no file")
        for filename in f:
            if filename in ['placeholder']: continue
            parts = filename.split('_')
            yyyy = parts[0][:4]
            mm = parts[0][4:6]
            dd = parts[0][-2:]
            stk_name = parts[1]
            counter[stk_name][(yyyy, mm, dd)] += 1

    f_dict = {k: sorted(list(v.keys())) for k, v in counter.items()}
    # 按权重排序
    _stknames = set(list(f_dict.keys()))
    w_sorted_stknames = np.array([True if k in _stknames else False for k in code_dict.keys()])
    f_dict = {k: f_dict[k] for k in np.array(list(code_dict.keys()))[w_sorted_stknames]}
    f_dict = {k: v for k, v in list(f_dict.items())[skip:limit]}
    f_dict1 = deepcopy(f_dict)
    info={k:len(v) for k,v in f_dict.items()}
    print(f"waiting list {info}")

    # 读数据
    tp = LobTimePreprocessor()
    feature_dict = defaultdict(lambda: defaultdict(list))  # feature_dict={date:{stkname:[data0,data1,data2,data3]}
    tar_dict = defaultdict(list)  # feature_dict={date:{stkname:tar_data}
    datafeed = LobDataFeed()
    loaded_files = defaultdict(list)
    for stk_name, ymd_list in list(f_dict.items()):
        for i in tqdm(range(len(ymd_list))):
            yyyy, mm, dd = ymd_list[i]
            try:
                update_date(yyyy, mm, dd)
                for num in range(4):
                    # feature = datafeed.load_feature(data_root + 'tick_data/', config.date, stk_name, num)
                    # feature=tp.del_untrade_time(feature,cut_tail=True,strip=strip_time,drop_last_row=True) # fixme feature数据最后一行会有0.01s的精度，需要drop_last_row=True，但最终应修改feature生成算法
                    feature_dict[config.date][stk_name] = [df.dropna(how='all') for df in
                                                           all_data_dict[config.date][stk_name]]
            except FileNotFoundError as e:
                logging.warning(f"missing feature {(stk_name, yyyy, mm, dd)}")
                if f_dict1.get(stk_name) is not None:
                    f_dict1.pop(stk_name)
                if feature_dict[config.date].get(stk_name) is not None:
                    feature_dict[config.date].pop(stk_name)
                continue

            # target
            tar = None
            temp = datafeed.load_clean_obh(data_root + 'tick_data/', config.date, stk_name, snapshot_window=use_level,
                                           use_cols=[str(LobColTemplate('a', 1, 'p')),
                                                     str(LobColTemplate('b', 1, 'p')),
                                                     str(LobColTemplate('a', 1, 'v')),
                                                     str(LobColTemplate('b', 1, 'v')),
                                                     LobColTemplate().current
                                                     ])
            temp = temp.groupby(level=0).last().resample(min_freq).last().sort_index().ffill()
            # wap1 = (temp[str(LobColTemplate('a', 1, 'p'))] * temp[str(LobColTemplate('b', 1, 'v'))] +
            #         temp[str(LobColTemplate('b', 1, 'p'))] * temp[str(LobColTemplate('a', 1, 'v'))]) / (temp[str(LobColTemplate('b', 1, 'v'))]+temp[str(LobColTemplate('a', 1, 'v'))])
            mid_p=(temp[str(LobColTemplate('a', 1, 'p'))]+temp[str(LobColTemplate('b', 1, 'p'))])/2
            shift_rows = int(pred_timedelta / min_timedelta)  # 预测 pred_timedelta 之后的涨跌幅
            if config.target == Target.mid_p_ret.name:
                tar = (temp[str(LobColTemplate('a', 1, 'p'))] + temp[str(LobColTemplate('b', 1, 'p'))]) / 2
                tar = np.log(tar / tar.shift(shift_rows))  # log ret
            elif config.target == Target.ret.name:
                tar = temp[LobColTemplate().current]
                tar = np.log(tar / tar.shift(shift_rows))  # log ret
            elif config.target == Target.vol.name:
                # 波动率
                # 计算(t+1,t+pred_delta]
                rolling_rows = shift_rows
                shrink=pred_timedelta/timedelta(minutes=1) # 用于将数据缩放到当前最优步长，如1min，因为当前是分钟级的预测，并且会去掉开收盘5min，会导致非10的整数分钟出现，因此较为理想的方式是缩放到步长1min
                step_rows = int(rolling_rows / shrink) # 每两个数据点之间相差1min，但rolling的仍旧是10min的数据，是为了方便不受开收盘strip的时间的影响
                tar = deepcopy(mid_p)
                try:
                    tar_shift = tar / tar.shift(1)
                    tar_shift = pd.Series(np.where(tar_shift > 0, tar_shift, 0), index=tar.index)
                    tar = np.log(tar_shift, where=tar_shift > 0)  # log ret
                except RuntimeWarning:
                    pass
                tar1 = deepcopy(tar)
                tar = tar.reset_index()[[0]].rolling(rolling_rows, min_periods=rolling_rows, center=False, step=step_rows).apply(
                    AggDataPreprocessor.calc_realized_volatility)
                # 用sqrt对vol的偏度进行修正
                tar = pd.Series(np.sqrt(tar.values.flatten()), index=tar1.iloc[tar.index].index, name=config.target)
            elif config.target == Target.vol_chg.name:
                rolling_rows = shift_rows
                shrink=pred_timedelta/timedelta(minutes=1) # 用于将数据缩放到当前最优步长，如1min，因为当前是分钟级的预测，并且会去掉开收盘5min，会导致非10的整数分钟出现，因此较为理想的方式是缩放到步长1min
                step_rows = int(rolling_rows / shrink) # 每两个数据点之间相差1min，但rolling的仍旧是10min的数据，是为了方便不受开收盘strip的时间的影响
                # tar = deepcopy(wap1)
                tar = deepcopy(mid_p)
                try:
                    # 先计算收益率 log ret
                    tar_shift = tar / tar.shift(1)
                    tar_shift = pd.Series(np.where(tar_shift > 0, tar_shift, 0), index=tar.index)
                    tar = np.log(tar_shift, where=tar_shift > 0)
                except RuntimeWarning:
                    pass
                tar1 = deepcopy(tar)
                tar = tar.reset_index()[[0]].rolling(rolling_rows, min_periods=rolling_rows, center=False, step=step_rows).apply(
                    AggDataPreprocessor.calc_realized_volatility)
                tar = pd.Series(tar.values.flatten(), index=tar1.iloc[tar.index].index)
                # 虽然step是1min但是用到了10min的数据去agg。这里虽然没有去掉下午开盘十分钟，但是因为之后在对齐的时候会删掉因此这边可以简单化处理直接diff
                tar=tar.diff(int(agg_timedelta/timedelta(seconds=int(agg_timedelta.seconds/shrink)))).dropna()
            # tar里面包含了中午的一个半小时
            tar = LobTimePreprocessor().del_untrade_time(tar, cut_tail=True, strip=strip_time,split_df=False)  # 删除午休后，下午的前10个数据（13:00:00~13:09:00）是有问题的，因为并未用满10min，这会在之后的处理中删除掉
            tar_dict[stk_name].append(tar.rename(config.target))
            loaded_files[stk_name].append(config.date)
    print(loaded_files)
    f_dict = f_dict1

    # 将tar_dict所有按日期划分的tar concat到一起并寻找合适的标签阈值
    threshold_dict={}
    for stk_name in tar_dict.keys():
        tar_dict[stk_name]=pd.concat(tar_dict[stk_name],axis=0).sort_index()
        threshold_dict[stk_name]=tar_dict[stk_name].quantile(0.8)
        tar_dict[stk_name]=tar_dict[stk_name].apply(lambda x: 1 if x>threshold_dict[stk_name] else 0)


    # train_test_split and concat X,y in different dates
    # 法1：train test按日划分，日内数据仍旧shuffle
    train_ratio = 4 / 5
    train_len = int(len(feature_dict.keys()) * train_ratio)
    dp = AggDataPreprocessor()
    # train data
    X_train_dict = defaultdict(lambda: defaultdict(list))
    y_train_dict = defaultdict(lambda: defaultdict(list))
    for ymd in list(feature_dict.keys())[: train_len]:
        for stk_name, features in feature_dict[ymd].items():
            for num, feature in enumerate(features):
                # num = str(num)
                # X, y按照iloc已经一一对应
                X, y = dp.align_Xy(feature, tar_dict[stk_name], pred_timedelta=pred_timedelta)  # 最重要的是对齐X y
                # if config.target==Target.vol_chg.name:
                #     base=tar_dict[ymd][stk_name].loc[y.index[0]-pred_timedelta]
                #     first=1 if y.iloc[0]-base>0 else 0
                #     y=y.diff().apply(lambda x: 1 if x>0 else 0)
                #     y.iloc[0]=first
                X_train_dict[stk_name][num].append(X)
                y_train_dict[stk_name][num].append(y.rename(config.target))
    for stk_name in y_train_dict.keys():
        for num in y_train_dict[stk_name].keys():
            # num = str(num)
            X_train_dict[stk_name][num] = pd.concat(X_train_dict[stk_name][num], axis=0)
            y_train_dict[stk_name][num] = pd.concat(y_train_dict[stk_name][num], axis=0)
    # test data
    X_test_dict = defaultdict(lambda: defaultdict(list))
    y_test_dict = defaultdict(lambda: defaultdict(list))
    for ymd in list(feature_dict.keys())[train_len:]:
        for stk_name, features in feature_dict[ymd].items():
            for num, feature in enumerate(features):
                # num = str(num)
                # X, y按照iloc已经一一对应
                X, y = dp.align_Xy(feature, tar_dict[stk_name], pred_timedelta=pred_timedelta)  # 最重要的是对齐X y
                # if config.target==Target.vol_chg.name:
                #     base=tar_dict[ymd][stk_name].loc[y.index[0]-pred_timedelta]
                #     first=1 if y.iloc[0]-base>0 else 0
                #     y=y.diff().apply(lambda x: 1 if x>0 else 0)
                #     y.iloc[0]=first
                X_test_dict[stk_name][num].append(X)
                y_test_dict[stk_name][num].append(y.rename(config.target))
    for stk_name in y_test_dict.keys():
        for num in y_test_dict[stk_name].keys():
            # num = str(num)
            X_test_dict[stk_name][num] = pd.concat(X_test_dict[stk_name][num], axis=0)
            y_test_dict[stk_name][num] = pd.concat(y_test_dict[stk_name][num], axis=0)
    # save
    mid_path = f'concat_tick_data_{config.target}_10min_10min/'
    save_concat_data(dict(X_train_dict), 'X_train_dict', mid_path=mid_path)
    save_concat_data(dict(X_test_dict), 'X_test_dict', mid_path=mid_path)
    save_concat_data(dict(y_train_dict), 'y_train_dict', mid_path=mid_path)
    save_concat_data(dict(y_test_dict), 'y_test_dict', mid_path=mid_path)
    save_concat_data(dict(tar_dict), 'tar_dict', mid_path=mid_path)
    save_concat_data(dict(threshold_dict), 'threshold_dict', mid_path=mid_path)


    '''

    # 读数据
    mid_path = f'concat_tick_data_{config.target}_10min_10min/'
    X_train_dict = load_concat_data('X_train_dict', mid_path=mid_path)
    X_test_dict = load_concat_data('X_test_dict', mid_path=mid_path)
    y_train_dict = load_concat_data('y_train_dict', mid_path=mid_path)
    y_test_dict = load_concat_data('y_test_dict', mid_path=mid_path)

    # 法2：shuffle后按比例划分
    ...

    # scale X data and save scaler
    scaler_dict = defaultdict(dict)
    for stk_name in X_train_dict.keys():
        for num in range(4):
            # num = str(num)
            dp = AggDataPreprocessor()
            X_train_dict[stk_name][num] = dp.std_scale(X_train_dict[stk_name][num], refit=True)
            X_test_dict[stk_name][num] = dp.std_scale(X_test_dict[stk_name][num], refit=False)
            dp.save_scaler(scaler_root, FILE_FMT_scaler.format(stk_name, num, '_'))
        config.complete_status['scalers'].append(stk_name)
        config.complete_status['scalers'] = list(set(config.complete_status['scalers']))
        save_status(is_tick=True)

    # # gather different stk data
    # all_X_train = defaultdict(pd.DataFrame)
    # all_y_train = defaultdict(pd.Series)
    # all_X_test = defaultdict(pd.DataFrame)
    # all_y_test = defaultdict(pd.Series)
    # for stk_name in X_train_dict.keys():
    #     for num in range(4):
    #         all_X_train[num] = pd.concat([all_X_train[num], X_train_dict[stk_name][num]], axis=0)
    #         all_y_train[num] = pd.concat([all_y_train[num].rename(config.target), y_train_dict[stk_name][num]], axis=0)
    # for stk_name in X_test_dict.keys():
    #     for num in range(4):
    #         all_X_test[num] = pd.concat([all_X_test[num], X_test_dict[stk_name][num]], axis=0)
    #         all_y_test[num] = pd.concat([all_y_test[num].rename(config.target), y_test_dict[stk_name][num]], axis=0)

    # train & save model
    USE_PCA = False
    USE_LINEAER = False
    USE_FLAML = False
    USE_LGBM = False
    USE_AUTOGLUON = True
    IMBALANCED_DATA_CONTINUES = False
    IMBALANCED_DATA_BINARY = True
    assert np.array([USE_LINEAER, USE_FLAML, USE_LGBM, USE_AUTOGLUON]).sum() <= 1

    mid_path = ''
    models = {}
    lob_stat = LobStatistics()
    for stk_name in X_train_dict.keys():
        stat = pd.DataFrame()
        all_y_pred = pd.Series(dtype='float32')
        all_y_pred_trunc = pd.Series(dtype='float32')
        all_y_test = pd.Series(dtype='float32')
        all_y_train = pd.Series(dtype='float32')
        all_w_train = pd.Series(dtype='float32')
        for num in range(4):
            model = None
            # num = str(num)

            X_train, y_train = X_train_dict[stk_name][num], y_train_dict[stk_name][num]
            X_test, y_test = X_test_dict[stk_name][num], y_test_dict[stk_name][num]

            # drop nan
            nan_idx = np.logical_or(X_train.isna().any(axis=1).values, y_train.isna().values)
            X_train, y_train = X_train.loc[~nan_idx], y_train.loc[~nan_idx]
            nan_idx = np.logical_or(X_test.isna().any(axis=1).values, y_test.isna().values)
            X_test, y_test = X_test.loc[~nan_idx], y_test.loc[~nan_idx]

            # shuffle
            seed = num
            X_train, y_train = shuffle(X_train, y_train, random_state=seed)
            X_test, y_test = shuffle(X_test, y_test, random_state=seed)
            _y_train = deepcopy(y_train)
            _y_test = deepcopy(y_test)
            _y_train.index = X_train.index
            _y_test.index = X_test.index

            # pca
            if USE_PCA:
                pca = PCA(0.95)  # 95% of variance
                pca.fit(X_train)
                print('n_components_', pca.n_components_)
                X_train = pd.DataFrame(pca.transform(X_train), index=X_train.index,
                                       columns=[f'pc{i}' for i in range(pca.n_components_)])
                X_test = pd.DataFrame(pca.transform(X_test), index=X_test.index,
                                      columns=[f'pc{i}' for i in range(pca.n_components_)])
                with open(scaler_root + f'pca_{num}.pkl', 'wb') as fw:
                    pickle.dump(pca, fw)

            # # linear
            if USE_LINEAER:
                # model = RidgeCV(cv=5, gcv_mode='auto')
                model = LassoCV(cv=5, n_jobs=1, random_state=0, tol=1e-5, precompute=True, selection='random',
                                max_iter=3000)
                model.fit(X_train, y_train)
                print(f"linear coef: {model.coef_}")
                all_y_train = pd.concat([all_y_train, _y_train], axis=0)

            # flaml. deprecated
            if USE_FLAML:
                from flaml import AutoML

                mid_path = f'flaml_{min_timedelta.total_seconds()}_{agg_timedelta.total_seconds()}_{pred_timedelta.total_seconds()}/'

                automl_settings = {
                    "time_budget": 120,  # in seconds
                    "metric": 'rmse',
                    "task": 'regression',
                    "log_file_name": model_root + f"flaml_training{num}_{min_timedelta.total_seconds()}_{agg_timedelta.total_seconds()}_{pred_timedelta.total_seconds()}.log",
                    "verbose": 2,  # int, default=3 | Controls the verbosity, higher means more messages.
                }
                model = AutoML()
                model.fit(X_train, _y_train, **automl_settings)
                print(f'best model for period {num}', model.model.estimator)  # Print the best model
                all_y_train = pd.concat([all_y_train, _y_train], axis=0)

            # autogluon
            if USE_AUTOGLUON:


                # [medium_quality,best_quality,good_quality] see autogluon/tabular/configs/presets_configs.py
                quality = 'medium_quality'
                mid_path = f'{config.target}_{quality}_{min_timedelta.total_seconds()}_{agg_timedelta.total_seconds()}_{pred_timedelta.total_seconds()}/'
                model_dir = model_root + f"autogluon/" + mid_path + f"{code_dict[stk_name]}/period{num}/"

                if IMBALANCED_DATA_CONTINUES:
                    _w_train = tar_weight(_y_train, bins=20)
                    _train_data = pd.concat([X_train, _y_train, _w_train], axis=1)
                elif IMBALANCED_DATA_BINARY:
                    value_counts = _y_train.value_counts().sort_index()
                    w_minority = len(_y_train) / value_counts.loc[1]
                    w_majority = len(_y_train) / value_counts.loc[0]
                    summation = w_majority + w_minority
                    w_minority = w_minority / summation
                    w_majority = w_majority / summation
                    _w_train = _y_train.replace({0: w_majority, 1: w_minority}).rename('weight')
                    _train_data = pd.concat([X_train, _y_train, _w_train], axis=1)
                else:
                    _train_data = pd.concat([X_train, _y_train], axis=1)
                _test_data = pd.concat([X_test, _y_test], axis=1)

                # ========================= model configs ==========================
                # nn_options = {  # specifies non-default hyperparameter values for neural network models
                #     'num_epochs': 10,  # number of training epochs (controls training time of NN models)
                #     'learning_rate': space.Real(1e-4, 1e-2, default=5e-4, log=True),
                #     # learning rate used in training (real-valued hyperparameter searched on log-scale)
                #     # todo 如果是预测ret，那需要改为tanh等输出负值的activation
                #     'activation': space.Categorical('relu', 'softrelu', 'sigmoid'),
                #     # activation function used in NN (categorical hyperparameter, default = first entry)
                #     'dropout_prob': space.Real(0.0, 0.3, default=0.1),
                #     # dropout probability (real-valued hyperparameter)
                # }
                # gbm_options = {  # specifies non-default hyperparameter values for lightGBM gradient boosted trees
                #     'num_boost_round': 100,  # number of boosting rounds (controls training time of GBM models)
                #     'num_leaves': space.Int(lower=26, upper=66, default=36),
                #     # number of leaves in trees (integer hyperparameter)
                # }
                # hyperparameters = {  # hyperparameters of each model type
                #     'GBM': gbm_options,
                #     'NN_TORCH': nn_options,  # NOTE: comment this line out if you get errors on Mac OSX
                # }  # When these keys are missing from hyperparameters dict, no models of that type are trained
                #
                # num_trials = 5  # try at most 5 different hyperparameter configurations for each type of model
                # search_strategy = 'auto'  # to tune hyperparameters using random search routine with a local scheduler
                # hyperparameter_tune_kwargs = {  # HPO is not performed unless hyperparameter_tune_kwargs is specified
                #     'num_trials': num_trials,
                #     'scheduler': 'local',
                #     'searcher': search_strategy,
                # }  # Refer to TabularPredictor.fit docstring for all valid values

                # problem_type = (options: ‘binary’, ‘multiclass’, ‘regression’, ‘quantile’)
                # eval_metric =
                # classification: [‘accuracy’, ‘balanced_accuracy’, ‘f1’, ‘f1_macro’, ‘f1_micro’, ‘f1_weighted’, ‘roc_auc’, ‘roc_auc_ovo_macro’, ‘average_precision’, ‘precision’, ‘precision_macro’, ‘precision_micro’, ‘precision_weighted’, ‘recall’, ‘recall_macro’, ‘recall_micro’, ‘recall_weighted’, ‘log_loss’, ‘pac_score’]
                # regression: [‘root_mean_squared_error’, ‘mean_squared_error’, ‘mean_absolute_error’, ‘median_absolute_error’, ‘r2’]
                task_config = ['regression', 'root_mean_squared_error']
                if config.target == Target.vol_chg.name:
                    if IMBALANCED_DATA_BINARY:
                        # metric=metric_dict['fbeta_score']
                        # metric=metric_dict['focal_loss']
                        # metric=metric_dict['focal_loss_noweight']
                        metric=metric_dict['fbeta_focal_score']

                        task_config = ['binary', metric]
                    else:
                        task_config = ['binary', 'accuracy']
                if IMBALANCED_DATA_CONTINUES or IMBALANCED_DATA_BINARY:
                    task_config.append('weight')
                else:
                    task_config.append(None)

                # - tabular predictor
                hyperparameters = get_hyperparameter_config('default')
                # hyperparameters.pop('RF')
                # - multimodal predictor
                # hyperparameters = {}
                # if IMBALANCED_DATA_BINARY:
                #     hyperparameters.update({
                #         # https://auto.gluon.ai/stable/tutorials/multimodal/advanced_topics/focal_loss.html#create-dataset
                #         "optimization.loss_function": "focal_loss",
                #         # Lin, T. Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017). Focal loss for dense object detection. In Proceedings of the IEEE international conference on computer vision (pp. 2980-2988).
                #         "optimization.focal_loss.alpha": [w_majority, w_minority],
                #         "optimization.focal_loss.gamma": 2.0,
                #         # optimization.focal_loss.gamma - float which controls how much to focus on the hard samples. Larger value means more focus on the hard samples.
                #         "optimization.focal_loss.reduction": "sum",
                #         "optimization.max_epochs": 10,
                #     })

                # todo 为不同模型添加random state
                # for m_type in hyperparameters.keys():
                #     try:
                #         if isinstance(hyperparameters[m_type],dict):
                #             hyperparameters[m_type]['random_state']=num
                #         elif isinstance(hyperparameters[m_type],list):
                #             new_hp=[]
                #             for d in hyperparameters[m_type]:
                #                 d.update({'random_state': num})
                #                 new_hp.append(d)
                #             hyperparameters[m_type]=new_hp
                #         else:
                #             raise NotImplementedError('hyperparameters[m_type] is not the type')
                #     except Exception as e:
                #         print('cannot add random state', hyperparameters[m_type])
                #         print(e.__traceback__)
                # ===========================================================

                # ========================= models ==========================
                # model = MultiModalPredictor(label=str(config.target),
                #                             path=model_dir,
                #                             problem_type=task_config[0],
                #                             eval_metric=task_config[1],
                #                             verbosity=2,
                #                             )
                # model.fit(_train_data, time_limit=int(60 * 5), presets=quality,
                #           hyperparameters=hyperparameters,
                #           holdout_frac=0.1,
                #           )

                model = TabularPredictor(label=str(config.target),
                                         path=model_dir,
                                         problem_type=task_config[0],
                                         eval_metric=task_config[1],
                                         log_to_file=True,
                                         log_file_path=model_dir + f"autogluon_training{num}.log",
                                         verbosity=2,
                                         sample_weight=task_config[2],
                                         )
                model.fit(_train_data, time_limit=60 * 2, fit_weighted_ensemble=True, presets=quality,
                          hyperparameters=hyperparameters,
                          holdout_frac=0.1,
                          # hyperparameter_tune_kwargs=hyperparameter_tune_kwargs,
                          )
                # ===========================================================

                all_y_train = pd.concat([all_y_train, _y_train], axis=0)
                if IMBALANCED_DATA_CONTINUES or IMBALANCED_DATA_BINARY:
                    all_w_train = pd.concat([all_w_train, _w_train], axis=0)

            # predict
            if USE_LINEAER or USE_FLAML:
                # train
                y_pred1 = model.predict(X_train)
                _train_stat = lob_stat.stat_pred_error(np.power(y_train, 2), np.power(y_pred1, 2),
                                                       name=f'train_stat_{num}')
                stat = pd.concat([stat, _train_stat], axis=1)
                # test
                y_pred = model.predict(X_test)
                _test_stat = lob_stat.stat_pred_error(np.power(y_test, 2), np.power(y_pred, 2), name=f'test_stat_{num}')
                stat = pd.concat([stat, _test_stat], axis=1)
                if config.target == Target.ret.name or config.target == Target.mid_p_ret.name:  # truncate
                    y_threshold = y_train.loc[(y_train != 0).values].quantile(0.05)  # 最小的5%
                    # train
                    y_pred1_trunc = deepcopy(y_pred1)
                    y_pred1_trunc.loc[(y_pred1_trunc < y_threshold)] = 0
                    _train_stat = lob_stat.stat_pred_error(np.power(y_train, 2), np.power(y_pred1_trunc, 2),
                                                           name=f'train_trunc{y_threshold:.6f}_stat_{num}')
                    stat = pd.concat([stat, _train_stat], axis=1)
                    print('train stat trunc\n', _train_stat)
                    # test
                    y_pred_trunc = deepcopy(y_pred)
                    y_pred_trunc.loc[(y_pred_trunc < y_threshold)] = 0
                    _test_stat = lob_stat.stat_pred_error(np.power(y_test, 2), np.power(y_pred_trunc, 2),
                                                          name=f'test_trunc{y_threshold:.6f}_stat_{num}')
                    stat = pd.concat([stat, _test_stat], axis=1)
                    print('test stat trunc\n', _test_stat)
                    all_y_pred_trunc = pd.concat([all_y_pred_trunc, pd.Series(y_pred_trunc)], axis=0, ignore_index=True)
                all_y_test = pd.concat([all_y_test, y_test], axis=0)
                all_y_pred = pd.concat([all_y_pred, y_pred], axis=0)
            if USE_LGBM:
                pass
            if USE_AUTOGLUON:
                y_pred1 = model.predict(_train_data)
                _train_stat = lob_stat.stat_pred_error(np.power(y_train, 2), np.power(y_pred1, 2),
                                                       name=f'autogluon_train_stat_{num}', task=task_config[0])
                stat = pd.concat([stat, _train_stat], axis=1)
                print('train stat\n', _train_stat)
                y_pred = model.predict(_test_data)
                _test_stat = lob_stat.stat_pred_error(np.power(y_test, 2), np.power(y_pred, 2),
                                                      name=f'autogluon_test_stat_{num}', task=task_config[0])
                stat = pd.concat([stat, _test_stat], axis=1)
                print('test stat\n', _test_stat)
                if config.target == Target.ret.name or config.target == Target.mid_p_ret.name:  # truncate
                    y_threshold = y_train.loc[(y_train != 0).values].quantile(0.05)  # 最小的5%
                    # train
                    y_pred1_trunc = deepcopy(y_pred1)
                    y_pred1_trunc.loc[(y_pred1_trunc < y_threshold)] = 0
                    _train_stat = lob_stat.stat_pred_error(np.power(y_train, 2), np.power(y_pred1_trunc, 2),
                                                           name=f'autogluon_train_trunc{y_threshold:.6f}_stat_{num}',
                                                           task=task_config[0])
                    stat = pd.concat([stat, _train_stat], axis=1)
                    print('train stat trunc\n', _train_stat)
                    # test
                    y_pred_trunc = deepcopy(y_pred)
                    y_pred_trunc.loc[(y_pred_trunc < y_threshold)] = 0
                    _test_stat = lob_stat.stat_pred_error(np.power(y_test, 2), np.power(y_pred_trunc, 2),
                                                          name=f'autogluon_test_trunc{y_threshold:.6f}_stat_{num}',
                                                          task=task_config[0])
                    stat = pd.concat([stat, _test_stat], axis=1)
                    print('test stat trunc\n', _test_stat)

                    all_y_pred_trunc = pd.concat([all_y_pred_trunc, pd.Series(y_pred_trunc)], axis=0, ignore_index=True)

                all_y_test = pd.concat([all_y_test, y_test], axis=0)
                all_y_pred = pd.concat([all_y_pred, y_pred], axis=0)
                # eval = model.evaluate(_test_data, silent=True)
                # print(eval)
                # lboard = model.leaderboard(_test_data, silent=True) # package bug
                # print(lboard)
                # results = model.fit_summary(show_plot=False)
                # print(results)
                # todo 为什么这么慢
                # feature_importance = model.feature_importance(_test_data, silent=True)
                # print(feature_importance)
                # feature_importance.to_csv(res_root + 'stats/'+f"feature_importance{num}.csv")

            # save
            mname = f'{stk_name}_{config.target}'
            save_model(model_root, FILE_FMT_model.format(mname, num, get_model_name(model)), model)
            suffix=f'{metric.__name__}_5_0.8/'
            # todo: recursively generate dirs
            res_root = config.res_root + '波动率方向预测/'
            if not os.path.exists(res_root):
                os.mkdir(res_root)
            if not os.path.exists(res_root + 'stats/'):
                os.mkdir(res_root + 'stats/')
            if not os.path.exists(res_root + 'preds/'):
                os.mkdir(res_root + 'preds/')
            if not os.path.exists(res_root + 'stats/' + mid_path):
                os.mkdir(res_root + 'stats/' + mid_path)
            if not os.path.exists(res_root + 'preds/' + mid_path):
                os.mkdir(res_root + 'preds/' + mid_path)
            mid_path=mid_path+suffix
            if not os.path.exists(res_root + 'stats/' + mid_path):
                os.mkdir(res_root + 'stats/' + mid_path)
            if not os.path.exists(res_root + 'preds/' + mid_path):
                os.mkdir(res_root + 'preds/' + mid_path)
            stat.to_csv(res_root + f"stats/{mid_path}stat_{stk_name}_{get_model_name(model)}_{config.target}.csv")
            np.power(all_y_pred, 2).to_csv(
                res_root + f"preds/{mid_path}all_y_pred_{stk_name}_{get_model_name(model)}_{config.target}.csv")
            np.power(all_y_pred_trunc, 2).to_csv(
                res_root + f"preds/{mid_path}all_y_pred_trunc_{stk_name}_{get_model_name(model)}_{config.target}.csv")
            np.power(all_y_test, 2).to_csv(res_root + f"preds/{mid_path}all_y_test_{stk_name}_{config.target}.csv")
            all_y_train.to_csv(res_root + f"preds/{mid_path}all_y_train_{stk_name}_{config.target}.csv")
            if IMBALANCED_DATA_CONTINUES or IMBALANCED_DATA_BINARY:
                all_w_train.to_csv(res_root + f"preds/{mid_path}all_w_train_{stk_name}_{config.target}.csv")
        print(stat)
        print(all_y_pred)
        config.complete_status['models'].append(stk_name)
        config.complete_status['models'] = list(set(config.complete_status['models']))
        save_status(is_tick=True)
    # '''

# from backtest import config
# import pandas as pd
# import matplotlib.pyplot as plt
# def load_concat_data(name,mid_path):
#     import pickle
#     import config
#     with open(config.data_root + mid_path+ f"{name}.pkl", 'rb') as fr:
#         data_dict = pickle.load(fr)
#     return data_dict
# # 读数据
# mid_path = f'concat_tick_data_{config.target}_10min_10min/'
# X_train_dict = load_concat_data('X_train_dict', mid_path=mid_path)
# X_test_dict = load_concat_data('X_test_dict', mid_path=mid_path)
# y_train_dict = load_concat_data('y_train_dict', mid_path=mid_path)
# y_test_dict = load_concat_data('y_test_dict', mid_path=mid_path)
# tar_dict = load_concat_data('tar_dict', mid_path=mid_path)
# threshold_dict = load_concat_data('threshold_dict', mid_path=mid_path)
