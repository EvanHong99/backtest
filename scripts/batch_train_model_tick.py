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

path = os.path.join(os.path.dirname(__file__), os.pardir)
sys.path.append(path)

from collections import defaultdict
from copy import deepcopy

from sklearn.linear_model import LassoCV
from sklearn.utils import shuffle
from sklearn.decomposition import PCA

from autogluon.tabular import TabularPredictor
from autogluon.common import space
from autogluon.tabular.configs.hyperparameter_configs import get_hyperparameter_config

from config import *
from support import *
from datafeeds.datafeed import LobDataFeed
from support import update_date, LobColTemplate, save_model, get_model_name
from preprocessors.preprocess import AggDataPreprocessor, LobTimePreprocessor
from statistic_tools.statistics import LobStatistics


def _rolling_rv(series: pd.Series):
    # assert (series>=0).values.all() # 已在外部代码保证这一点，出于效率考虑，暂时注释该行
    temp = series.dropna()
    return np.sqrt(np.matmul(temp.T, temp))


if __name__ == '__main__':
    load_status(is_tick=True)

    '''
    skip = 0
    limit = 10
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

    counter1 = deepcopy(counter)
    for k in counter.keys():
        if len(counter[k]) < 3:
            if counter1.get(k) is not None:
                counter1.pop(k)
        for kk in counter[k]:
            if counter[k][kk] < 6:
                if counter1[k].get(kk) is not None:
                    counter1[k].pop(kk)
                    logging.warning(f"incompletely constructed data {kk}")

    counter = counter1
    f_dict = {k: sorted(list(v.keys())) for k, v in counter.items()}
    # 按权重排序
    _stknames = set(list(f_dict.keys()))
    w_sorted_stknames = np.array([True if k in _stknames else False for k in code_dict.keys()])
    f_dict = {k: f_dict[k] for k in np.array(list(code_dict.keys()))[w_sorted_stknames]}
    f_dict = {k: v for k, v in list(f_dict.items())[skip:limit]}
    f_dict1 = deepcopy(f_dict)
    for k, v in f_dict1.items():
        if len(v) < 3:
            f_dict.pop(k)
            logging.warning(f"incompletely constructed data {k}")
            if k in ['AAPL', 'AMZN', 'GOOG', 'INTC', 'MSFT']: continue
            # assert k not in config.complete_status['scalers']
    print(f"waiting list {list(f_dict.keys())}")


    # 读数据
    tp=LobTimePreprocessor()
    feature_dict = defaultdict(lambda: defaultdict(list))  # feature_dict={date:{stkname:[data0,data1,data2,data3]}
    tar_dict = defaultdict(dict)  # feature_dict={date:{stkname:tar_data}
    datafeed = LobDataFeed()
    f_dict1 = deepcopy(f_dict)
    loaded_files=defaultdict(list)
    for stk_name, ymd_tuple in list(f_dict.items()):
        for yyyy, mm, dd in ymd_tuple:
            try:
                update_date(yyyy, mm, dd)
                for num in range(4):
                    feature = datafeed.load_feature(data_root + 'tick_data/', config.date, stk_name, num)
                    feature=tp.del_untrade_time(feature,cut_tail=True,strip=strip_time,drop_last_row=True) # fixme feature数据最后一行会有0.01s的精度，需要drop_last_row=True，但最终应修改feature生成算法
                    # feature=tp.add_head_tail(feature,head_timestamp=config.ranges[num][0],tail_timestamp=config.ranges[num][1])
                    feature_dict[config.date][stk_name].append(feature.dropna(how='all'))
                    # print(f"new feature {stk_name} {num}")
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
            # temp=LobTimePreprocessor().del_untrade_time(temp, cut_tail=True,strip=strip_time)
            wap1 = (temp[str(LobColTemplate('a', 1, 'p'))] * temp[str(LobColTemplate('b', 1, 'v'))] +
                    temp[str(LobColTemplate('b', 1, 'p'))] * temp[str(LobColTemplate('a', 1, 'v'))]) / 2
            temp = temp.groupby(level=0).last().resample(min_freq).last().sort_index().ffill()
            wap1 = wap1.groupby(level=0).last().resample(min_freq).last().sort_index().ffill()
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
                # 计算(t+1,t+pred_delta]
                ratio = 1
                rolling_rows = int(shift_rows / ratio)
                tar = (temp[str(LobColTemplate('a', 1, 'p'))] + temp[str(LobColTemplate('b', 1, 'p'))]) / 2
                try:
                    tar = np.log(tar / tar.shift(ratio))  # log ret
                except RuntimeWarning:
                    pass
                tar1 = deepcopy(tar)
                tar = tar.shift(-rolling_rows).reset_index()[[0]].rolling(rolling_rows, min_periods=rolling_rows,
                                                                          center=False,
                                                                          step=rolling_rows).apply(_rolling_rv)
                tar = pd.Series(tar.values.flatten(), index=tar1.iloc[tar.index].index)
            # todo 若预测ret等
            tar= pd.Series(np.sqrt(tar),index=tar.index,name=config.target)
            tar = LobTimePreprocessor().del_untrade_time(tar, cut_tail=True,strip=strip_time)  # 不能忘
            tar_dict[config.date][stk_name] = tar.rename(config.target)
            loaded_files[stk_name].append(config.date)
    print(loaded_files)
    f_dict = f_dict1

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
                X, y = dp.align_Xy(feature, tar_dict[ymd][stk_name], pred_timedelta=pred_timedelta)  # 最重要的是对齐X y
                X_train_dict[stk_name][num].append(X)
                y_train_dict[stk_name][num].append(y.rename(config.target))
    for stk_name in y_train_dict.keys():
        for num in y_train_dict[stk_name].keys():
            # num = str(num)
            X_train_dict[stk_name][num]=pd.concat(X_train_dict[stk_name][num],axis=0)
            y_train_dict[stk_name][num]=pd.concat(y_train_dict[stk_name][num],axis=0)
    # test data
    X_test_dict = defaultdict(lambda: defaultdict(list))
    y_test_dict = defaultdict(lambda: defaultdict(list))
    for ymd in list(feature_dict.keys())[train_len:]:
        for stk_name, features in feature_dict[ymd].items():
            for num, feature in enumerate(features):
                # num = str(num)
                # X, y按照iloc已经一一对应
                X, y = dp.align_Xy(feature, tar_dict[ymd][stk_name], pred_timedelta=pred_timedelta)  # 最重要的是对齐X y
                X_test_dict[stk_name][num].append(X)
                y_test_dict[stk_name][num].append(y.rename(config.target))
    for stk_name in y_test_dict.keys():
        for num in y_test_dict[stk_name].keys():
            # num = str(num)
            X_test_dict[stk_name][num]=pd.concat(X_test_dict[stk_name][num],axis=0)
            y_test_dict[stk_name][num]=pd.concat(y_test_dict[stk_name][num],axis=0)
    # save
    save_concat_data(dict(X_train_dict),'X_train_dict')
    save_concat_data(dict(X_test_dict),'X_test_dict')
    save_concat_data(dict(y_train_dict),'y_train_dict')
    save_concat_data(dict(y_test_dict),'y_test_dict')
    
    '''

    # 读数据
    X_train_dict = load_concat_data('X_train_dict')
    X_test_dict = load_concat_data('X_test_dict')
    y_train_dict = load_concat_data('y_train_dict')
    y_test_dict = load_concat_data('y_test_dict')

    # todo 法2：shuffle后按比例划分
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

    # # gather different stk data todo: 不同股票不同模型
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
    USE_LGBM=False
    USE_AUTOGLUON = True
    assert np.array([USE_LINEAER, USE_FLAML, USE_LGBM,USE_AUTOGLUON]).sum() <= 1

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
            seed=num
            X_train, y_train = shuffle(X_train, y_train,random_state=seed)
            X_test, y_test = shuffle(X_test, y_test,random_state=seed)
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
                def tar_weight(_y_train, bins=20):
                    nrows = len(_y_train)
                    hist, bin_edge = np.histogram(_y_train, bins=bins)
                    hist[hist==0]+=1 # 避免出现RuntimeWarning: divide by zero encountered in divide  bin_w = np.sqrt(nrows / hist)
                    bin_idx = np.digitize(_y_train, bin_edge,
                                          right=False) - 1  # series中元素是第几个bin，可能会导致溢出，因为有些值（比如0）刚好在bin边界上，所以要把这些归到第一个bin（即现在的bin_idx==1处）
                    bin_idx[bin_idx == 0] += 1
                    bin_idx -= 1
                    bin_w = np.sqrt(nrows / hist)
                    series_w = np.array([bin_w[i] for i in bin_idx])
                    series_w = nrows / sum(series_w) * series_w
                    series_w = pd.Series(series_w, index=_y_train.index, name='weight')
                    return series_w


                mid_path = f'good_quality_{min_timedelta.total_seconds()}_{agg_timedelta.total_seconds()}_{pred_timedelta.total_seconds()}/'

                _w_train = tar_weight(_y_train, bins=20)
                _train_data = pd.concat([X_train, _y_train, _w_train], axis=1)
                _test_data = pd.concat([X_test, _y_test], axis=1)
                # _train_data = TabularDataset(_train_data)
                # _test_data = TabularDataset(_test_data)
                model_dir = model_root + f"autogluon/"+mid_path+f"{code_dict[stk_name]}/period{num}/"

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

                model = TabularPredictor(label=str(config.target),
                                         path=model_dir,
                                         problem_type='regression',
                                         eval_metric='rmse',
                                         log_to_file=True,
                                         log_file_path=model_dir + f"autogluon_training{num}.log",
                                         verbosity=2,
                                         sample_weight='weight',
                                         )

                hyperparameters=get_hyperparameter_config('default')
                hyperparameters.pop('RF')
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

                # [medium_quality,best_quality,good_quality] see autogluon/tabular/configs/presets_configs.py
                model.fit(_train_data, time_limit=60*5, fit_weighted_ensemble=True, presets='medium_quality',
                          hyperparameters=hyperparameters,
                          holdout_frac=0.2,
                          # hyperparameter_tune_kwargs=hyperparameter_tune_kwargs,
                          )
                all_y_train = pd.concat([all_y_train, _y_train], axis=0)
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
                if target != Target.vol.name:  # truncate
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
                                                       name=f'autogluon_train_stat_{num}')
                stat = pd.concat([stat, _train_stat], axis=1)
                print('train stat\n', _train_stat)
                y_pred = model.predict(_test_data)
                _test_stat = lob_stat.stat_pred_error(np.power(y_test, 2), np.power(y_pred, 2),
                                                      name=f'autogluon_test_stat_{num}')
                stat = pd.concat([stat, _test_stat], axis=1)
                print('test stat\n', _test_stat)
                if target != Target.vol.name:  # truncate
                    y_threshold = y_train.loc[(y_train != 0).values].quantile(0.05)  # 最小的5%
                    # train
                    y_pred1_trunc = deepcopy(y_pred1)
                    y_pred1_trunc.loc[(y_pred1_trunc < y_threshold)] = 0
                    _train_stat = lob_stat.stat_pred_error(np.power(y_train, 2), np.power(y_pred1_trunc, 2),
                                                           name=f'autogluon_train_trunc{y_threshold:.6f}_stat_{num}')
                    stat = pd.concat([stat, _train_stat], axis=1)
                    print('train stat trunc\n', _train_stat)
                    # test
                    y_pred_trunc = deepcopy(y_pred)
                    y_pred_trunc.loc[(y_pred_trunc < y_threshold)] = 0
                    _test_stat = lob_stat.stat_pred_error(np.power(y_test, 2), np.power(y_pred_trunc, 2),
                                                          name=f'autogluon_test_trunc{y_threshold:.6f}_stat_{num}')
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
                feature_importance = model.feature_importance(_test_data, silent=True)
                print(feature_importance)
                feature_importance.to_csv(res_root + 'stats/'+f"feature_importance{num}.csv")

            # save
            mname = f'{stk_name}_{config.target}'
            save_model(model_root, FILE_FMT_model.format(mname, num, get_model_name(model)), model)


            res_root=res_root+'波动率预测/'
            if not os.path.exists(res_root+'stats/'+mid_path):
                os.mkdir(res_root+'stats/'+mid_path)
            if not os.path.exists(res_root + 'preds/' + mid_path):
                os.mkdir(res_root +'preds/' + mid_path)
            stat.to_csv(res_root + f"stats/{mid_path}stat_{stk_name}_{get_model_name(model)}_{config.target}.csv")
            np.power(all_y_pred, 2).to_csv(
                res_root + f"preds/{mid_path}all_y_pred_{stk_name}_{get_model_name(model)}_{config.target}.csv")
            np.power(all_y_pred_trunc, 2).to_csv(
                res_root + f"preds/{mid_path}all_y_pred_trunc_{stk_name}_{get_model_name(model)}_{config.target}.csv")
            np.power(all_y_test, 2).to_csv(res_root + f"preds/{mid_path}all_y_test_{stk_name}_{config.target}.csv")
            all_y_train.to_csv(res_root + f"preds/{mid_path}all_y_train_{stk_name}_{config.target}.csv")
            all_w_train.to_csv(res_root + f"preds/{mid_path}all_w_train_{stk_name}_{config.target}.csv")
        print(stat)
        print(all_y_pred)
        config.complete_status['models'].append(stk_name)
        config.complete_status['models'] = list(set(config.complete_status['models']))
        save_status(is_tick=True)
    # '''
