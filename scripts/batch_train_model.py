# -*- coding=utf-8 -*-
# @File     : batch_train_model.py
# @Time     : 2023/8/11 16:33
# @Author   : EvanHong
# @Email    : 939778128@qq.com
# @Project  : 2023.06.08超高频上证50指数计算
# @Description: 训练模型，并保存相关的scaler和model
import os
import sys
import config

path = os.path.join(os.path.dirname(__file__), os.pardir)
sys.path.append(path)

from collections import defaultdict
from copy import deepcopy

from sklearn.linear_model import LassoCV
from sklearn.utils import shuffle
from sklearn.decomposition import PCA

from flaml import AutoML
from autogluon.tabular import TabularPredictor

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

    skip = 0
    limit = 10
    load_status()
    # 读取已处理数据文件名从而确认有多少数据可用于后续计算
    f_dict = defaultdict(list)
    legal_cobh = set()
    legal_feat = set()
    for r, d, f in os.walk(detail_data_root):
        if len(f) == 0: logging.error("find no file")
        counter = defaultdict(lambda: defaultdict(int))
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
            if counter[k][kk] < 10:
                if counter1.get(k) is not None:
                    counter1.pop(k)
    counter = counter1
    f_dict = {k: sorted(list(v.keys())) for k, v in counter.items()}
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
    data_dict = defaultdict(lambda: defaultdict(list))  # data_dict={date:{stkname:[data0,data1,data2,data3]}
    tar_dict = defaultdict(dict)  # data_dict={date:{stkname:tar_data}
    datafeed = LobDataFeed()
    f_dict1 = deepcopy(f_dict)
    for stk_name, features in f_dict.items():
        for yyyy, mm, dd in features:
            try:
                for num in range(4):
                    update_date(yyyy, mm, dd)
                    feature = datafeed.load_feature(detail_data_root, config.date, stk_name, num)
                    data_dict[config.date][stk_name].append(feature.dropna(how='all'))
                    # print(f"new feature {stk_name} {num}")
            except FileNotFoundError as e:
                logging.warning(f"missing feature {(stk_name, yyyy, mm, dd)}")
                if f_dict1.get(stk_name) is not None:
                    f_dict1.pop(stk_name)
                if data_dict[config.date].get(stk_name) is not None:
                    data_dict[config.date].pop(stk_name)
                continue

            # target
            tar = None
            temp = datafeed.load_clean_obh(detail_data_root, config.date, stk_name, snapshot_window=use_level,
                                           use_cols=[str(LobColTemplate('a', 1, 'p')),
                                                     str(LobColTemplate('b', 1, 'p')),
                                                     str(LobColTemplate('a', 1, 'v')),
                                                     str(LobColTemplate('b', 1, 'v')),
                                                     LobColTemplate().current
                                                     ])
            wap1 = (temp[str(LobColTemplate('a', 1, 'p'))] * temp[str(LobColTemplate('b', 1, 'v'))] +
                    temp[str(LobColTemplate('b', 1, 'p'))] * temp[str(LobColTemplate('a', 1, 'v'))]) / 2
            temp = temp.groupby(level=0).last().resample('10ms').last().sort_index().ffill()
            wap1 = wap1.groupby(level=0).last().resample('10ms').last().sort_index().ffill()
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
                ratio = 1
                rolling_rows = int(shift_rows / ratio)
                tar = (temp[str(LobColTemplate('a', 1, 'p'))] + temp[str(LobColTemplate('b', 1, 'p'))]) / 2
                tar = np.log(tar / tar.shift(ratio))  # log ret
                tar1 = deepcopy(tar)
                tar = tar.shift(-rolling_rows).reset_index()[[0]].rolling(rolling_rows, min_periods=rolling_rows,
                                                                          center=False,
                                                                          step=rolling_rows).apply(_rolling_rv)
                tar = pd.Series(tar.values.flatten(), index=tar1.iloc[tar.index].index)
            tar = LobTimePreprocessor().del_untrade_time(tar, cut_tail=True)  # 不能忘
            tar_dict[config.date][stk_name] = tar.rename(config.target)
            print("load", detail_data_root, stk_name, config.date)
    f_dict = f_dict1

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
                num=str(num)
                X, y = dp.align_Xy(feature, tar_dict[ymd][stk_name], pred_timedelta=pred_timedelta)  # 最重要的是对齐X y
                X_train_dict[stk_name][num] = pd.concat([X_train_dict[stk_name][num], X], axis=0)
                y_train_dict[stk_name][num] = pd.concat([y_train_dict[stk_name][num].rename(config.target), y], axis=0)
    # test data
    X_test_dict = defaultdict(lambda: defaultdict(pd.DataFrame))
    y_test_dict = defaultdict(lambda: defaultdict(pd.Series))
    for ymd, stk_data in list(data_dict.items())[train_len:]:
        for stk_name, features in stk_data.items():
            for num, feature in enumerate(features):
                num=str(num)
                X, y = dp.align_Xy(feature, tar_dict[ymd][stk_name], pred_timedelta=pred_timedelta)  # 最重要的是对齐X y
                X_test_dict[stk_name][num] = pd.concat([X_test_dict[stk_name][num], X], axis=0)
                y_test_dict[stk_name][num] = pd.concat([y_test_dict[stk_name][num].rename(config.target), y], axis=0)

    # 法2：shuffle后按比例划分
    ...

    # scale X data and save scaler
    scaler_dict = defaultdict(dict)
    for stk_name in X_train_dict.keys():
        for num in range(4):
            num = str(num)
            dp = AggDataPreprocessor()
            X_train_dict[stk_name][num] = dp.std_scale(X_train_dict[stk_name][num], refit=True)[0]
            X_test_dict[stk_name][num] = dp.std_scale(X_test_dict[stk_name][num], refit=False)[0]
            dp.save_scaler(scaler_root, FILE_FMT_scaler.format(stk_name, num, '_'))
        config.complete_status['scalers'].append(stk_name)
        config.complete_status['scalers'] = list(set(config.complete_status['scalers']))
        save_status()

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
    USE_AUTOGLUON = True
    assert np.array([USE_LINEAER, USE_FLAML, USE_AUTOGLUON]).sum() <= 1

    models = {}
    lob_stat = LobStatistics()
    for stk_name in X_train_dict.keys():
        stat = pd.DataFrame()
        all_y_pred = pd.Series()
        all_y_pred_trunc = pd.Series()
        all_y_true = pd.Series()
        for num in range(4):
            model=None
            num = str(num)
            X_train, y_train = shuffle(X_train_dict[stk_name][num], y_train_dict[stk_name][num])
            X_test, y_test = shuffle(X_test_dict[stk_name][num], y_test_dict[stk_name][num])

            # drop nan
            nan_idx = np.logical_or(X_train.isna().any(axis=1).values, y_train.isna().values)
            X_train, y_train = X_train.loc[~nan_idx], y_train.loc[~nan_idx]
            nan_idx = np.logical_or(X_test.isna().any(axis=1).values, y_test.isna().values)
            X_test, y_test = X_test.loc[~nan_idx], y_test.loc[~nan_idx]

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

            # flaml
            if USE_FLAML:
                automl_settings = {
                    "time_budget": 120,  # in seconds
                    "metric": 'rmse',
                    "task": 'regression',
                    "log_file_name": model_root + f"flaml_training{num}.log",
                    "verbose": 2,  # int, default=3 | Controls the verbosity, higher means more messages.
                }
                model = AutoML()
                model.fit(X_train, y_train, **automl_settings)
                print(f'best model for period {num}', model.model.estimator)  # Print the best model

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
                model_dir = model_root + f"{stk_name}/"
                model = TabularPredictor(label=str(config.target),
                                         path=model_dir,
                                         problem_type='regression',
                                         eval_metric='rmse',
                                         log_to_file=True,
                                         log_file_path=model_dir + f"autogluon_training{num}.log",
                                         verbosity=2,
                                         )
                # [medium_quality,best_quality]
                model.fit(_train_data, time_limit=60, fit_weighted_ensemble=True, presets='medium_quality', )

            # predict
            if USE_LINEAER or USE_FLAML:
                y_pred1 = model.predict(X_train)
                _train_stat = lob_stat.stat_pred_error(y_train, y_pred1, name=f'train_stat_{num}')
                stat = pd.concat([stat, _train_stat], axis=1)

                y_train_min = y_train.loc[(y_train != 0).values].min()
                y_pred1.loc[(y_pred1 < y_train_min)] = 0
                _train_stat = lob_stat.stat_pred_error(y_train, y_pred1,
                                                       name=f'train_trunc{y_train_min:.6f}_stat_{num}')
                stat = pd.concat([stat, _train_stat], axis=1)

                y_pred = model.predict(X_test)
                y_pred_trunc = deepcopy(y_pred)
                y_pred_trunc.loc[(y_pred_trunc < y_train_min)] = 0

                stat = pd.concat([stat, lob_stat.stat_pred_error(y_test, y_pred, name=f'test_stat_{num}')], axis=1)
                stat = pd.concat([stat, lob_stat.stat_pred_error(y_test, y_pred_trunc,
                                                                 name=f'test_trunc{y_train_min:.6f}_stat_{num}')],
                                 axis=1)

                all_y_true = pd.concat([all_y_true, pd.Series(y_test)], axis=0, ignore_index=True)
                all_y_pred = pd.concat([all_y_pred, pd.Series(y_pred)], axis=0, ignore_index=True)
                all_y_pred_trunc = pd.concat([all_y_pred_trunc, pd.Series(y_pred_trunc)], axis=0, ignore_index=True)
            if USE_AUTOGLUON:
                y_pred1 = model.predict(_train_data)
                _train_stat = lob_stat.stat_pred_error(y_train, y_pred1, name=f'autogluon_train_stat_{num}')
                stat = pd.concat([stat, _train_stat], axis=1)
                print('train stat', _train_stat)

                y_train_min = y_train.loc[(y_train != 0).values].min()
                y_pred1.loc[(y_pred1 < y_train_min)] = 0
                _train_stat = lob_stat.stat_pred_error(y_train, y_pred1,
                                                       name=f'autogluon_train_trunc{y_train_min:.6f}_stat_{num}')
                stat = pd.concat([stat, _train_stat], axis=1)
                print('train stat trunc', _train_stat)

                y_pred = model.predict(_test_data)
                y_pred_trunc = deepcopy(y_pred)
                y_pred_trunc.loc[(y_pred_trunc < y_train_min)] = 0

                _test_stat = lob_stat.stat_pred_error(y_test, y_pred, name=f'autogluon_test_stat_{num}')
                stat = pd.concat([stat, _test_stat], axis=1)
                print('test stat', _test_stat)
                _test_stat = lob_stat.stat_pred_error(y_test, y_pred_trunc,
                                                      name=f'autogluon_test_trunc{y_train_min:.6f}_stat_{num}')
                stat = pd.concat([stat, _test_stat], axis=1)
                print('test stat trunc', _test_stat)

                all_y_true = pd.concat([all_y_true, pd.Series(y_test)], axis=0, ignore_index=True)
                all_y_pred = pd.concat([all_y_pred, pd.Series(y_pred)], axis=0, ignore_index=True)
                all_y_pred_trunc = pd.concat([all_y_pred_trunc, pd.Series(y_pred_trunc)], axis=0, ignore_index=True)

                eval = model.evaluate(_test_data, silent=True)
                lboard = model.leaderboard(_test_data, silent=True)
                results = model.fit_summary(show_plot=False)
                feature_importance = model.feature_importance(_test_data, silent=True)
                print(eval)
                # print(lboard)
                # print(results)
                print(feature_importance)
                feature_importance.to_csv(res_root + f"feature_importance{num}.csv")

            # save
            mname = f'{stk_name}_{config.target}'
            save_model(model_root, FILE_FMT_model.format(mname, num, get_model_name(model)), model)
        config.complete_status['models'].append(stk_name)
        config.complete_status['models'] = list(set(config.complete_status['models']))
        save_status()

        print(stat)
        print(all_y_pred)
        stat.to_csv(res_root + f"stats/stat_{stk_name}_{get_model_name(model)}_{config.target}.csv")
        all_y_pred.to_csv(res_root + f"preds/all_y_pred_{stk_name}_{get_model_name(model)}_{config.target}.csv")
        all_y_pred_trunc.to_csv(
            res_root + f"preds/all_y_pred_trunc_{stk_name}_{get_model_name(model)}_{config.target}.csv")
        all_y_true.to_csv(res_root + f"preds/all_y_true_{stk_name}_{config.target}.csv")
