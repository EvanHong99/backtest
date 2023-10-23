# -*- coding=utf-8 -*-
# @File     : lob_strategy.py
# @Time     : 2023/8/7 9:20
# @Author   : EvanHong
# @Email    : 939778128@qq.com
# @Project  : 2023.06.08超高频上证50指数计算
# @Description:


from preprocessors.preprocess import LobTimePreprocessor
from strategies.base_strategy import BaseStrategy
from support import *


class LobStrategy(BaseStrategy):
    """
    todo 使用L3 ret，即各档位逐笔委托的大小也可以计算出相应的因子
    """

    def __init__(self,
                 max_close_timedelta,
                 # model_root,
                 # file_root,
                 # timestamp,
                 # stk_names,
                 # levels,
                 # target,
                 # freq,
                 # pred_n_steps,
                 # use_n_steps,
                 # drop_current=False,
                 *args,
                 **kwargs):
        """
        note
        ----
        应该以order为单位去管理止损止盈，在backtest中进行维护？

        :param model_root:
        :param file_root:
        :param date:
        :param stk_name:
        :param levels:
        :param target: 需要确保data中有一列==target的值
        :param args:
        :param kwargs:
        """
        super().__init__(*args, **kwargs)
        self.max_close_timedelta = max_close_timedelta
        # self.clean_obh_dict = None
        # self.models = []
        # self.model_root = model_root
        # self.file_root = file_root
        # self.timestamp = timestamp
        # self.stk_names = stk_names
        # self.levels = levels
        # self.target = target
        # self.freq = freq
        # self.pred_n_steps = pred_n_steps
        # self.use_n_steps = use_n_steps
        # self.drop_current = drop_current
        # self.param = {
        #     'drop_current': self.drop_current,
        #     'pred_n_steps': self.pred_n_steps,
        #     'target': self.target,
        #     'use_n_steps': self.use_n_steps,
        # }

    # def load_models(self, model_root, stk_names):
    #     model_loader = LobModelFeed(model_root=model_root, stk_names=stk_names)
    #     self.models = model_loader.models
    #     return self.models
    #
    # def load_data(self, file_root, timestamp, stk_names) -> pd.DataFrame:
    #     """
    #
    #     :param file_root:
    #     :param timestamp:
    #     :param stk_names:
    #     :return: pd.DataFrame,(clean_obh_dict+vol_tov).asfreq(freq='10ms', method='ffill')
    #     """
    #     ret = LobDataFeed(file_root=file_root, timestamp=timestamp, stk_names=stk_names)
    #     self.clean_obh_dict = ret.load_clean_obh(snapshot_window=self.levels)
    #     self.vol_tov = ret.load_vol_tov()
    #     self.alldata = pd.concat([self.clean_obh_dict, self.vol_tov], axis=1).ffill()
    #     # note 必须先将clean_obh填充到10ms，否则交易频率是完全不规律的，即可能我只想用5个frame的数据来预测，但很可能用上了十秒的信息
    #     self.alldata = self.alldata.asfreq(freq='10ms', method='ffill')
    #     return self.alldata
    #
    # def calc_features(self, df):
    #     # todo: 时间不连续、不规整，过于稀疏，归一化细节
    #     # LobTimePreprocessor.split_by_trade_period(self.alldata)
    #     fe = LobFeatureEngineering()
    #     feature = fe.generate(df)
    #     return feature
    #
    # # testit
    # def preprocess_data(self):
    #     """
    #     将数据划分为4份，每份一小时
    #     :return:
    #     """
    #     ltp = LobTimePreprocessor()
    #     # 必须先将数据切分，否则会导致11:30和13:00之间出现跳变
    #     self.alldatas = ltp.split_by_trade_period(self.alldata)
    #     self.alldatas = [ltp.add_head_tail(cobh, head_timestamp=pd.to_datetime(s),
    #                                        tail_timestamp=pd.to_datetime(e)) for cobh, (s, e) in
    #                      zip(self.alldatas, ranges)]
    #     # 不能对alldatas change freq，否则会导致损失数据点
    #     self.features = [self.calc_features(ret) for ret in self.alldatas]
    #     self.features = [ltp.add_head_tail(feature, head_timestamp=pd.to_datetime(s),
    #                                        tail_timestamp=pd.to_datetime(e)) for feature, (s, e) in
    #                      zip(self.features, ranges)]
    #     self.features = [ltp.change_freq(feature, freq=freq) for feature in self.features]
    #
    #     # self.alldata = pd.concat([self.alldata, self.feature], axis=1)
    #
    #     # 过早change freq会导致损失数据点
    #     self.alldatas = [ltp.change_freq(d, freq=freq) for d in self.alldatas]
    #
    #     self.alldatas = [pd.merge(ret, feature, left_index=True, right_index=True) for ret, feature in
    #                      zip(self.alldatas, self.features)]
    #
    #     return self.alldatas
    #
    # def transform_data(self, alldatas):
    #     """
    #     主要是归一化和跳取数据，用于信号生成和回测，无需打乱
    #     :param alldatas:
    #     :return:
    #     """
    #     Xs = []
    #     ys = []
    #     for num in range(len(alldatas)):
    #         dp = ShiftDataPreprocessor()
    #
    #         X, y = dp.get_flattened_Xy(alldatas, num, self.target, self.pred_n_steps, self.use_n_steps,
    #                                    self.drop_current)
    #         param = dp.sub_illegal_punctuation(str(self.param))
    #         dp.load_scaler(scaler_root, FILE_FMT_scaler.format(self.stk_names, num, param))
    #
    #         cols = X.columns
    #         index = X.index
    #         X = pd.DataFrame(dp.scaler.transform(X), columns=cols, index=index)
    #
    #         X = X.iloc[::self.use_n_steps]
    #         y = y.iloc[::self.use_n_steps]
    #
    #         Xs.append(X)
    #         ys.append(y)
    #
    #     return Xs, ys

    # def generate_signals(self, y_pred: pd.Series):
    #     """
    #     输出需要统一范式，即(timestamp,side,type,price,volume)
    # 
    #     :return:
    #     """
    # 
    #     def calc_signal(pred_ret, threshold=0.0008, drift=0):
    #         """
    # 
    #         :param pred_ret:
    #         :param threshold:
    #         :param drift: 用于修正偏度至正态
    #         :return:
    #         """
    #         if pred_ret > drift + threshold:
    #             return 1
    #         elif pred_ret < drift - threshold:
    #             return -1
    #         else:
    #             return 0
    # 
    #     def calc_net_ret(direction, ret_long, ret_short, name):
    #         net_ret = np.zeros_like(direction)
    #         net_ret = np.where(direction == 1, ret_long, net_ret)
    #         net_ret = np.where(direction == -1, ret_short, net_ret)
    #         net_ret = pd.Series(net_ret, index=direction.index, name=name)
    #         return net_ret
    # 
    #         # self.models = self.load_models(self.model_root, self.stk_names)
    #         # self.alldata = self.load_data(file_root=self.file_root, timestamp=self.timestamp, stk_names=self.stk_names)
    #         # self.alldatas = self.preprocess_data()
    #         # self.Xs, self.ys = self.transform_data(self.alldatas)
    # 
    #         # self.y_preds = {}
    #         # self.all_signals = {}
    #         # all_stats = pd.DataFrame()
    #         # all_stats_winrate = pd.DataFrame()
    #         # for num, (X_test, y_test, model) in enumerate(zip(self.Xs, self.ys, self.models)):
    #         #     y_pred = model.predict(X_test)
    #         #     y_pred = pd.Series(y_pred,
    #         #                        index=X_test.index + timedelta(milliseconds=int(self.freq[:-2]) * self.pred_n_steps),
    #         #                        name=f'pred_{self.target}_{self.pred_n_steps * 0.2}s').sort_index()
    #         #     self.y_preds[num] = y_pred
    #         self.all_signals[num] = y_pred.apply(lambda x: calc_signal(x, threshold=0.0008, drift=0))
    #         self.all_signals[num] = self.all_signals[num].rename(y_pred.name.replace('pred', 'signal'))
    # 
    #         # return self.y_preds,self.all_signals
    # 
    #         stat = LobStatistics.stat_pred_error(y_test, y_pred,
    #                                              name="{}_period{}".format(
    #                                                  re.findall('\w*', str(type(model)).split('.')[-1])[0], num))
    #         all_stats = pd.concat([all_stats, stat], axis=1)
    # 
    #         tempdata = self.alldatas[num][['current', 'a1_p', 'b1_p']]
    #         temp = pd.concat(
    #             [tempdata.rename(columns={k: k + '_open' for k in tempdata.columns}).drift(self.pred_n_steps),
    #              tempdata.rename(columns={k: k + '_close' for k in tempdata.columns})], axis=1)
    #         temp = pd.concat([temp, self.y_preds[num]], axis=1).dropna(how='any')
    #         temp['ret'] = np.log(temp['current_close'] / temp['current_open'])  # 用于看预测方向准确率
    #         temp['ret_long'] = np.log(
    #             temp['b1_p_close'] / temp['a1_p_open'])  # 做多时对手价的收益率（仅仅是固定时间平仓，且该时间为预测的期数pred_n_steps）
    #         temp['ret_short'] = np.log(temp['b1_p_open'] / temp['a1_p_close'])  # short时是open/close, -log(a/b)==log(b/a)
    #         self.temp = temp
    #         direction = temp[f'pred_{self.target}_{self.pred_n_steps * 0.2}s'].apply(
    #             lambda x: calc_signal(x, threshold=0.0008, drift=0))
    #         # net_ret赚了就是正
    #         net_ret = calc_net_ret(direction, temp['ret_long'], temp['ret_short'], name=f'period{num}')
    #         stat = LobStatistics.stat_winrate(net_ret, direction, counterpart=True, params=self.param)  # 统计对手价下的胜率
    #         net_ret1 = calc_net_ret(direction, temp['ret'], -temp['ret'], name=f'period{num}')
    #         stat1 = LobStatistics.stat_winrate(net_ret1, direction, counterpart=False, params=self.param)  # 统计方向预测准确率
    #         stat['drct_acc'] = stat1['win_rate']  # 预测方向准确率
    #         all_stats_winrate = pd.concat([all_stats_winrate, stat], axis=1)
    #         self.net_ret = net_ret
    #         self.net_ret1 = net_ret1
    # 
    #     print(all_stats)
    #     print(all_stats_winrate)
    #     return all_stats, all_stats_winrate

    def generate_signals(self, y_pred: pd.Series, stk_name: str, threshold=0.0008, drift=0) -> 'pd.DataFrame':
        """
        输出需要统一范式，即(timestamp,stk_name,side,type,price,volume)

        :return:
        """

        def calc_side(pred_ret):
            """

            :param pred_ret:
            :param threshold:
            :param drift: 用于修正偏度至正态
            :return:
            """
            if pred_ret > drift + threshold:
                return 1
            elif pred_ret < drift - threshold:
                return -1
            else:
                return 0

        def calc_type(pred_ret):
            return OrderTypeInt.market.value

        def calc_price_limit(pred_ret):
            return 0

        def calc_volume(pred_ret):
            return 100

        def calc_close_time(pred_ret):
            return pred_ret + self.max_close_timedelta

        # 计算开仓信号
        signals = y_pred.to_frame().agg([
            calc_side,
            calc_type,
            calc_price_limit,
            calc_volume
        ])
        signals.index = y_pred.index  # 不做这一步出来的index很怪，会有一个后缀
        signals.columns = ['side', 'type', 'price_limit', 'volume']
        signals.loc[:, 'stk_name'] = [stk_name] * len(signals)
        signals.loc[:, 'timestamp'] = y_pred.index
        signals.loc[:, 'action'] = ['open'] * len(signals)
        signals.loc[:, 'seq'] = list(range(len(signals)))  # 平仓signal和开仓公用
        # todo 去除连续出现的信号

        # 计算平仓信号
        close_signals = signals.copy(deep=True)
        close_signals.loc[:, 'timestamp'] = calc_close_time(signals['timestamp'])
        close_signals.loc[:, 'action'] = ['close'] * len(close_signals)

        # 拼接数据
        signals = pd.concat([signals, close_signals], axis=0)
        signals = signals.set_index(['timestamp', 'side'], drop=False).sort_index(ascending=True)  # 先卖后买
        # signals = y_pred.apply(lambda x: calc_side(x))
        # signals = signals.rename(str(y_pred.name).replace('pred', 'signal')).to_frame()
        signals = LobTimePreprocessor.del_untrade_time(signals, cut_tail=True)
        return signals

        # return self.y_preds,self.all_signals
