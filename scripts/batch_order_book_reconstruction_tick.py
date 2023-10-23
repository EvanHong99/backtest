# -*- coding=utf-8 -*-
# @File     : batch_order_book_reconstruction.py
# @Time     : 2023/7/28 16:05
# @Author   : EvanHong
# @Email    : 939778128@qq.com
# @Project  : 2023.06.08超高频上证50指数计算
# @Description: 生成obh，即列为价格，行为该价格下的委托数
import os
import sys

import pandas as pd

from preprocessors.preprocess import LobTimePreprocessor

path = os.path.join(os.path.dirname(__file__), os.pardir)
sys.path.append(path)

import config
from support import *
from config import *
from calc_events import calc_events
from datafeeds.datafeed import LobDataFeed
from order_book_reconstruction import *

if __name__ == '__main__':

    skip = 0
    limit = 5
    snapshot_window = 5
    # ohlc = pd.read_csv(data_root + 'hs300_ohlc.csv', index_col='code')
    load_status(is_tick=True)

    stks=['贵州茅台']
    symbols=[code_dict[stk_name] for stk_name in stks]
    for stk_name in stks[skip:limit]:
        if stk_name in exclude: continue
        if stk_name != '贵州茅台':raise ValueError
        # [2020-01-01_2020-05-31, 2020-06-01_2020-12-31, 2021-01-01_2021-05-31, 2021-06-01_2021-12-31, 2022-01-01_2022-05-31, 2022-06-01_2022-10-31, 2022-11-01_2023-03-31, 2023-04-01_2023-08-28]

        files=['tick_贵州茅台_2017-01-01_2017-05-31.csv','tick_贵州茅台_2017-06-01_2017-09-30.csv','tick_贵州茅台_2017-10-01_2017-12-31.csv','tick_贵州茅台_2018-01-01_2018-05-31.csv','tick_贵州茅台_2018-06-01_2018-12-31.csv','tick_贵州茅台_2019-01-01_2019-05-31.csv','tick_贵州茅台_2019-06-01_2019-12-31.csv']
        for filename in files:
            data=pd.read_csv(data_root + filename)
            data = data.rename(columns={'time': 'timestamp'})
            data['timestamp'] = pd.to_datetime(data['timestamp'])
            data['date'] = data['timestamp'].apply(lambda x: str(x.date()))
            for d, g in data.groupby('date'):
                if (config.complete_status['orderbooks'].get(stk_name) is not None) and (d in config.complete_status['orderbooks'].get(stk_name)):continue
                g = g.set_index('timestamp')
                g['mid_p']=(g['a1_p']+g['b1_p'])/2

                # vol tov
                volume = g['volume'].diff().rename('volume')
                volume.iloc[0] = g['volume'].iloc[0]
                cum_vol = g['volume'].rename('cum_vol')
                turnover = g['money'].diff().rename('turnover')
                turnover.iloc[0] = g['money'].iloc[0]
                cum_turnover = g['money'].rename('cum_turnover')
                vol_tov = pd.concat([volume, cum_vol, turnover, cum_turnover], axis=1)

                g=g.drop(columns=['volume','money','date'])
                dd=g.index[0].date().strftime("%Y-%m-%d").split('-')
                update_date(dd[0],dd[1],dd[2])
                g = LobTimePreprocessor.del_untrade_time(g, cut_tail=True)
                g = LobTimePreprocessor.add_head_tail(g,
                                                              head_timestamp=config.important_times[
                                                                  'continues_auction_am_start'],
                                                              tail_timestamp=config.important_times[
                                                                  'continues_auction_pm_end'])

                g.to_csv(data_root + f"tick_data/{str(d).replace('-', '')}_{stk_name}_clean_obh.csv")
                vol_tov.to_csv(data_root + f"tick_data/{str(d).replace('-', '')}_{stk_name}_vol_tov.csv")

                print(f"finish {stk_name} {d}")

                # pd.concat([trade_details['price'].reset_index()['price'].rename('ground_truth'),self.my_trade_details['price'].rename('recnstr')],axis=1).plot(title=stk_names).get_figure().savefig(res_root+f'current_{stk_names}.png',dpi=1200,bbox_inches='tight')
                if config.complete_status['orderbooks'].get(stk_name) is not None:
                    config.complete_status['orderbooks'][stk_name].append(d)
                else: config.complete_status['orderbooks'][stk_name]=[d]
            if config.complete_status['orderbooks'].get(stk_name) is not None:
                config.complete_status['orderbooks'][stk_name]=list(sorted(set(config.complete_status['orderbooks'][stk_name])))
            save_status(is_tick=True)