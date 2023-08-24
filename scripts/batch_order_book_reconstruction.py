# -*- coding=utf-8 -*-
# @File     : batch_order_book_reconstruction.py
# @Time     : 2023/7/28 16:05
# @Author   : EvanHong
# @Email    : 939778128@qq.com
# @Project  : 2023.06.08超高频上证50指数计算
# @Description: 生成obh，即列为价格，行为该价格下的委托数
import os
import sys
path = os.path.join(os.path.dirname(__file__), os.pardir)
sys.path.append(path)

import config
from support import *
# from config import *
from calc_events import calc_events
from datafeeds.datafeed import LobDataFeed
from order_book_reconstruction import *

if __name__ == '__main__':

    skip = 0
    limit = 50
    snapshot_window = 10
    ohlc = pd.read_csv(data_root + 'hs300_ohlc.csv', index_col='code')
    load_status()
    y, m = '2022', '06'
    for stk_name in list(code_dict.keys())[skip:limit]:
        if stk_name in exclude: continue
        if stk_name in config.complete_status['orderbooks']:continue
        for dd in [23, 28, 29]:
            config.y, config.m, config.d, config.date, config.date1, config.start, config.end, config.important_times, config.ranges = update_date(
                y, m, str(dd))
            y, m, d, date, date1, start, end, important_times, ranges = config.y, config.m, config.d, config.date, config.date1, config.start, config.end, config.important_times, config.ranges
            print(f"update to date {date} {date1}")

            if code_dict[stk_name].endswith('XSHE'): continue
            print(f"start {stk_name} {date}")
            symbol = code_dict[stk_name]
            order_details = get_order_details(data_root, date, symbol)
            trade_details = get_trade_details(data_root, date, symbol)

            self = OrderBook(stk_name, symbol, snapshot_window=snapshot_window)
            self.reconstruct(order_details, trade_details)

            self.check_trade_details(trade_details)
            self.events = calc_events(trade_details, order_details)

            self.price_history.to_csv(detail_data_root + FILE_FMT_price_history.format(date, stk_name), index=False)
            self.order_book_history.to_csv(detail_data_root + FILE_FMT_order_book_history.format(date, stk_name))
            self.my_trade_details.to_csv(detail_data_root + FILE_FMT_my_trade_details.format(date, stk_name),
                                         index=False)
            self.vol_tov.to_csv(detail_data_root + FILE_FMT_vol_tov.format(date, stk_name), index=True)
            self.events.to_csv(detail_data_root + FILE_FMT_events.format(date, stk_name))

            datafeed = LobDataFeed(detail_data_root, date, stk_name=stk_name)
            datafeed.load_basic(detail_data_root, date, stk_name=stk_name)
            cobh_pp = LobCleanObhPreprocessor()
            cobh_pp.gen_and_save(datafeed, detail_data_root, date, stk_name=stk_name, snapshot_window=snapshot_window)

            print(f"finish {stk_name} {date}")

            # pd.concat([trade_details['price'].reset_index()['price'].rename('ground_truth'),self.my_trade_details['price'].rename('recnstr')],axis=1).plot(title=stk_names).get_figure().savefig(res_root+f'current_{stk_names}.png',dpi=1200,bbox_inches='tight')
        config.complete_status['orderbooks'].append(stk_name)
        save_status()