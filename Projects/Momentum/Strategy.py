#!/usr/bin/env python3

import logging
import sys
import traceback
sys.path.append('../..')

import random
import math
import pickle as pk
# import argparse
from random import randrange
# import threading

from collections import deque
from collections import defaultdict
from datetime import date, timedelta

import numpy as np
import pandas as pd

from scipy import stats

from matplotlib import pyplot as plt

from Broker import Broker
from Portfolio import Portfolio
from Order import Order
pd.set_option('display.max_columns', None)
# sys.path[0] = '../..'
import config
from QuantumCapital import constants
from QuantumCapital.DBManager import DBManager
from QuantumCapital.PlotManager import *

# Configuring logging rules -------------------------------------------------------------- /
a = logging.Logger.manager.loggerDict  # Disabling other loggers
for k in a:
    a[k].disabled = True

for handler in logging.root.handlers[:]:  # Disabling root logger handlers
    logging.root.removeHandler(handler)

LOG_FILE_PATH = f'Logs/{datetime.now()}'

logging.basicConfig(
    filename=LOG_FILE_PATH,
                    format='%(asctime)s %(levelname)s [%(funcName)s]: %(message)s',
                    datefmt='%Y-%m-%d | %H:%M:%S',
                    level=logging.ERROR)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

requests_logger = logging.getLogger('requests')
requests_logger.setLevel(logging.ERROR)

handler = logging.StreamHandler()
handler.setLevel(logging.ERROR)
logger.addHandler(handler)
requests_logger.addHandler(handler)

# logging.getLogger().setLevel(logging.DEBUG)  # If you want logger write in console
logger.info(
    '\n\n_______________________________________________________________________________________________________')
# --------------------------------------------------------------------------------------- /

# try:
    # parser = argparse.ArgumentParser()
    # parser.add_argument('param1_name', type=str)
    # parser.add_argument('param1_value', type=float)
    # parser.add_argument('param2_name', type=str)
    # parser.add_argument('param2_value', type=float)
    # args = parser.parse_args()

params = {'dots_n': 125,
          'step': 1,
          'rebalance_window': 5,
          'vola_window': 20,
          'cap_quantile': 0.25,
          'momentum_top_n': 20,
          'momentum_thresh': 40,
          'com_reserve_perc': 1,
          'ma_window': 200,
          'change_thresh': 0.2,
          'close_all_at_low_ma': False}

# params[args.param1_name] = args.param1_value
# params[args.param2_name] = args.param2_value
#
# params = {'dots_n': int(params['dots_n']),
#           'step': int(params['step']),
#           'rebalance_window': int(params['rebalance_window']),
#           'vola_window': int(params['vola_window']),
#           'cap_quantile': float(params['cap_quantile']),
#           'momentum_top_n': int(params['momentum_top_n']),
#           'momentum_thresh': float(params['momentum_thresh']),
#           'com_reserve_perc': float(params['com_reserve_perc']),
#           'ma_window': int(params['ma_window']),
#           'change_thresh': float(params['change_thresh']),
#           'close_all_at_low_ma': bool(params['close_all_at_low_ma'])}
#
# params['vola_window'] = params['dots_n']

logger.info(params)
print(params)

dots_n = params['dots_n']
step = params['step']
rebalance_window = params['rebalance_window']
vola_window = params['vola_window']
# assert dots_n >= vola_window + 1

cap_quantile = params['cap_quantile']
momentum_top_n = params['momentum_top_n']
momentum_thresh = params['momentum_thresh']
com_reserve_perc = params['com_reserve_perc']  # Commission reserve in percent
ma_window = params['ma_window']
change_thresh = params['change_thresh']
close_all_at_low_ma = params['close_all_at_low_ma']

filename = f'Data/market_val_{datetime.now()}.pk'

unequal_dates_n = 0
momentum_table = 'momentum_tuning'

assert step <= dots_n
assert rebalance_window <= dots_n

# start_date = date(2002, 7, 3)  # RUSSEL
# start_date = date(2016, 1, 1)
# start_date = date(2000, 7, 3)  # SPY

# start_date = date(2019, 7, 5)
# end_date = date(2020, 8, 26)  # RUSSEL
broker = Broker(step=step, logger=logger)
# broker.current_tickers = random.sample(broker.current_tickers, 1)
# broker.current_tickers = ['med', 'amd']
init_cash = 1e6
portfolio = Portfolio(init_cash)

DB_USERNAME = config.DB_USERNAME
DB_PASS = config.DB_PASS
# dbm = DBManager(DB_USERNAME, DB_PASS, 'ODS')
# except Exception as e:
#     with open(f'Errors/{args.param1_value}|{args.param2_value}.txt', 'w') as file:
#         file.write(repr(e))
#         file.write(traceback.format_exc())
#         exit()


def get_random_tickers(num):
    bars = broker.get_curr_bars()
    tickers = [ticker for ticker in bars if bars[ticker][0] is not None and bars[ticker][0] > 50]
    return random.sample(tickers, num)


# subset_tickers = get_random_tickers(1)
# subset_tickers = []


figures = {}
# tickers_for_plot = get_random_tickers(1)
# tickers_for_plot = subset_tickers
# tickers_for_plot = [broker.current_tickers[0]]
# tickers_for_plot = broker.current_tickers
# tickers_for_plot = ['AMD']
tickers_for_plot = []

for ticker in tickers_for_plot:
    fig, ax = plt.subplots(1, 2, figsize=(12, 8))
    figures[ticker] = (fig, ax)


def update_history_and_coefs(dt, new_bars, day_cnt):
    absent_tickers = set(history.keys()) - set(new_bars.keys())
    for ticker in absent_tickers:
        del history[ticker]

    for ticker in new_bars:
        if not (new_bars[ticker][0] == new_bars[ticker][1] == None):
            history[ticker][0].append((dt,) + new_bars[ticker])
            if len(history[ticker][0]) > max(dots_n, vola_window + 1):
                history[ticker][0].popleft()
            if len(history[ticker][0]) == max(dots_n, vola_window + 1) and day_cnt % rebalance_window == 0:
                history[ticker][1] = momentum_score([close for dt, close, vol in list(history[ticker][0])[-dots_n:]])
                # if ticker == 'algn':
                #     logger.info(f'Mom vals: {history[ticker][1]}')
                #     print(f'Mom vals: {history[ticker][1]}')
                history[ticker][2] = cap_score([close for dt, close, vol in list(history[ticker][0])[-rebalance_window:]],
                                               [vol for dt, close, vol in list(history[ticker][0])[-rebalance_window:]])
                history[ticker][3] = volatility_score(
                    [close for dt, close, vol in list(history[ticker][0])[-vola_window - 1:]])


def momentum_score(closes):
    log_closes = np.log(closes)
    slope, intercept, r_value, p_value, std_err = stats.linregress(np.arange(len(closes)), log_closes)
    annualized_slope = (np.power(np.exp(slope), 252) - 1) * 100
    score = annualized_slope * (r_value ** 2)
    return score, slope, intercept, r_value, annualized_slope


def exp_function(intercept, slope, days):
    res = []
    for i in days:
        res.append(math.exp(intercept) * math.exp(i * slope))
    return res


def cap_score(closes, volumes):
    assert len(closes) == len(volumes) == rebalance_window
    avg_close = sum(closes) / rebalance_window
    avg_vol = sum(volumes) / rebalance_window
    return avg_close * avg_vol


def volatility_score(prices):
    prices = pd.Series(prices)
    return float(prices.pct_change().rolling(vola_window).std().iloc[-1])


prev_c = 0


def plot_figures():
    global prev_c
    for ticker in tickers_for_plot:
        # print(ticker)
        fig, ax = figures[ticker]
        fig.suptitle(ticker, fontsize=16)
        ax[0].clear()
        # print(ticker)
        # print(history[ticker][0])
        # print([close for dt, close, vol in list(history[ticker][0])[-dots_n:]])
        plot_closes(ax[0], [dt.date() for dt, close, vol in list(history[ticker][0])[-dots_n:]],
                    [close for dt, close, vol in list(history[ticker][0])[-dots_n:]],
                    handle_ylims=True, color='green')
        ax[1].clear()
        plot_closes(ax[1], [dt.date() for dt, close, vol in list(history[ticker][0])[-dots_n:]],
                    [close for dt, close, vol in list(history[ticker][0])[-dots_n:]],
                    linestyle='None', marker='.', handle_ylims=True, n_lines=2)

        m_vals = history[ticker][1]
        c_val = history[ticker][3]

        if m_vals is not None:
            x = np.arange(min(dots_n, len(history[ticker][0])))
            days = [dt.date() for dt, _, _ in list(history[ticker][0])]
            plot_closes(ax[1], days,
                        exp_function(m_vals[2], m_vals[1], x), color='red', n_lines=2)
            # ax[1].set_title(f'Ann. Slope: {round(m_vals[4], 2)}% \n'
            #                 f'R2: {round(m_vals[3], 5)}')
            text_kwargs = dict(ha='left', va='center', fontsize=28,
                               color='blue' if m_vals[4] > 0 else 'red',
                               transform=ax[1].transAxes)
            ax[1].text(0.1, 0.9, f'Ann. Return: {round(m_vals[4], 2)}% \n'
                                 f'Confidence: {round(m_vals[3], 5)}', **text_kwargs)

        open_poses = [op for op in portfolio.open_positions if op.ticker == ticker]
        plot_open_poses(ax[0], open_poses)

        closed_poses = [cp for cp in portfolio.closed_positions if cp.ticker == ticker]
        plot_closed_poses(ax[0], closed_poses)
        text_kwargs = dict(ha='left', va='center', fontsize=28, color='blue' if c_val is not None and round(c_val, 2) >= prev_c else 'red', transform=ax[0].transAxes)
        ax[0].text(0.1, 0.9, f'Power: {round(m_vals[0], 2) if m_vals is not None else None} \n'
                           f'Volatility: {round(c_val, 2) if c_val is not None else None}', **text_kwargs)

        # ax[0].set_title(f'M: {round(m_vals[0], 2) if m_vals is not None else None} \n'
        #                 f'C: {round(c_val, 2) if c_val is not None else None}')

        plt.draw()
        plt.pause(0.001)
        prev_c = c_val if c_val is not None else 0

# def plot_mv():
#     # ax.clear()
#     # ax.plot(hist)
#     ax_mv.clear()
#     plot_closes(ax_mv, dt_hist, mv_hist)
#     # plt.draw()
#     # plt.pause(0.001)
#     plt.savefig('market_val.png', bbox_inches='tight')


def save_mv():
    with open(filename, 'wb') as file:
        pk.dump((params, (dt_hist, mv_hist)), file, protocol=pk.HIGHEST_PROTOCOL)


def make_orders(day_cnt):

    if day_cnt % rebalance_window == 0 and day_cnt + 1 >= dots_n:
        dt = broker.current_dt
        spx_curr = spx[spx.date.isin(pd.date_range(datetime(dt.year, dt.month, dt.day, 0, 0, 0) - timedelta(5),
                                                   datetime(dt.year, dt.month, dt.day, 23, 59, 59)))]
        spx_curr = spx_curr.iloc[-1]

        assert broker.current_dt >= spx_curr.date
        if broker.current_dt != spx_curr.date:
            global unequal_dates_n
            unequal_dates_n += 1
        # if not np.isnan(spx_curr.ma) and spx_curr.ma < spx_curr.spx:
        buy_tickers = {'ticker': [], 'price': [], 'momentum': [], 'cap': [], 'volatility': []}
        for ticker in history:
            ticker_hist = history[ticker]
            if ticker_hist[3] is not None and ticker_hist[3] != 0 and len(ticker_hist[0]) == max(dots_n, vola_window + 1):
                buy_tickers['ticker'].append(ticker)
                buy_tickers['price'].append(ticker_hist[0][-1][1])
                buy_tickers['momentum'].append(ticker_hist[1][0])
                buy_tickers['cap'].append(ticker_hist[2])
                buy_tickers['volatility'].append(ticker_hist[3])
        buy_tickers = pd.DataFrame(buy_tickers)
        cap_thresh = buy_tickers.cap.quantile(cap_quantile)
        buy_tickers = buy_tickers[buy_tickers.cap > cap_thresh]

        buy_tickers.sort_values(by='momentum', inplace=True, ascending=False)
        buy_tickers = buy_tickers.iloc[:momentum_top_n]
        buy_tickers['inv_vola'] = 1 / buy_tickers.volatility
        inv_vola_sum = buy_tickers.inv_vola.sum()
        buy_tickers['inv_vola_norm'] = buy_tickers['inv_vola'] / inv_vola_sum
        market_val = 0.01 * (100 - com_reserve_perc) * portfolio.market_value
        buy_tickers['how_many_to_buy'] = (market_val * buy_tickers.inv_vola_norm / buy_tickers.price).apply(np.floor)
        buy_tickers = buy_tickers[buy_tickers.how_many_to_buy > 0]
        buy_tickers = buy_tickers[buy_tickers.momentum > momentum_thresh]

        buy_tickers.sort_values(by='how_many_to_buy', inplace=True)
        buy_tickers.reset_index(drop=True, inplace=True)
        # print(buy_tickers[['ticker', 'price', 'momentum', 'inv_vola_norm', 'how_many_to_buy']])

        buy_tickers_names = set(buy_tickers.ticker.values.tolist())
        open_tickers_names = portfolio.open_tickers()
        close_tickers_names = list(open_tickers_names - buy_tickers_names)

        for ticker in close_tickers_names:
            amount = portfolio.number_of_shares_open(ticker)
            broker.make_order(portfolio, Order(ticker, 'market', 'sell', amount, 'shares'))

        rem_idx_buy_ticks = []
        diff_list = []
        for i in range(buy_tickers.shape[0]):
            row = buy_tickers.iloc[i]
            ticker = row.ticker
            to_open = int(row.how_many_to_buy)
            assert to_open > 0
            open_shares = portfolio.number_of_shares_open(ticker)
            assert open_shares >= 0
            shares_diff = to_open - open_shares
            change = shares_diff / open_shares if open_shares != 0 else None

            if change is None:  # new ticker came
                if not np.isnan(spx_curr.ma) and spx_curr.ma < spx_curr.spx:
                    diff_list.append((ticker, shares_diff))
                else:
                    rem_idx_buy_ticks.append(i)
            else:
                if not close_all_at_low_ma:
                    if abs(change) > change_thresh:
                        diff_list.append((ticker, shares_diff))
                    else:
                        buy_tickers.loc[buy_tickers.iloc[i].name, 'how_many_to_buy'] = open_shares
                else:
                    if not np.isnan(spx_curr.ma) and spx_curr.ma < spx_curr.spx:
                        if abs(change) > change_thresh:
                            diff_list.append((ticker, shares_diff))
                        else:
                            buy_tickers.loc[buy_tickers.iloc[i].name, 'how_many_to_buy'] = open_shares
                    else:
                        diff_list.append((ticker, -open_shares))
                        rem_idx_buy_ticks.append(i)

            # Buy new or change > thresh
            # if (change is None and not np.isnan(spx_curr.ma) and spx_curr.ma < spx_curr.spx) or \
            #    (change is not None and abs(change) > change_thresh):
            #     diff_list.append((ticker, shares_diff))
            # else:
            #     if change is not None and abs(change) <= change_thresh:
            #         buy_tickers.loc[buy_tickers.iloc[i].name, 'how_many_to_buy'] = open_shares
            #     elif change is None and not np.isnan(spx_curr.ma) and spx_curr.ma >= spx_curr.spx:
            #         rem_idx_buy_ticks.append(i)
        buy_tickers.drop(buy_tickers.index[rem_idx_buy_ticks], inplace=True)
        print(f'SPX: {spx_curr.spx}, MA: {spx_curr.ma}')
        print(buy_tickers[['ticker', 'price', 'momentum', 'volatility', 'inv_vola_norm', 'how_many_to_buy']])

        logger.info(f'SPX: {spx_curr.spx}, MA: {spx_curr.ma}')
        logger.info(buy_tickers[['ticker', 'price', 'momentum', 'volatility', 'inv_vola_norm', 'how_many_to_buy']])

        diff_list.sort(key=lambda x: x[1])
        for i, (ticker, diff) in enumerate(diff_list):
            if diff < 0:
                broker.make_order(portfolio, Order(ticker, 'market', 'sell', abs(diff), 'shares'))
            elif diff > 0:
                broker.make_order(portfolio, Order(ticker, 'market', 'buy', diff, 'shares'))
            else:
                raise Exception
        bought_tickers = portfolio.get_open_tickers_df()
        assert buy_tickers.shape[0] == bought_tickers.shape[0]

        if buy_tickers.shape[0] != 0:
            merged = pd.merge(buy_tickers[['ticker', 'how_many_to_buy']], bought_tickers, on='ticker')
            error = merged['how_many_to_buy'] - merged['amount']
            error = float(error.sum())
        else:
            error = 0
        assert error == 0


# def write_results():
#     row = {}
#     for k in params:
#         row[k] = [params[k]]
#
#     row['profit'] = [portfolio.market_value / init_cash]
#
#     mv_series = pd.Series(mv_hist)
#     r = mv_series.diff()
#     sr = r.mean() / r.std() * np.sqrt(252)
#     row['sharpe'] = [sr]
#
#     spx_curr = spx.spx[spx.date.isin(pd.date_range(broker.first_dt, broker.last_dt))]
#     spx_np = spx_curr.to_numpy()
#     mv_np = np.array(mv_hist)
#
#     error = mv_np.shape[0] - spx_np.shape[0]
#     print(error)
#     print(spx_np.shape[0], mv_np.shape[0])
#
#     if abs(error) < 10:
#         if error < 0:
#             remove_idxs = random.sample(list(range(0, spx_np.shape[0])), error)
#             spx_np = np.delete(spx_np, remove_idxs)
#         elif error > 0:
#             remove_idxs = random.sample(list(range(0, mv_np.shape[0])), error)
#             print(remove_idxs)
#             mv_np = np.delete(mv_np, remove_idxs)
#     print(spx_np.shape[0], mv_np.shape[0])
#
#     assert spx_np.shape[0] == mv_np.shape[0]
#
#     cov = np.cov(mv_np, spx_np)[0, 1]
#     var = np.var(spx_np)
#     beta = cov / var
#     row['beta'] = [beta]
#     row['filename'] = [filename]
#     try:
#         dbm.insert_df_fast(pd.DataFrame(row), momentum_table)
#     except:
#         raise Exception(pd.DataFrame(row))
#     dbm.commit()


spx = pd.read_csv('spx index.csv')
spx['date'] = pd.to_datetime(spx.date, format='%d.%m.%Y')
# spx = spx[spx.date >= datetime(broker.current_dt.year, broker.current_dt.month, broker.current_dt.day)]
spx['ma'] = spx['spx'].rolling(window=ma_window).mean()
# spx['ma'] = 0
# spx = spx.iloc[:900]
print('Last date spx:', spx.iloc[-1])

history = defaultdict(lambda: [deque(), None, None, None])


fig, ax_mv = plt.subplots(1, 1, figsize=(12, 8))
mv_hist = []
dt_hist = []


def main():
    day_cnt = 0
    sum_day_runtime = 0
    finished = False

    # df = dbm.select_df(f"select * from {momentum_table} where dots_n={params['dots_n']}"
    #                    f"                               and step={params['step']}"
    #                    f"                               and rebalance_window={params['rebalance_window']}"
    #                    f"                               and vola_window={params['vola_window']}"
    #                    f"                               and cap_quantile={params['cap_quantile']}"
    #                    f"                               and momentum_top_n={params['momentum_top_n']}"
    #                    f"                               and momentum_thresh={params['momentum_thresh']}"
    #                    f"                               and ma_window={params['ma_window']}"
    #                    f"                               and change_thresh={params['change_thresh']}"
    #                    f"                               and close_all_at_low_ma={params['close_all_at_low_ma']}")
    # if df.shape[0] > 0:
    #     raise Exception('This set of params already run')
    # else:
    while not finished:
        start_time = time.time()

        print(day_cnt, broker.current_dt)
        print('Market value:', portfolio.market_value)

        logger.info(broker.current_dt)
        logger.info(f'Market value: {portfolio.market_value}')

        mv_hist.append(portfolio.market_value)
        dt_hist.append(broker.current_dt)
        # broker.current_tickers = subset_tickers
        # broker.current_tickers = ['amzn']
        # broker.current_tickers = ['med']
        # broker.current_tickers = ['med', 'amd']

        bars = broker.get_curr_bars()
        update_history_and_coefs(broker.current_dt, bars, day_cnt)
        make_orders(day_cnt)
        # print(len())

        # plot_figures()
        # plot_mv()
        save_mv()
        finished = broker.go_next_rth_day(portfolio)
        day_cnt += 1
        spent_time = time.time() - start_time
        sum_day_runtime += spent_time
        print('Average runtime:', sum_day_runtime / day_cnt)
        print('----------------------------')
        print()
        logger.info('--------------------------------\n')


if __name__ == '__main__':
    import time
    import cProfile

    # with cProfile.Profile() as pr:
    #     start = time.time()
    try:
        main()

        # params[args.param1_name] = args.param1_value
        # params[args.param2_name] = args.param2_value
    except Exception as e:
        pass
        # with open(f'Errors/{args.param1_value}|{args.param2_value}.txt', 'w') as file:
        #     file.write(repr(e))
        #     file.write(traceback.format_exc())

        # print('Spent time: ', time.time() - start)
    # write_results()
    # pr.print_stats()
    # print(filename)

    # with open('market_val200.pk', 'rb') as file:
    #     cc = pk.load(file)
    #
    # print(cc)
