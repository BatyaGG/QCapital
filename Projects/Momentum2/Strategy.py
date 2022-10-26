#!/usr/bin/env python3

import sys
import traceback
sys.path.append('../..')

import random
import math
import pickle as pk
import argparse
from random import randrange
# import threading

from collections import deque
from collections import defaultdict
from datetime import date, timedelta

import numpy as np
import pandas as pd
import torch

from scipy import stats

from matplotlib import pyplot as plt

sys.path.append('../../Backtester')
sys.path.append('../../QuantumCapital')

from Backtester.BrokerGPU import Broker
from Backtester.Portfolio import Portfolio
from Backtester.Order import Order
# pd.set_option('display.max_columns', None)
# sys.path[0] = '../..'
import config
from QuantumCapital import constants
from QuantumCapital.DBManager import DBManager
from QuantumCapital.PlotManager import *

# try:
# parser = argparse.ArgumentParser()
# parser.add_argument('param1_name', type=str)
# parser.add_argument('param1_value', type=float)
# parser.add_argument('param2_name', type=str)
# parser.add_argument('param2_value', type=float)
# args = parser.parse_args()

params = {'dots_n': 125,
          'step': 5,
          'rebalance_window': 5,
          'vola_window': 20,
          'cap_quantile': 0.25,
          'momentum_top_n': 20,
          'momentum_thresh': 40,
          'com_reserve_perc': 1,
          'ma_window': 200,
          'change_thresh': 0.0,
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

filename = f'Data/market_val_{datetime.today()}.pk'

unequal_dates_n = 0
momentum_table = 'momentum_tuning_'

assert step <= dots_n
assert rebalance_window <= dots_n

# start_date = date(2002, 7, 3)  # RUSSEL
# start_date = date(2016, 1, 1)
# start_date = date(2000, 7, 3)  # SPY

# start_date = date(2019, 7, 5)
# end_date = date(2020, 8, 26)  # RUSSEL
broker = Broker(dots_n, step=rebalance_window)
# broker.current_tickers = random.sample(broker.current_tickers, 1)
# broker.current_tickers = ['med', 'amd']
init_cash = 1e6
portfolio = Portfolio(init_cash)

DB_USERNAME = config.DB_USERNAME
DB_PASS = config.DB_PASS
dbm = DBManager(DB_USERNAME, DB_PASS, 'DWH')
# except Exception as e:
#     with open(f'Errors/{args.param1_value}|{args.param2_value}.txt', 'w') as file:
#         file.write(repr(e))
#         file.write(traceback.format_exc())
#         exit()


if torch.cuda.is_available():
    gpu = torch.device("cuda:0")
else:
    raise Exception('No GPU')


def get_random_tickers(num):
    bars = broker.get_curr_bars()
    tickers = [ticker for ticker in bars if bars[ticker][0] is not None and bars[ticker][0] > 50]
    return random.sample(tickers, num)


# subset_tickers = get_random_tickers(200)


figures = {}
# tickers_for_plot = get_random_tickers(1)
# tickers_for_plot = [broker.current_tickers[0]]
# tickers_for_plot = broker.current_tickers
tickers_for_plot = []
for ticker in tickers_for_plot:
    fig, ax = plt.subplots(1, 2, figsize=(12, 8))
    figures[ticker] = (fig, ax)


def momentum_score_calc():
    x = torch.arange(dots_n).view(dots_n, 1).repeat(1, len(tickers)).to(gpu)
    # yhat = torch.randn(df.shape[1], requires_grad=True, dtype=torch.float, device=gpu)

    price_df_log = torch.log(price_df)

    x_mean = (x * 1.0).mean(axis=0)
    x_minus_xmean = x - x_mean
    price_df_log_mean = price_df_log.mean(axis=0)
    price_df_log_minus_mean = price_df_log - price_df_log_mean
    num = (x_minus_xmean * price_df_log_minus_mean).sum(axis=0)
    den = (x_minus_xmean ** 2).sum(axis=0)
    b = num / den
    a = price_df_log_mean - b * x_mean

    # n = price_df_log.shape[0]
    # # denominator
    # d = (n * (x ** 2).sum(axis=0) - x.sum(axis=0) ** 2)
    # # intercept
    # a = (price_df_log.sum(axis=0) * (x ** 2).sum(axis=0) - x.sum(axis=0) * (x * price_df_log).sum(axis=0)) / d
    # # slope
    # b = (n * (x * price_df_log).sum(axis=0) - x.sum(axis=0) * price_df_log.sum(axis=0)) / d

    yhat = b * x + a
    yhat_real = torch.exp(a) * torch.exp(x * b)
    # plt.clf()
    # plt.plot(price_df_log.cpu().numpy()[:, 2000])
    # plt.plot(yhat.cpu().detach().numpy()[:, 2000])
    # plt.draw()
    # plt.pause(0.001)

    # plt.clf()
    # plt.plot(price_df.cpu().numpy()[:, 1000])
    # plt.plot(yhat_real.cpu().detach().numpy()[:, 1000])
    # plt.draw()
    # plt.pause(0.001)

    # plt.show()

    annualized_slope = (torch.pow(torch.exp(b), 252) - 1) * 100

    num = ((yhat - price_df_log_mean) ** 2).sum(axis=0)
    den = ((price_df_log - price_df_log_mean) ** 2).sum(axis=0)
    r_value = num / den

    score = annualized_slope * r_value

    i = tickers.index('ALGN')
    print('Mom vals:', score.cpu().numpy()[i], b.cpu().numpy()[i], a.cpu().numpy()[i], r_value.cpu().numpy()[i], annualized_slope.cpu().numpy()[i])
    return score.cpu().numpy()


def cap_score_calc():
    return (price_df[-rebalance_window:, :] * volume_df[-rebalance_window:, :]).sum(axis=0).cpu().numpy()


def vola_score_calc():
    pct_change = (price_df[-vola_window:, :] - price_df[-vola_window - 1:-1, :]) / price_df[-vola_window - 1:-1, :]
    std = torch.std(pct_change, 0)
    return std.cpu().numpy()
    # print(std)
    # print(std.shape)
    # print('asdasdasdasdasd\n')
    # float(prices.pct_change().rolling(vola_window).std().iloc[-1])


def update_history_and_coefs():
    momentum_scores = momentum_score_calc()
    cap_scores = cap_score_calc()
    vola_scores = vola_score_calc()
    return momentum_scores, cap_scores, vola_scores
    # print(vola_score)
    # print(vola_score.shape)
    # absent_tickers = set(history.keys()) - set(new_bars.keys())
    # for ticker in absent_tickers:
    #     del history[ticker]

    # for ticker in new_bars:
    #     if not (new_bars[ticker][0] == new_bars[ticker][1] == None):
    #         history[ticker][0].append((dt,) + new_bars[ticker])
    #         if len(history[ticker][0]) > max(dots_n, vola_window + 1):
    #             history[ticker][0].popleft()
    #         if len(history[ticker][0]) == max(dots_n, vola_window + 1) and day_cnt % rebalance_window == 0:
    #             history[ticker][1] = momentum_score([close for dt, close, vol in list(history[ticker][0])[-dots_n:]])
    #             history[ticker][2] = cap_score([close for dt, close, vol in list(history[ticker][0])[-rebalance_window:]],
    #                                            [vol for dt, close, vol in list(history[ticker][0])[-rebalance_window:]])
    #             history[ticker][3] = volatility_score(
    #                 [close for dt, close, vol in list(history[ticker][0])[-vola_window - 1:]])


# def momentum_score(df):
#     log_closes = np.log(closes)
#     slope, intercept, r_value, p_value, std_err = stats.linregress(np.arange(len(closes)), log_closes)
#     annualized_slope = (np.power(np.exp(slope), 252) - 1) * 100
#     score = annualized_slope * (r_value ** 2)
#     return score, slope, intercept, r_value, annualized_slope


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


def plot_figures():
    for ticker in tickers_for_plot:
        # print(ticker)
        fig, ax = figures[ticker]
        fig.suptitle(ticker, fontsize=16)
        ax[0].clear()
        plot_closes(ax[0], [dt for dt, close, vol in list(history[ticker][0])[-dots_n:]],
                    [close for dt, close, vol in list(history[ticker][0])[-dots_n:]],
                    handle_ylims=True)
        ax[1].clear()
        plot_closes(ax[1], [dt for dt, close, vol in list(history[ticker][0])[-dots_n:]],
                    [close for dt, close, vol in list(history[ticker][0])[-dots_n:]],
                    linestyle='None', marker='.', handle_ylims=True, n_lines=2)

        m_vals = history[ticker][1]
        c_val = history[ticker][2]

        if m_vals is not None:
            x = np.arange(min(dots_n, len(history[ticker][0])))
            days = [dt for dt, _, _ in list(history[ticker][0])]
            plot_closes(ax[1], days,
                        exp_function(m_vals[2], m_vals[1], x), color='red', n_lines=2)
            ax[1].set_title(f'Ann. Slope: {round(m_vals[4], 2)}% \n'
                            f'R2: {round(m_vals[3], 5)}')

        open_poses = [op for op in portfolio.open_positions if op.ticker == ticker]
        plot_open_poses(ax[0], open_poses)

        closed_poses = [cp for cp in portfolio.closed_positions if cp.ticker == ticker]
        plot_closed_poses(ax[0], closed_poses)

        ax[0].set_title(f'M: {round(m_vals[0], 2) if m_vals is not None else None} \n'
                        f'C: {round(c_val, 2) if c_val is not None else None}')
        plt.draw()
        plt.pause(0.001)


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
        pk.dump((unequal_dates_n, params, (dt_hist, mv_hist)), file, protocol=pk.HIGHEST_PROTOCOL)


def make_orders(scores):
    momentum_scores, cap_scores, vola_scores = scores

    dt = broker.dates[broker.current_dt_i]

    # spx_curr = spx.iloc[broker.current_dt_i, :]
    # assert dt == spx_curr.date

    spx_curr = spx[spx.date.isin(pd.date_range(datetime(dt.year, dt.month, dt.day, 0, 0, 0) - timedelta(5),
                                               datetime(dt.year, dt.month, dt.day, 23, 59, 59)))]
    spx_curr = spx_curr.iloc[-1]

    assert dt >= spx_curr.date

    # if dt != spx_curr.date:
    #     global unequal_dates_n
    #     unequal_dates_n += 1
    # if not np.isnan(spx_curr.ma) and spx_curr.ma < spx_curr.price:
    # buy_tickers = {'ticker': [], 'price': [], 'momentum': [], 'cap': [], 'volatility': []}
    # for ticker in history:
    #     ticker_hist = history[ticker]
    #     if ticker_hist[3] is not None and ticker_hist[3] != 0 and len(ticker_hist[0]) == max(dots_n, vola_window + 1):
    #         buy_tickers['ticker'].append(ticker)
    #         buy_tickers['price'].append(ticker_hist[0][-1][1])
    #         buy_tickers['momentum'].append(ticker_hist[1][0])
    #         buy_tickers['cap'].append(ticker_hist[2])
    #         buy_tickers['volatility'].append(ticker_hist[3])
    buy_tickers = {'ticker': tickers,
                   'price': price_df[-1, :].cpu().numpy(),
                   'momentum': momentum_scores,
                   'cap': cap_scores,
                   'volatility': vola_scores}

    buy_tickers = pd.DataFrame(buy_tickers)
    # print(buy_tickers)
    # print(buy_tickers)
    cap_thresh = buy_tickers.cap.quantile(cap_quantile)
    # print(cap_thresh)
    buy_tickers = buy_tickers[buy_tickers.cap > cap_thresh]
    # print(buy_tickers)

    buy_tickers.sort_values(by='momentum', inplace=True, ascending=False)
    buy_tickers = buy_tickers.iloc[:momentum_top_n]
    buy_tickers['inv_vola'] = 1 / buy_tickers.volatility
    inv_vola_sum = buy_tickers.inv_vola.sum()
    buy_tickers['inv_vola_norm'] = buy_tickers['inv_vola'] / inv_vola_sum
    market_val = 0.01 * (100 - com_reserve_perc) * portfolio.market_value
    print(portfolio.market_value)
    buy_tickers['how_many_to_buy'] = (market_val * buy_tickers.inv_vola_norm / buy_tickers.price).apply(np.floor)
    buy_tickers = buy_tickers[buy_tickers.how_many_to_buy > 0]
    # print(buy_tickers)
    buy_tickers = buy_tickers[buy_tickers.momentum > momentum_thresh]
    # print(buy_tickers)

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
            if not np.isnan(spx_curr.ma) and spx_curr.ma < spx_curr.price:
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
                if not np.isnan(spx_curr.ma) and spx_curr.ma < spx_curr.price:
                    if abs(change) > change_thresh:
                        diff_list.append((ticker, shares_diff))
                    else:
                        buy_tickers.loc[buy_tickers.iloc[i].name, 'how_many_to_buy'] = open_shares
                else:
                    diff_list.append((ticker, -open_shares))
                    rem_idx_buy_ticks.append(i)

        # Buy new or change > thresh
        # if (change is None and not np.isnan(spx_curr.ma) and spx_curr.ma < spx_curr.price) or \
        #    (change is not None and abs(change) > change_thresh):
        #     diff_list.append((ticker, shares_diff))
        # else:
        #     if change is not None and abs(change) <= change_thresh:
        #         buy_tickers.loc[buy_tickers.iloc[i].name, 'how_many_to_buy'] = open_shares
        #     elif change is None and not np.isnan(spx_curr.ma) and spx_curr.ma >= spx_curr.price:
        #         rem_idx_buy_ticks.append(i)
    buy_tickers.drop(buy_tickers.index[rem_idx_buy_ticks], inplace=True)
    print(buy_tickers[['ticker', 'price', 'momentum', 'volatility', 'inv_vola_norm', 'how_many_to_buy']])

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


def write_results():
    row = {}
    for k in params:
        row[k] = [params[k]]

    row['profit'] = [portfolio.market_value / init_cash]

    mv_series = pd.Series(mv_hist)
    r = mv_series.diff()
    sr = r.mean() / r.std() * np.sqrt(252)
    row['sharpe'] = [sr]

    spx_curr = spx.spx[spx.date.isin(pd.date_range(broker.first_dt, broker.last_dt))]
    spx_np = spx_curr.to_numpy()
    mv_np = np.array(mv_hist)

    error = mv_np.shape[0] - spx_np.shape[0]
    print(error)
    print(spx_np.shape[0], mv_np.shape[0])

    if abs(error) < 10:
        if error < 0:
            remove_idxs = random.sample(list(range(0, spx_np.shape[0])), error)
            spx_np = np.delete(spx_np, remove_idxs)
        elif error > 0:
            remove_idxs = random.sample(list(range(0, mv_np.shape[0])), error)
            print(remove_idxs)
            mv_np = np.delete(mv_np, remove_idxs)
    print(spx_np.shape[0], mv_np.shape[0])

    assert spx_np.shape[0] == mv_np.shape[0]

    cov = np.cov(mv_np, spx_np)[0, 1]
    var = np.var(spx_np)
    beta = cov / var
    row['beta'] = [beta]
    row['filename'] = [filename]
    try:
        dbm.insert_df_fast(pd.DataFrame(row), momentum_table)
    except:
        raise Exception(pd.DataFrame(row))
    dbm.commit()


spx = pd.read_csv('../../DataParsers/Bloomberg/Data/SPXT_290721.csv')
spx.columns = ['date', 'price']
spx['date'] = pd.to_datetime(spx.date, format='%d-%b-%y')
spx = pd.DataFrame({'dt': broker.dates}).merge(spx, how='left', left_on='dt', right_on='date')[['dt', 'price']]
spx['price'].fillna(method='ffill', inplace=True)
spx['ma'] = spx['price'].rolling(window=ma_window).mean()
spx.columns = ['date', 'price', 'ma']

# spx = pd.read_csv('spx index.csv')
# spx['date'] = pd.to_datetime(spx.date, format='%d.%m.%Y')
# # spx = spx[spx.date >= datetime(broker.current_dt.year, broker.current_dt.month, broker.current_dt.day)]
# spx['ma'] = spx['spx'].rolling(window=ma_window).mean()
# spx.columns = ['date', 'price', 'ma']
# print(spx)

print('Last date spx:', spx.iloc[-1])

history = defaultdict(lambda: [deque(), None, None, None])


fig, ax_mv = plt.subplots(1, 1, figsize=(12, 8))
mv_hist = []
dt_hist = []

tickers, price_df, volume_df = [], torch.Tensor(), torch.Tensor()


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
        day_cnt += 1
        start_time = time.time()
        print(day_cnt, broker.dates[broker.current_dt_i])
        print('Market value:', portfolio.market_value)
        mv_hist.append(portfolio.market_value)
        dt_hist.append(broker.dates[broker.current_dt_i])
        # broker.current_tickers = subset_tickers
        # broker.current_tickers = ['amzn']
        # broker.current_tickers = ['med']
        # broker.current_tickers = ['med', 'amd']
        global tickers, price_df, volume_df
        tickers, price_df, volume_df = broker.get_curr_bars()

        assert len(tickers) == price_df.shape[1] == volume_df.shape[1]

        scores = update_history_and_coefs()
        make_orders(scores)
        # print(len())

        # plot_figures()
        # plot_mv()
        save_mv()
        finished = broker.go_next_rth_day(portfolio)
        # day_cnt += 1
        spent_time = time.time() - start_time
        sum_day_runtime += spent_time
        print('Average runtime:', sum_day_runtime / day_cnt)
        print('----------------------------')
        print()


if __name__ == '__main__':
    import time
    import cProfile

    # with cProfile.Profile() as pr:
    #     start = time.time()
    # try:
    main()

        # params[args.param1_name] = args.param1_value
        # params[args.param2_name] = args.param2_value
    # except Exception as e:
    #     with open(f'Errors/{args.param1_value}|{args.param2_value}.txt', 'w') as file:
    #         file.write(repr(e))
    #         file.write(traceback.format_exc())

        # print('Spent time: ', time.time() - start)
    # write_results()
    # pr.print_stats()
    # print(filename)

    # with open('market_val200.pk', 'rb') as file:
    #     cc = pk.load(file)
    #
    # print(cc)
