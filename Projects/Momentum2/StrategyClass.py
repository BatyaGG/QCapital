import sys
import pickle as pk
import logging
from importlib import reload
logging.shutdown()
reload(logging)
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import linregress

from Backtester.BrokerGPU2 import Broker
from Backtester.Portfolio import Portfolio
from Backtester.Order import Order

import config
from QuantumCapital import constants
# from QuantumCapital.DBManager import DBManager
from QuantumCapital.PlotManager import *


import warnings
warnings.filterwarnings('ignore')


# def __init__(self,
#              dots_n=105,
#              step=5,
#              vola_window=160,
#              cap_quantile_gt=0.25,
#              momentum_top_n=20,
#              momentum_threshold=40,
#              com_reserve_perc=0,
#              ma_window=200,
#              change_thresh=0.0,
#              close_all_at_low_ma=False,
#              use_trailing_stop=False,
#              precise_stops=False,
#              trailing_stop_window=None,
#              trailing_stop_coeff=None,
#              init_cash=1e6,
#              write_log=False,
#              write_mv=False):


class Strategy:
    def __init__(self,
                 dots_n=125,
                 step=5,
                 vola_window=20,
                 cap_quantile_gt=0.25,
                 momentum_top_n=20,
                 momentum_threshold=40,
                 com_reserve_perc=0,
                 ma_window=200,
                 change_thresh=0.7,
                 close_all_at_low_ma=False,
                 use_trailing_stop=False,
                 precise_stops=False,
                 trailing_stop_window=None,
                 trailing_stop_coeff=None,
                 init_cash=1e6,
                 write_log=False,
                 write_mv=False):

        self.dt_now = datetime.now().strftime('%Y-%m-%d||%H:%M:%S')
        self.params = {'dots_n': dots_n,
                       'step': step,
                       'vola_window': vola_window,
                       'cap_quantile_gt': cap_quantile_gt,
                       'momentum_top_n': momentum_top_n,
                       'momentum_threshold': momentum_threshold,
                       'com_reserve_perc': com_reserve_perc,
                       'ma_window': ma_window,
                       'change_thresh': change_thresh,
                       'close_all_at_low_ma': close_all_at_low_ma,
                       'use_trailing_stop': use_trailing_stop,
                       'trailing_stop_window': trailing_stop_window,
                       'trailing_stop_coeff': trailing_stop_coeff,
                       'precise_stops': precise_stops}
        self.market_val_file = f'Data/market_vals_{self.dt_now}.pk'
        self.logs_file = f'Logs/{self.dt_now}.log'

        if not torch.cuda.is_available():
            raise Exception('No GPU')

        self.write_log = write_log
        self.write_mv = write_mv

        if write_log:
            # Configuring logging rules -------------------------------------------------------------- /
            a = logging.Logger.manager.loggerDict  # Disabling other loggers
            for k in a:
                a[k].disabled = True

            for handler in logging.root.handlers[:]:  # Disabling root logger handlers
                logging.root.removeHandler(handler)

            logging.basicConfig(
                filename=self.logs_file,
                format='%(asctime)s %(levelname)s [%(funcName)s]: %(message)s',
                datefmt='%Y-%m-%d | %H:%M:%S',
                level=logging.ERROR)

            # self.logger = logging.getLogger(__name__)
            self.logger = logging.getLogger(self.logs_file)
            self.logger.setLevel(logging.DEBUG)

            requests_logger = logging.getLogger('requests')
            requests_logger.setLevel(logging.ERROR)

            handler = logging.StreamHandler()
            handler.setLevel(logging.ERROR)
            self.logger.addHandler(handler)
            requests_logger.addHandler(handler)

            # logging.getLogger().setLevel(logging.DEBUG)  # If you want logger write in console
            self.logger.info(
                '\n\n_________________________________________________________________________________________________')

        self.init_cash = init_cash

        self.gpu = torch.device("cuda:0")

        self.dt_hist = []
        self.mv_hist = []
        self.cash_hist = []
        self.spx_hist = []
        self.spx_ma_hist = []
        self.mv_change_idx = None
        self.market_val_table = None
        self.stop_tickers = set()
        self.tickers = self.price_df = self.volume_df = None
        # self.exempt_tickers = ['GME', 'AMC', 'IWV']
        self.exempt_tickers = []

        self.ns_tickers_in_portfolio = []

    def init_broker(self):
        df = pd.read_parquet('../../Models_&_Files/Bloom_Files/rus_aligned_df_140721_280721.parquet')
        # df = df[df.index.isin(pd.date_range(datetime(2005, 1, 1), datetime(2021, 8, 9)))]
        # iwv = pd.read_csv('../../Models_&_Files/Indexes/IWV.csv')
        # iwv['Date'] = pd.to_datetime(iwv.Date, format='%d-%b-%y', errors='coerce')
        # iwv.set_index('Date', drop=True, inplace=True)
        # iwv = df.iloc[:, [1]].merge(iwv, how='left', left_index=True, right_index=True)[['IWV']]
        # iwv['IWV'].fillna(method='ffill', inplace=True)
        # # pd.set_option('display.max_rows', None)
        # df['IWV_price'] = iwv
        # df['IWV_rep_date'] = False
        # df['IWV_volume'] = 0
        self.broker = Broker(df,
                             max(self.params['dots_n'], self.params['vola_window'] + 1,
                self.params['trailing_stop_window'] + 1 if self.params['use_trailing_stop'] else 0),
                             step=self.params['step'],
                             logger=self.logger if self.write_log else None, check_stops=self.params['use_trailing_stop'],
                             precise_take_stop=self.params['precise_stops'])
        self.portfolio = Portfolio(self.init_cash, self.broker.dates[self.broker.current_dt_i])

        spx = pd.read_csv('../../DataParsers/Bloomberg/Data/SPXT_290721.csv')
        # spx = pd.read_csv('spx index.csv')
        spx.columns = ['date', 'price']
        # spx['date'] = spx['date'].astype(str)
        spx['date'] = pd.to_datetime(spx['date'], format='%d-%b-%y')
        # print(spx.date)
        # spx['date'] = pd.to_datetime(spx['date'], format='%d.%m.%Y')
        spx['price'].fillna(method='ffill', inplace=True)
        spx['ma'] = spx['price'].rolling(window=self.params['ma_window']).mean()
        self.spx = pd.DataFrame({'dt': self.broker.dates}).merge(
            spx, how='left', left_on='dt', right_on='date')[['dt', 'price', 'ma']]
        self.spx['price'].fillna(method='ffill', inplace=True)
        self.spx['ma'] = self.spx['price'].rolling(window=self.params['ma_window']).mean()
        self.spx.columns = ['date', 'price', 'ma']
        # pd.set_option('display.max_rows', None)
        # print(self.spx)

    def _momentum_score_calc(self):
        x = torch.arange(self.params['dots_n']).view(self.params['dots_n'], 1).repeat(1, len(self.tickers)).to(self.gpu)
        # yhat = torch.randn(df.shape[1], requires_grad=True, dtype=torch.float, device=gpu)

        price_df_log = torch.log(self.price_df[-self.params['dots_n']:, :])

        # print(x)
        #
        x_mean = (x * 1.0).mean(axis=0)
        x_minus_xmean = x - x_mean
        price_df_log_mean = price_df_log.mean(axis=0)
        price_df_log_minus_mean = price_df_log - price_df_log_mean
        num = (x_minus_xmean * price_df_log_minus_mean).sum(axis=0)
        den = (x_minus_xmean ** 2).sum(axis=0)
        b = num / den
        a = price_df_log_mean - b * x_mean

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
        # score = annualized_slope

        # i = tickers.index('ALGN')
        # print('Mom vals:', score.cpu().numpy()[i], b.cpu().numpy()[i], a.cpu().numpy()[i], r_value.cpu().numpy()[i],
        #       annualized_slope.cpu().numpy()[i])
        return score.cpu().numpy()

    def _cap_score_calc(self):
        return (self.price_df[-self.params['step']:, :] * self.volume_df[-self.params['step']:, :]
                ).sum(axis=0).cpu().numpy()

    def _vola_score_calc(self, window):
        pct_change = (self.price_df[-window:, :] - self.price_df[-window - 1:-1, :]) / self.price_df[-window - 1:-1, :]
        std = torch.std(pct_change, 0)
        return std.cpu().numpy()

    def _update_history_and_coefs(self):
        momentum_scores = self._momentum_score_calc()
        cap_scores = self._cap_score_calc()
        vola_scores_alloc = self._vola_score_calc(self.params['vola_window'])
        if self.params['use_trailing_stop']:
            vola_scores_trlstp = self._vola_score_calc(self.params['trailing_stop_window'])
        else:
            vola_scores_trlstp = None
        return momentum_scores, cap_scores, vola_scores_alloc, vola_scores_trlstp

    def make_orders(self, scores):
        # pd.set_option('display.max_rows', None)

        momentum_scores, cap_scores, vola_scores_alloc, vola_scores_trlstp = scores

        dt = self.broker.dates[self.broker.current_dt_i]

        spx_curr = self.spx.iloc[self.broker.current_dt_i, :]
        assert dt == spx_curr.date

        buy_tickers = {'ticker': self.tickers,
                       'price': self.price_df[-1, :].cpu().numpy(),
                       'momentum': momentum_scores,
                       'cap': cap_scores,
                       'volatility': vola_scores_alloc,
                       'volatility_trlstp': vola_scores_trlstp}

        buy_tickers = pd.DataFrame(buy_tickers)
        # print(buy_tickers.shape)
        # print(buy_tickers[buy_tickers.ticker.isin(self.exempt_tickers)])
        buy_tickers = buy_tickers[~buy_tickers.ticker.isin(self.exempt_tickers)]
        # print('-----+++++-----')
        cap_thresh = buy_tickers.cap.quantile(self.params['cap_quantile_gt'])
        buy_tickers = buy_tickers[buy_tickers.cap > cap_thresh]

        buy_tickers.sort_values(by='momentum', inplace=True, ascending=False)
        buy_tickers = buy_tickers.iloc[:self.params['momentum_top_n'] + len(self.stop_tickers)]
        buy_tickers.reset_index(drop=True, inplace=True)

        # top_half = set(buy_tickers.ticker.iloc[:self.params['momentum_top_n']].tolist())
        # print(len(self.stop_tickers))
        # print(buy_tickers)

        # -------------------------------------------------------------------------------------
        # new_stop_tickers = set()

        # while True:
        #     # print(buy_tickers)
        #     top_half = buy_tickers.iloc[:self.params['momentum_top_n']]
        #     bot_half = buy_tickers.iloc[self.params['momentum_top_n']:]
        #
        #     to_rmv_ticks = sum(top_half.ticker.isin(self.stop_tickers))
        #     if to_rmv_ticks == 0:
        #         buy_tickers = top_half
        #         self.stop_tickers = new_stop_tickers
        #         break
        #     new_stop_tickers.update(set(top_half.ticker[top_half.ticker.isin(self.stop_tickers)].tolist()))
        #     buy_tickers = pd.concat((top_half[~top_half.ticker.isin(self.stop_tickers)], bot_half), axis=0)
        # -------------------------------------------------------------------------------------

        buy_tickers.reset_index(drop=True, inplace=True)
        # print(buy_tickers)
        # print('-----------------------------------------------------\n')
        buy_tickers = buy_tickers.iloc[:self.params['momentum_top_n']]
        buy_tickers['inv_vola'] = 1 / buy_tickers.volatility
        inv_vola_sum = buy_tickers.inv_vola.sum()
        buy_tickers['inv_vola_norm'] = buy_tickers['inv_vola'] / inv_vola_sum
        market_val = 0.01 * (100 - self.params['com_reserve_perc']) * self.portfolio.market_value
        buy_tickers['how_many_to_buy'] = (market_val * buy_tickers.inv_vola_norm / buy_tickers.price).apply(np.floor)
        buy_tickers = buy_tickers[buy_tickers.how_many_to_buy > 0]
        buy_tickers = buy_tickers[buy_tickers.momentum > self.params['momentum_threshold']]

        # buy_tickers.sort_values(by='how_many_to_buy', inplace=True)

        buy_tickers_names = set(buy_tickers.ticker.values.tolist())
        open_tickers_names = self.portfolio.open_tickers()
        close_tickers_names = list(open_tickers_names - buy_tickers_names)

        buy_tickers['how_many_pre'] = buy_tickers['how_many_to_buy']

        # -------------------------------------------------------------------------------------
        # print(buy_tickers[['ticker', 'price', 'momentum', 'volatility', 'how_many_to_buy', 'how_many_pre']], '\n')
        # print(self.stop_tickers)
        buy_tickers = buy_tickers.iloc[:self.params['momentum_top_n']]
        self.stop_tickers = set(buy_tickers.ticker[buy_tickers.ticker.isin(self.stop_tickers)].tolist())
        buy_tickers = buy_tickers[~buy_tickers.ticker.isin(self.stop_tickers)]

        self.ns_tickers_in_portfolio.append(buy_tickers.shape[0])
        # -------------------------------------------------------------------------------------

        for ticker in close_tickers_names:
            amount = self.portfolio.number_of_shares_open(ticker)
            self.broker.make_order(self.portfolio, Order(ticker, 'market', 'sell', amount, 'shares'))

        rem_idx_buy_ticks = []
        diff_list = []
        for i in range(buy_tickers.shape[0]):
            row = buy_tickers.iloc[i]
            ticker = row.ticker
            to_open = int(row.how_many_to_buy)
            assert to_open > 0
            open_shares = self.portfolio.number_of_shares_open(ticker)
            assert open_shares >= 0
            shares_diff = to_open - open_shares
            change = shares_diff / open_shares if open_shares != 0 else None

            if change is None:  # new ticker came
                if not np.isnan(spx_curr.ma) and spx_curr.ma < spx_curr.price:
                    diff_list.append((ticker, shares_diff))
                else:
                    rem_idx_buy_ticks.append(i)
            else:
                if not self.params['close_all_at_low_ma']:
                    if abs(change) > self.params['change_thresh']:
                        diff_list.append((ticker, shares_diff))
                    else:
                        buy_tickers.loc[buy_tickers.iloc[i].name, 'how_many_to_buy'] = open_shares
                else:
                    if not np.isnan(spx_curr.ma) and spx_curr.ma < spx_curr.price:
                        if abs(change) > self.params['change_thresh']:
                            diff_list.append((ticker, shares_diff))
                        else:
                            buy_tickers.loc[buy_tickers.iloc[i].name, 'how_many_to_buy'] = open_shares
                    else:
                        diff_list.append((ticker, -open_shares))
                        rem_idx_buy_ticks.append(i)
        buy_tickers.drop(buy_tickers.index[rem_idx_buy_ticks], inplace=True)
        buy_tickers.reset_index(drop=True, inplace=True)

        # print(buy_tickers[['ticker', 'price', 'momentum', 'volatility', 'how_many_to_buy', 'how_many_pre']], '\n')
        # print('-----------------------------------------------------------------\n')

        if self.write_log:
            self.logger.info(buy_tickers[['ticker', 'price', 'momentum', 'volatility', 'inv_vola_norm',
                                          'how_many_to_buy', 'volatility_trlstp']])

        diff_list.sort(key=lambda x: x[1])
        for i, (ticker, diff) in enumerate(diff_list):
            if diff < 0:
                self.broker.make_order(self.portfolio, Order(ticker, 'market', 'sell', abs(diff), 'shares'))
            elif diff > 0:
                self.broker.make_order(self.portfolio, Order(ticker, 'market', 'buy', diff, 'shares'))
            else:
                raise Exception
        bought_tickers = self.portfolio.get_open_tickers_df()
        assert buy_tickers.shape[0] == bought_tickers.shape[0]

        if buy_tickers.shape[0] != 0:
            merged = pd.merge(buy_tickers[['ticker', 'how_many_to_buy']], bought_tickers, on='ticker')
            error = merged['how_many_to_buy'] - merged['amount']
            error = float(error.sum())
        else:
            error = 0
        assert error == 0

        # setting trailing stops
        if self.params['use_trailing_stop']:
            for i in range(buy_tickers.shape[0]):
                row = buy_tickers.iloc[i, :]
                ticker = row.ticker
                ts_offset_pct = self.params['trailing_stop_coeff'] * row.volatility_trlstp
                ts_price = row.price - ts_offset_pct * row.price
                self.broker.set_stop(self.portfolio, ticker, float(ts_price))

        # mv_long = self.broker.get_long_positions_market_val(self.portfolio)
        # cash_for_short = self.portfolio.market_value - mv_long
        # iwv_position = self.portfolio.number_of_shares_open('IWV')
        # iwv_idx = self.tickers.index('IWV')
        # iwv_price = float(self.price_df[-1, iwv_idx].cpu().numpy())
        # iwv_amount = -int(cash_for_short / iwv_price)
        # iwv_amnt_diff = int(iwv_position - iwv_amount)
        # print(iwv_amnt_diff)
        # if iwv_amnt_diff < 0:
        #     # cover
        #     print('cover')
        #     self.broker.make_order(self.portfolio, Order('IWV', 'market', 'cover', -abs(iwv_amnt_diff), 'shares'))
        # elif iwv_amnt_diff > 0:
        #     print('short')
        #     self.broker.make_order(self.portfolio, Order('IWV', 'market', 'short', -abs(iwv_amnt_diff), 'shares'))
        # print(iwv_amount, self.portfolio.number_of_shares_open('IWV'))

    def plot_mv(self):
        plt.clf()
        # i = 0
        # mv_i = self.mv_hist[i]
        # while mv_i == self.init_cash:
        #     i += 1
        #     mv_i = self.mv_hist[i]
        if self.mv_hist[-1] != self.init_cash and self.mv_change_idx is None:
            self.mv_change_idx = len(self.mv_hist) - 2

        spx_hist = pd.Series(self.spx_hist)
        scale_ratio = self.init_cash / (spx_hist.iloc[0] if self.mv_change_idx is None
                                        else spx_hist.iloc[self.mv_change_idx])
        spx_hist *= scale_ratio
        # spx_hist /= spx_hist.iloc[0] if self.mv_change_idx is None else spx_hist.iloc[self.mv_change_idx]
        # spx_hist *= self.init_cash
        spx_ma_hist = pd.Series(self.spx_ma_hist)
        spx_ma_hist *= scale_ratio
        # spx_ma_hist = spx_hist.rolling(self.params['ma_window'] // self.params['step']).mean()

        # spx_ma_hist = pd.Series(self.spx_ma_hist)
        # spx_ma_hist /= spx_hist.iloc[0] if self.mv_change_idx is None else spx_hist.iloc[self.mv_change_idx]
        # spx_ma_hist *= self.init_cash

        plt.plot(self.dt_hist, self.mv_hist)
        plt.plot(self.dt_hist, self.cash_hist)
        plt.plot(self.dt_hist, spx_hist)
        plt.plot(self.dt_hist, spx_ma_hist)
        plt.draw()
        plt.pause(0.001)

    def _save_mv(self):
        df = pd.DataFrame({'date': self.dt_hist,
                           'market_val': self.mv_hist,
                           'spxt_price': self.spx_hist,
                           'spxt_ma': self.spx_ma_hist})
        self.market_val_table = df
        if self.write_mv:
            with open(self.market_val_file, 'wb') as file:
                pk.dump((self.params, df), file, protocol=pk.HIGHEST_PROTOCOL)

    def _make_positions_df(self):
        positions = {'Ticker': [],
                     'Entry DT': [],
                     'Exit DT': [],
                     'Entry Price': [],
                     'Exit Price': [],
                     'Amount': [],
                     'PNL': [],
                     'Trade %': [],
                     'MV %': []}

        for pos in self.portfolio.closed_positions:
            positions['Ticker'].append(pos.ticker)
            positions['Entry DT'].append(pos.entry_dt)
            positions['Exit DT'].append(pos.exit_dt)
            positions['Entry Price'].append(pos.entry_price)
            positions['Exit Price'].append(pos.exit_price)
            positions['Amount'].append(pos.amount)
            positions['PNL'].append(pos.pnl)
            positions['Trade %'].append(100 * ((pos.exit_price - pos.entry_price) / pos.entry_price))
            mv = self.market_val_table[self.market_val_table.date < pos.entry_dt].iloc[-1]['market_val']
            positions['MV %'].append(100 * (pos.pnl / mv))

        self.positions = pd.DataFrame(positions)

    def total_results(self):
        row = {}
        for k in self.params:
            row[k] = [self.params[k]]

        row['profit'] = [self.portfolio.market_value / self.init_cash]

        mv_series = pd.Series(self.mv_hist)
        r = mv_series.diff()
        sr = r.mean() / r.std() * np.sqrt(252)
        row['sharpe'] = [sr]

        spx_returns = pd.Series(self.spx_hist).pct_change().iloc[1:]
        mv_returns = pd.Series(self.mv_hist).pct_change().iloc[1:]

        beta, _, _, _, _ = linregress(spx_returns, mv_returns)
        row['beta'] = [beta]
        # row['dt_key'] = [self.dt_now]

        max_dd = ((self.market_val_table['market_val'].cummax() - self.market_val_table['market_val']) / self.market_val_table['market_val'].cummax()).max()

        row['max_dd'] = [100 * max_dd]
        row['profit_per_trade_perc'] = [self.positions['MV %'].sum() / self.positions.shape[0]]
        ptrades = self.positions[self.positions.PNL >= 0]
        ntrades = self.positions[self.positions.PNL < 0]
        row['profit_per_ptrade_perc'] = [ptrades['MV %'].sum() / ptrades.shape[0]]
        row['profit_per_ntrade_perc'] = [ntrades['MV %'].sum() / ntrades.shape[0]]

        row['avg_n_top_tickers'] = sum(self.ns_tickers_in_portfolio) / len(self.ns_tickers_in_portfolio)

        return pd.DataFrame(row)

    def run(self, plot_mv=False):
        if self.write_log:
            self.logger.info(self.params)

        finished = False

        while not finished:
            dt = self.broker.dates[self.broker.current_dt_i]

            self.dt_hist.append(dt)
            self.mv_hist.append(self.portfolio.market_value)
            self.cash_hist.append(self.portfolio.cash)
            spx_curr = self.spx.iloc[self.broker.current_dt_i, :]
            self.spx_hist.append(spx_curr.price)
            self.spx_ma_hist.append(spx_curr.ma)

            if self.write_log:
                self.logger.info(self.broker.dates[self.broker.current_dt_i].date())
                self.logger.info(f'Market value: {self.portfolio.market_value}')
                self.logger.info(f'SPX: {spx_curr.price}, MA: {spx_curr.ma}')

            self.tickers, self.price_df, self.volume_df = self.broker.get_curr_bars()

            # print('IWV' in self.tickers)

            assert len(self.tickers) == self.price_df.shape[1] == self.volume_df.shape[1]

            scores = self._update_history_and_coefs()
            self.make_orders(scores)
            finished, stop_tickers = self.broker.go_next_rth_day([self.portfolio])
            if plot_mv:
                self.plot_mv()
            self.stop_tickers.update(stop_tickers)
        self._save_mv()
        self._make_positions_df()
        plt.close()
        # logging.shutdown()
        # del logging
        return self.total_results()


if __name__ == '__main__':
    # data = pd.read_parquet('../../Models_&_Files/Bloom_Files/rus_aligned_df_140721_280721.parquet')
    # print(data.index)

    s = Strategy(write_log=False, write_mv=False,
                 vola_window=160,
                 dots_n=105,
                 use_trailing_stop=True,
                 trailing_stop_window=60,
                 trailing_stop_coeff=3,
                 momentum_top_n=20,
                 step=5)
    s.init_broker()
    s.run(plot_mv=True)
