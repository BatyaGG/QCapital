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
from datetime import timedelta

from Backtester.BrokerGPU2 import Broker
from Backtester.Portfolio import Portfolio
from Backtester.Order import Order

import config
from QuantumCapital import constants
from QuantumCapital.PlotManager import *


import warnings
warnings.filterwarnings('ignore')

pd.set_option("display.max_rows", None)


class Strategy:
    def __init__(self,
                 k=0,
                 c=0,
                 top_n=1,
                 take=0.1,
                 stop=0.1,
                 cap_quantile_gt=0.25,
                 cap_window=5,
                 vola_window=20,
                 init_cash=1e6,
                 write_log=True,
                 write_mv=False):

        self.params = {'k': k,
                       'c': c,
                       'top_n': top_n,
                       'take': take,
                       'stop': stop,
                       'cap_quantile_gt': cap_quantile_gt,
                       'cap_window': cap_window,
                       'vola_window': vola_window}

        self.dt_now = datetime.now().strftime('%Y-%m-%d||%H:%M:%S')
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
        self.exempt_tickers = []

        self.ns_tickers_in_portfolio = []

    def init_broker(self):
        print(self.params)
        df = pd.read_parquet('../../Models_&_Files/Bloom_Files/rus_aligned_df_140721_280721.parquet')
        df = df[df.index.isin(pd.date_range(datetime(2002, 1, 1), datetime(2003, 8, 9)))]
        self.broker = Broker(df,
            max(self.params['k'] + 2, self.params['vola_window'] + 1, self.params['cap_window'] + 1),
            logger=self.logger if self.write_log else None)
        self.portfolio = Portfolio(self.init_cash, self.broker.dates[self.broker.current_dt_i])

        spx = pd.read_csv('../../DataParsers/Bloomberg/Data/SPXT_290721.csv')
        spx.columns = ['date', 'price']
        spx['date'] = pd.to_datetime(spx['date'], format='%d-%b-%y')
        self.spx = pd.DataFrame({'dt': self.broker.dates}).merge(
            spx, how='left', left_on='dt', right_on='date')[['dt', 'price']]
        self.spx['price'].fillna(method='ffill', inplace=True)
        self.spx['ma'] = self.spx['price'].rolling(window=200).mean()
        self.spx.columns = ['date', 'price', 'ma']
        # pd.set_option('display.max_rows', None)
        # print(self.spx)

    def _cap_score_calc(self, window, k):
        assert k >= 0
        if k == 0:
            cap_score = (self.price_df[-window:, :] * self.volume_df[-window:, :]).sum(axis=0)
        else:
            cap_score = (self.price_df[-window - k:-k, :] * self.volume_df[-window - k:-k, :]).sum(axis=0).cpu().numpy()
        cap_mean = torch.mean(cap_score)
        cap_std = torch.std(cap_score)
        cap_score_dev = (cap_score - cap_mean) / cap_std
        return cap_score.cpu().numpy(), cap_score_dev.cpu().numpy()

    def _vola_score_calc(self, window, k):
        assert k >= 0
        if k == 0:
            pct_change = (self.price_df[-window:, :] - self.price_df[-window - 1:-1, :]) / self.price_df[-window - 1:-1, :]
        else:
            pct_change = (self.price_df[-window - k:-k, :] - self.price_df[-window - 1 - k:-1 - k, :]) / self.price_df[-window - 1 - k:-1 - k, :]
        vola_score = torch.std(pct_change, 0)
        vola_mean = torch.mean(vola_score)
        vola_std = torch.std(vola_score)
        vola_score_dev = (vola_score - vola_mean) / vola_std
        return vola_score.cpu().numpy(), vola_score_dev.cpu().numpy()

    def _diff_score_calc(self, k):
        pct_change = (self.price_df[-k - 1, :] - self.price_df[-k - 2, :]) / self.price_df[-k - 2, :]
        pct_mean = torch.mean(pct_change)
        pct_std = torch.std(pct_change)
        pct_change_dev = (pct_change - pct_mean) / pct_std
        return pct_change.cpu().numpy(), pct_change_dev.cpu().numpy()

    def _update_history_and_coefs(self):
        diff_scores, diff_scores_dev = self._diff_score_calc(self.params['k'])
        cap_scores, cap_scores_dev = self._cap_score_calc(self.params['cap_window'], 0)
        vola_scores, vola_scores_dev = self._vola_score_calc(self.params['vola_window'], 0)
        return diff_scores, diff_scores_dev, cap_scores, cap_scores_dev, vola_scores, vola_scores_dev

    def make_orders(self, scores):
        pd.set_option('display.max_columns', None)
        pd.set_option('display.expand_frame_repr', False)

        dt = self.broker.dates[self.broker.current_dt_i]
        # print(dt)
        diff_scores, diff_scores_dev, cap_scores, cap_scores_dev, vola_scores, vola_scores_dev = scores

        buy_tickers = {'ticker': self.tickers,
                       'price': self.price_df[-1, :].cpu().numpy(),
                       'diff_score': diff_scores,
                       'diff_score_dev': diff_scores_dev,
                       'cap': cap_scores,
                       'cap_dev': cap_scores_dev,
                       'vola': vola_scores,
                       'vola_dev': vola_scores_dev}

        buy_tickers = pd.DataFrame(buy_tickers)
        buy_tickers = buy_tickers[~buy_tickers.ticker.isin(self.exempt_tickers)]
        cap_thresh = buy_tickers.cap.quantile(self.params['cap_quantile_gt'])
        buy_tickers = buy_tickers[(buy_tickers.cap > cap_thresh) | (buy_tickers.ticker.isin(list(self.portfolio.open_tickers())))]

        buy_tickers.sort_values(by='diff_score', inplace=True, ascending=True)
        sell_tickers = buy_tickers.iloc[-self.params['top_n']:].sort_values(by='diff_score', inplace=False, ascending=False)
        buy_tickers = buy_tickers.iloc[:self.params['top_n']]

        buy_tickers.reset_index(drop=True, inplace=True)
        sell_tickers.reset_index(drop=True, inplace=True)

        buy_tickers['inv_vola'] = 1 / buy_tickers.vola
        inv_vola_sum = buy_tickers.inv_vola.sum()
        buy_tickers['inv_vola_norm'] = buy_tickers['inv_vola'] / inv_vola_sum
        sell_tickers['inv_vola'] = 1 / sell_tickers.vola
        inv_vola_sum = sell_tickers.inv_vola.sum()
        sell_tickers['inv_vola_norm'] = sell_tickers['inv_vola'] / inv_vola_sum

        buy_tickers['how_many_to_buy'] = ((self.portfolio.market_value / 2) * buy_tickers.inv_vola_norm / buy_tickers.price).apply(np.floor)
        sell_tickers['how_many_to_sell'] = ((self.portfolio.market_value / 2) * sell_tickers.inv_vola_norm / sell_tickers.price).apply(np.floor)
        buy_tickers = buy_tickers[buy_tickers.how_many_to_buy > 0]
        sell_tickers = sell_tickers[sell_tickers.how_many_to_sell > 0]

        print(buy_tickers)

        assert self.portfolio.assert_orders_onesided()
        # open_positions = self.portfolio.get_open_tickers_df()

        # if self.broker.current_dt_i % 10 == 0:
        #     # print(open_positions)
        #     # print(len(self.portfolio.open_positions))
        #     while len(self.portfolio.open_positions) > 0:
        #         op = self.portfolio.open_positions[0]
        #     # for op in self.portfolio.open_positions:
        #     #     print(op.ticker)
        #         if op.amount > 0:
        #             self.broker.make_order(self.portfolio, Order(op.ticker, 'market', 'sell', op.amount, 'shares'))
        #         else:
        #             pass
        #             self.broker.make_order(self.portfolio, Order(op.ticker, 'market', 'cover', op.amount, 'shares'))
        #
        #
        # # if self.portfolio.cash == self.portfolio.market_value:
        # if self.broker.current_dt_i % 10 == 0:
        #     print(self.broker.current_dt_i)
        #     print(open_positions)
        #     print(len(self.portfolio.open_positions))
        #     print('Mv:', self.portfolio.market_value)
        #     print('Cash:', self.portfolio.cash)
        #
        #     assert round(self.portfolio.market_value, 6) == round(self.portfolio.cash, 6)
        #     # print(open_orders)
        #     print('----------------------------------------')

        # print(buy_tickers)
        # print(sell_tickers)

        # buy_tickers_names = set(buy_tickers.ticker.values.tolist())
        # open_tickers_names = self.portfolio.open_tickers()
        # close_tickers_names = list(open_tickers_names - buy_tickers_names)
        #
        # for ticker in close_tickers_names:
        #     amount = self.portfolio.number_of_shares_open(ticker)
        #     if amount > 0:
        #         self.broker.make_order(self.portfolio, Order(ticker, 'market', 'sell', amount, 'shares'))
        #     elif amount < 0:
        #         self.broker.make_order(self.portfolio, Order(ticker, 'market', 'cover', amount, 'shares'))
        #     else:
        #         raise Exception

        buy_tickers['how_many_pre'] = buy_tickers['how_many_to_buy']

        for op in self.portfolio.open_positions[::-1]:
            if dt - op.entry_dt > timedelta(self.params['c']):
                self.broker.make_order(self.portfolio, Order(op.ticker, 'market', 'sell' if op.amount > 0 else 'cover',
                                                             op.amount, 'shares'))

        open_positions = self.portfolio.get_open_tickers_df()

        for i in range(buy_tickers.shape[0]):
            buy_ticker = buy_tickers.iloc[i]
            # if buy_ticker.ticker == 'WEBX':
            #     print(buy_ticker.ticker in open_positions.ticker.tolist())
            if buy_ticker.ticker in open_positions.ticker.tolist() and \
                open_positions[open_positions.ticker == buy_ticker.ticker].iloc[0].amount < 0:
                self.broker.make_order(self.portfolio,
                                       Order(buy_ticker.ticker, 'market', 'cover',
                                             int(open_positions[open_positions.ticker == buy_ticker.ticker].amount), 'shares'))
                # self.broker.make_order(self.portfolio,
                #                        Order(buy_ticker.ticker, 'market', 'sell',
                #                              int(open_positions[open_positions.ticker == buy_ticker.ticker].amount),
                #                              'shares'))
            else:
                self.broker.make_order(self.portfolio,
                                       Order(buy_ticker.ticker, 'market', 'buy',
                                             int(buy_ticker.how_many_to_buy), 'shares', info=buy_ticker))
                # self.broker.make_order(self.portfolio,
                #                        Order(buy_ticker.ticker, 'market', 'short', -int(buy_ticker.how_many_to_buy),
                #                              'shares'))
        for i in range(sell_tickers.shape[0]):
            sell_ticker = sell_tickers.iloc[i]
            if sell_ticker.ticker in open_positions.ticker.tolist() and \
                open_positions[open_positions.ticker == sell_ticker.ticker].iloc[0].amount > 0:
                self.broker.make_order(self.portfolio,
                                       Order(sell_ticker.ticker, 'market', 'sell',
                                             int(open_positions[open_positions.ticker == sell_ticker.ticker].amount),
                                             'shares'))
                # self.broker.make_order(self.portfolio,
                #                        Order(sell_ticker.ticker, 'market', 'cover',
                #                              int(open_positions[open_positions.ticker == sell_ticker.ticker].amount),
                #                              'shares'))
            else:
                self.broker.make_order(self.portfolio,
                                       Order(sell_ticker.ticker, 'market', 'short',
                                             -int(sell_ticker.how_many_to_sell), 'shares', info=sell_ticker))
                # self.broker.make_order(self.portfolio,
                #                        Order(sell_ticker.ticker, 'market', 'buy',
                #                              int(sell_ticker.how_many_to_sell), 'shares'))
        self.ns_tickers_in_portfolio.append(len(self.portfolio.open_positions))

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
        # plt.plot(self.dt_hist, spx_hist)
        # plt.plot(self.dt_hist, spx_ma_hist)
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
                     'MV %': [],
                     'diff_score': [],
                     'diff_score_dev': [],
                     'cap': [],
                     'cap_dev': [],
                     'vola': [],
                     'vola_dev': [],
                     'inv_vola': [],
                     'inv_vola_norm': []}

        for pos in self.portfolio.closed_positions:
            positions['Ticker'].append(pos.ticker)
            positions['Entry DT'].append(pos.entry_dt)
            positions['Exit DT'].append(pos.exit_dt)
            positions['Entry Price'].append(pos.entry_price)
            positions['Exit Price'].append(pos.exit_price)
            positions['Amount'].append(pos.amount)
            positions['PNL'].append(pos.pnl)
            positions['Trade %'].append(100 * ((pos.exit_price - pos.entry_price) / pos.entry_price) * np.copysign(1, pos.amount))
            mv = self.market_val_table[self.market_val_table.date <= pos.entry_dt].iloc[-1]['market_val']
            positions['MV %'].append(100 * (pos.pnl / mv))
            positions['diff_score'].append(pos.info.diff_score)
            positions['diff_score_dev'].append(pos.info.diff_score_dev)
            positions['cap'].append(pos.info.cap)
            positions['cap_dev'].append(pos.info.cap_dev)
            positions['vola'].append(pos.info.vola)
            positions['vola_dev'].append(pos.info.vola_dev)
            positions['inv_vola'].append(pos.info.inv_vola)
            positions['inv_vola_norm'].append(pos.info.inv_vola_norm)

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

            # print(self.short_n_longs())

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

    def short_n_longs(self):
        longs_n = shorts_n = 0
        for pos in self.portfolio.open_positions:
            if pos.amount > 0:
                longs_n += 1
            elif pos.amount < 0:
                shorts_n += 1
            else:
                raise Exception('0 amount order')
        return longs_n, shorts_n


if __name__ == '__main__':
    # data = pd.read_parquet('../../Models_&_Files/Bloom_Files/rus_aligned_df_140721_280721.parquet')
    # print(data.index)

    s = Strategy()
    s.init_broker()
    s.run(plot_mv=True)
