from datetime import date

import pickle as pk

import matplotlib.pyplot as plt
import pandas as pd

from QuantumCapital.PlotManager import *


class Portfolio:
    def __init__(self, cash: float, first_dt: date):
        self._cash = cash
        self.cash_hist = [[first_dt, cash]]
        self._market_value = cash
        self.mv_hist = [[first_dt, cash]]
        self.open_positions = []
        self.closed_positions = []

    @property
    def cash(self):
        return self._cash

    @cash.setter
    def cash(self, val):
        self._cash = val[1]
        self.cash_hist.append(val)

    @property
    def market_value(self):
        return self._market_value

    @market_value.setter
    def market_value(self, val):
        self._market_value = val[1]
        self.mv_hist.append(val)

    def number_of_shares_open(self, ticker):
        n_shares = 0
        for op in self.open_positions:
            if op.ticker == ticker:
                n_shares += op.amount
        return int(n_shares)

    def open_tickers(self):
        tickers = set()
        for op in self.open_positions:
            tickers.add(op.ticker)
        return tickers

    def closed_tickers(self):
        tickers = set()
        for cp in self.closed_positions:
            tickers.add(cp.ticker)
        return tickers

    def assert_orders_onesided(self):
        ops = {'ticker': [], 'amount': []}
        for op in self.open_positions:
            ops['ticker'].append(op.ticker)
            ops['amount'].append(op.amount)
        ops = pd.DataFrame(ops)
        ops.sort_values(by=['ticker'], inplace=True)
        for ticker in set(ops.ticker.tolist()):
            ticker_ops = ops[ops.ticker == ticker]
            side = set()
            for j in range(ticker_ops.shape[0]):
                ticker_op = ticker_ops.iloc[j]
                dir = 'long' if ticker_op.amount > 0 else 'short'
                side.add(dir)
            if len(side) != 1:
                return False
        return True

    def get_open_tickers_df(self):
        open_tickers = self.open_tickers()
        df = {'ticker': [], 'amount': []}
        for ticker in open_tickers:
            df['ticker'].append(ticker)
            df['amount'].append(self.number_of_shares_open(ticker))
        df = pd.DataFrame(df)
        df.sort_values(by='amount', inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df


if __name__ == '__main__':
    with open('Data/market_val_2021-06-30 11:54:54.560678.pk', 'rb') as file:
        market_val = pk.load(file)
    print(market_val[1])
    market_val = market_val[2]
    print('# of days:', len(market_val[0]))
    offset = 10
    market_val = (market_val[0][offset:], [1e6 * mv / market_val[1][offset] for mv in market_val[1][offset:]])
    print(f"""
    First date: {market_val[0][0]}
    Last date: {market_val[0][-1]}
    """)

    spx = pd.read_csv('spx_index_new.csv')
    spx['date'] = pd.to_datetime(spx.date, format='%d-%b-%y', errors='coerce')
    print(spx.date)

    ndx = pd.read_csv('ndx index.csv')
    ndx['date'] = pd.to_datetime(ndx.date, format='%d.%m.%Y', errors='coerce')
    print(ndx.date)

    rus = pd.read_csv('rus_index.csv')
    rus['date'] = pd.to_datetime(rus.date, format='%d-%b-%y', errors='coerce')
    print(rus.date)

    spx_first_dt = spx.iloc[0].date
    ndx_first_dt = ndx.iloc[0].date
    rus_first_dt = rus.iloc[0].date
    mvv_first_dt = market_val[0][0]

    spx_last_dt = spx.iloc[-1].date
    ndx_last_dt = ndx.iloc[-1].date
    rus_last_dt = rus.iloc[-1].date
    mvv_last_dt = market_val[0][-1]

    first_dt = max(spx_first_dt, mvv_first_dt, mvv_first_dt)
    last_dt = min(spx_last_dt, rus_last_dt, mvv_last_dt)

    print(first_dt)

    spx = spx[spx.date >= datetime(first_dt.year, first_dt.month, first_dt.day)]
    ndx = ndx[ndx.date >= datetime(first_dt.year, first_dt.month, first_dt.day)]
    rus = rus[rus.date >= datetime(first_dt.year, first_dt.month, first_dt.day)]

    spx = spx[spx.date <= datetime(last_dt.year, last_dt.month, last_dt.day)]
    ndx = ndx[ndx.date >= datetime(first_dt.year, first_dt.month, first_dt.day)]
    rus = rus[rus.date <= datetime(last_dt.year, last_dt.month, last_dt.day)]

    # # ndx = ndx.iloc[0:5189]
    # print('MV len', len(market_val[1]))
    # rus = rus.iloc[0:4570]
    # spx = spx.iloc[0:4570]

    spx['spx'] = 1e6 * (spx.spx / spx.spx.iloc[0])
    ndx['spx'] = 1e6 * (ndx.spx / ndx.spx.iloc[0])
    rus['rus'] = 1e6 * (rus.rus / rus.rus.iloc[0])
    spx['ma'] = spx['spx'].rolling(window=200).mean()


    max_dd_spx = 0
    max_dd_ndx = 0
    max_dd_rus = 0
    max_dd_mv = 0

    step = 14

    for i in range(step, min(spx.shape[0], len(market_val[1]))):
        max_spx = spx.spx.iloc[i-step:i].max()
        max_ndx = ndx.spx.iloc[i-step:i].max()
        max_rus = rus.rus.iloc[i-step:i].max()
        max_mv = max(market_val[1][i-step:i])

        min_spx = spx.spx.iloc[i:i+step].min()
        min_ndx = ndx.spx.iloc[i:i+step].min()
        min_rus = rus.rus.iloc[i:i+step].min()
        min_mv = min(market_val[1][i:i+step])

        max_dd_spx = max(max_dd_spx, ((max_spx - min_spx) / max_spx))
        max_dd_ndx = max(max_dd_ndx, ((max_ndx - min_ndx) / max_ndx))
        max_dd_rus = max(max_dd_rus, ((max_rus - min_rus) / max_rus))
        max_dd_mv = max(max_dd_mv, ((max_mv - min_mv) / max_mv))

    print(f"""
            Max DD SPX: {max_dd_spx}
            Max DD RUS: {max_dd_rus}
            Max DD MOM: {max_dd_mv}
""")
    ndx_profit = (ndx.spx.iloc[-1] / ndx.spx.iloc[0]) ** (1/18)
    spx_profit = (spx.spx.iloc[-1] / spx.spx.iloc[0]) ** (1/18)
    rus_profit = (rus.rus.iloc[-1] / rus.rus.iloc[0]) ** (1/18)
    mom_profit = (market_val[1][-1] / market_val[1][0]) ** (1/18)
    print(f"""
                Profit SPX: {spx_profit}
                Profit RUS: {rus_profit}
                Profit MOM: {mom_profit}
    """)
    _, ax1 = plt.subplots(figsize=(16, 9))
    plot_closes(ax1, market_val[0], market_val[1], color='blue')
    plot_closes(ax1, spx.date, spx.spx, color='black')
    plot_closes(ax1, ndx.date, ndx.spx, color='orange')
    plot_closes(ax1, spx.date, spx.ma, color='red', linestyle='-')
    # plot_closes(ax1, rus.date, rus.rus, color='orange')
    #
    # mv = pd.DataFrame({'Date': market_val[0], 'Momentum': market_val[1]})
    # mv.to_csv('momentum_market_val.csv')
    # spx.to_csv('spy_market_val.csv')
    # ndx.to_csv('ndx_market_val.csv')
    plt.show()
