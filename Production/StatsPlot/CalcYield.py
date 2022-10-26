import pandas as pd

from ib_insync import Stock, IB, util

import sys
from datetime import timedelta, date

from matplotlib import pyplot as plt
import matplotlib.ticker as ticker

sys.path.append('../..')
from DataParsers.DataEngine import DataEngine
from Backtester.BrokerGPU2 import Broker
# from Backtester.Broker import Broker
from Backtester.Order import Order
from Backtester.Portfolio import Portfolio


def create_mv_df():
    alloc_prc = 10

    orders = pd.read_csv('ml_quarter (2).csv')
    orders['Odate'] = pd.to_datetime(orders.Odate, format='%m/%d/%y')
    orders['Cdate'] = pd.to_datetime(orders.Cdate, format='%m/%d/%y')
    orders['SPY_bought'] = None
    orders['asset_bought'] = None

    # data_parser = DataEngine(client_id=10)
    # date_range = min(orders['Odate'].min(), orders['Cdate'].min()), max(orders['Odate'].max(), orders['Cdate'].max())
    # all_tickers = list(set(orders.Ticker.tolist()))
    #
    # data = data_parser.get_bars('SPY', date_range[0], date_range[1], False)
    # data.index = data['dt']
    # data = data[['close']]
    # data.columns = ['SPY_price']
    #
    # for ticker in all_tickers:
    #     df_ticker = data_parser.get_bars(ticker, date_range[0].date(), date_range[1].date(), True)
    #     if df_ticker is None:
    #         ticker_in_orders = orders[orders.Ticker == ticker]
    #         assert ticker_in_orders.shape[0] == 1
    #         df_ticker = data_parser.get_bars(ticker, date_range[0].date(), ticker_in_orders.iloc[0].Cdate.date(), True)
    #         print(df_ticker)
    #     df_ticker = df_ticker[['dt', 'close']]
    #     df_ticker['close'].ffill(inplace=True)
    #     df_ticker.columns = ['dt', f'{ticker}_price']
    #     data = data.merge(df_ticker, how='left', left_index=True, right_on='dt')
    #     data.set_index('dt', drop=True, inplace=True)
    # dates = pd.to_datetime(data.index - timedelta(1)).tolist()
    # dates = pd.to_datetime([date(d.year, d.month, d.day) for d in dates])
    # data.index = dates
    #
    # data.to_csv('temp_data.csv', index=True)
    data = pd.read_csv('temp_data.csv', index_col=0)
    data.index = pd.to_datetime(data.index)
    broker = Broker(data, 1,
                    step=1,
                    min_commission=0,
                    commission_rate=0,
                    slippage=False,
                    reindex=False,
                    return_vols=False)
    str_portfolio = Portfolio(1e6, data.index[0])
    spy_portfolio = Portfolio(1e6, data.index[0])

    dts_mv_hist = []
    str_mv_hist = []
    spy_mv_hist = []

    finished = False

    while not finished:
        # print(orders)
        dt = broker.dates[broker.current_dt_i]
        print(dt)
        tickers, price_df, _ = broker.get_curr_bars()
        spy_idx = tickers.index('SPY')
        spy_price = price_df.cpu().numpy()[0, spy_idx]

        buy_orders = orders[orders.Odate == dt]
        sell_orders = orders[orders.Cdate == dt]

        for i in range(buy_orders.shape[0]):
            buy_order = buy_orders.iloc[i]
            print(buy_order.Ticker)
            # order_alloc_prc = alloc_prc * buy_order.Alloc * buy_order.Weighted * 0.01
            order_alloc_prc = alloc_prc * buy_order.Weighted * 0.01
            order_alloc_cash = order_alloc_prc * str_portfolio.market_value
            order_alloc_amnt = int(order_alloc_cash / buy_order.Oprice)

            orders.loc[buy_order.name, 'asset_bought'] = order_alloc_amnt

            broker.make_order(str_portfolio,
                              Order(buy_order.Ticker, kind='market', dir='buy', amount=order_alloc_amnt, what='shares'),
                              set_price=buy_order.Oprice)

            order_alloc_cash = order_alloc_prc * spy_portfolio.market_value
            order_alloc_amnt = int(order_alloc_cash / spy_price)
            broker.make_order(spy_portfolio,
                              Order('SPY', kind='market', dir='buy', amount=order_alloc_amnt, what='shares'))
            # print(buy_order.name)
            orders.loc[buy_order.name, 'SPY_bought'] = order_alloc_amnt

        for i in range(sell_orders.shape[0]):
            sell_order = sell_orders.iloc[i]
            print(sell_order.Ticker)
            amount = int(sell_order.asset_bought)
            broker.make_order(str_portfolio,
                              Order(sell_order.Ticker, kind='market', dir='sell', amount=amount, what='shares'),
                              set_price=sell_order.Cprice)

            amount = int(sell_order.SPY_bought)
            broker.make_order(spy_portfolio,
                              Order('SPY', kind='market', dir='sell', amount=amount, what='shares'))

        dts_mv_hist.append(dt)
        str_mv_hist.append(str_portfolio.market_value)
        spy_mv_hist.append(spy_portfolio.market_value)

        finished, _ = broker.go_next_rth_day([str_portfolio, spy_portfolio])
    df = pd.DataFrame({'dt': dts_mv_hist, 'strategy_mv': str_mv_hist, 'spy_mv': spy_mv_hist})
    print(df)
    df.to_csv('mv_comparison.csv', index=False)


def plot_mv():
    plt.rcParams.update({'font.size': 15})

    df = pd.read_csv('mv_comparison.csv')
    fig, ax = plt.subplots(1, 1)
    ax.plot(df['dt'], df['strategy_mv'] / 1e4, label='Машинное обучение', color='darkblue')
    ax.plot(df['dt'], df['spy_mv'] / 1e4, label='Аналогичные позиции в SPY', color='black')
    ax.legend()

    plt.xticks(rotation=45)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(21))
    # fig.autofmt_xdate()
    plt.show()


if __name__ == '__main__':
    # create_mv_df()
    plot_mv()
