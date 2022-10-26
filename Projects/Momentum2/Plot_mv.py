import pickle as pk

import matplotlib.pyplot as plt
import pandas as pd

from QuantumCapital.PlotManager import *

if __name__ == '__main__':
    # with open('Data/market_val_2021-08-02 17:39:20.395645.pk', 'rb') as file:
    with open('../Momentum/Data/market_val', 'rb') as file:

        market_val = pk.load(file)
    # print(market_val[1])
    df = market_val[1]
    df = pd.DataFrame({'date': df[0], 'mv': df[1]})
    print('# of days:', df.shape[0])
    # offset = 10
    # market_val = (market_val[0][offset:], [1e6 * mv / market_val[1][offset] for mv in market_val[1][offset:]])
    print(f"""
                First date: {df.iloc[0]['date']}
                Last date: {df.iloc[-1]['date']}
    """)

    spx = pd.read_csv('spx_index_new.csv')
    spx['date'] = pd.to_datetime(spx.date, format='%d-%b-%y', errors='coerce')
    # print(spx)

    ndx = pd.read_csv('ndx index.csv')
    ndx['date'] = pd.to_datetime(ndx.date, format='%d.%m.%Y', errors='coerce')
    ndx.columns = ['date', 'ndx']
    # print(ndx.date)

    rus = pd.read_csv('rus_index.csv')
    rus['date'] = pd.to_datetime(rus.date, format='%d-%b-%y', errors='coerce')
    # print(rus.date)

    df = df.merge(spx, how='left', left_on='date', right_on='date')
    df = df.merge(ndx, how='left', left_on='date', right_on='date')
    df = df.merge(rus, how='left', left_on='date', right_on='date')
    df['spx'] = (df.spx / df.spx.iloc[0]) * 1e6
    df['ndx'] = (df.ndx / df.ndx.iloc[0]) * 1e6
    df['rus'] = (df.rus / df.rus.iloc[0]) * 1e6
    df['spx_ma'] = spx['spx'].rolling(window=200).mean()
#
#     step = 14
#
#     for i in range(step, min(spx.shape[0], len(market_val[1]))):
#         max_spx = spx.spx.iloc[i-step:i].max()
#         max_ndx = ndx.spx.iloc[i-step:i].max()
#         max_rus = rus.rus.iloc[i-step:i].max()
#         max_mv = max(market_val[1][i-step:i])
#
#         min_spx = spx.spx.iloc[i:i+step].min()
#         min_ndx = ndx.spx.iloc[i:i+step].min()
#         min_rus = rus.rus.iloc[i:i+step].min()
#         min_mv = min(market_val[1][i:i+step])
#
#         max_dd_spx = max(max_dd_spx, ((max_spx - min_spx) / max_spx))
#         max_dd_ndx = max(max_dd_ndx, ((max_ndx - min_ndx) / max_ndx))
#         max_dd_rus = max(max_dd_rus, ((max_rus - min_rus) / max_rus))
#         max_dd_mv = max(max_dd_mv, ((max_mv - min_mv) / max_mv))
#
#     print(f"""
#             Max DD SPX: {max_dd_spx}
#             Max DD RUS: {max_dd_rus}
#             Max DD MOM: {max_dd_mv}
# """)
#     ndx_profit = (ndx.spx.iloc[-1] / ndx.spx.iloc[0]) ** (1/18)
#     spx_profit = (spx.spx.iloc[-1] / spx.spx.iloc[0]) ** (1/18)
#     rus_profit = (rus.rus.iloc[-1] / rus.rus.iloc[0]) ** (1/18)
#     mom_profit = (market_val[1][-1] / market_val[1][0]) ** (1/18)
#     print(f"""
#                 Profit SPX: {spx_profit}
#                 Profit RUS: {rus_profit}
#                 Profit MOM: {mom_profit}
#     """)
    _, ax1 = plt.subplots(figsize=(16, 9))
    plot_closes(ax1, df.date, df.mv, color='blue')
    plot_closes(ax1, df.date, df.spx, color='black')
    plot_closes(ax1, df.date, df.ndx, color='orange')
    plot_closes(ax1, df.date, df.rus, color='red', linestyle='-')
    # plot_closes(ax1, rus.date, rus.rus, color='orange')
    #
    # mv = pd.DataFrame({'Date': market_val[0], 'Momentum': market_val[1]})
    # mv.to_csv('momentum_market_val.csv')
    # spx.to_csv('spy_market_val.csv')
    # ndx.to_csv('ndx_market_val.csv')
    plt.show()
