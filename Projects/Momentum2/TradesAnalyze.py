import pickle as pk
from datetime import timedelta

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from StrategyClass import Strategy
from QuantumCapital.PlotManager import plot_closes, plot_closed_poses, plot_stops

mode = 'run'  # save | run
plot_option = 'pnl_loss'  # pnl_gain | pnl_loss

delta_day = 200
change_perc = 0.01
# ---------------------------------------------------------------------------------- ##

assert mode in ('save', 'run')

if mode == 'save':
    strategy = Strategy(use_trailing_stop=True, precise_stops=True)
    strategy.params['trailing_stop_window'] = 180
    strategy.params['trailing_stop_coeff'] = 4
    strategy.init_broker()

    strategy.run(plot_mv=True)
    # with open('temp_strategy.pk', 'wb') as file:
    #     pk.dump(strategy, file, protocol=pk.HIGHEST_PROTOCOL)
else:
    with open('temp_strategy.pk', 'rb') as file:
        strategy = pk.load(file)
data = strategy.broker.data

assert plot_option in ('pnl_gain', 'pnl_loss')

res = strategy.total_results().iloc[0]
print(res)

strategy.plot_mv()

trades = strategy.positions
ptrades = trades[trades.PNL >= 0]
ntrades = trades[trades.PNL < 0]

print(trades)

print(trades.PNL.sum())

print(trades['MV %'].describe())
print(ptrades['MV %'].describe())
print(ntrades['MV %'].describe())

# sns.histplot(trades, x='MV %')
# sns.histplot(ptrades, x='MV %')
# sns.histplot(ntrades, x='MV %')
# plt.show()

stop_close_diffs = []

counter = 0
for i, pos in enumerate(strategy.portfolio.closed_positions[::-1]):
    if (plot_option == 'pnl_gain' and pos.pnl <= 0) or (plot_option == 'pnl_loss' and pos.pnl > 0):
        continue
    mv = strategy.market_val_table[strategy.market_val_table.date < pos.entry_dt].iloc[-1]['market_val']
    if abs(pos.pnl / mv) * 100 < change_perc:
        continue
    if not pos.stop_triggered:
        continue
    data_pos = data[data.index.isin(pd.date_range(pos.entry_dt - timedelta(delta_day),
                                                  pos.exit_dt + timedelta(delta_day)))][pos.ticker + '_price']
    stop_bar_close = data[data.index.isin(pd.date_range(pos.exit_dt,
                                                        pos.exit_dt))][pos.ticker + '_price'].values[0]
    stop_close_diff = (pos.ts - stop_bar_close) / pos.ts
    stop_close_diffs.append(100 * stop_close_diff)
    counter += 1
    print(abs(pos.pnl / mv) * 100, '%')
    print(counter)

# sns.distplot(stop_close_diffs)
# plt.show()

    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.set_title(pos.ticker)
    plot_closes(ax, data_pos.index.date, data_pos)
    plot_closed_poses(ax, pos)
    plot_stops(ax, pos)
    plt.show()
    # fig.savefig(f"../../PresentationPlots/{'Momentum_NTrades' if plot_option == 'pnl_loss' else 'Momentum_PTrades'}/{i}.png", dpi=fig.dpi)
