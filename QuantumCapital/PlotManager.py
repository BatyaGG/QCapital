import sys
from collections import deque
from datetime import datetime

import pandas as pd

from matplotlib import pyplot as plt


def plot_candles(bars):
    def default_color(index, open_price, close_price):
        return 'r' if open_price[index] > close_price[index] else 'g'

    open_price = bars["open"]
    close_price = bars["close"]
    low = bars["low"]
    high = bars["high"]
    oc_min = pd.concat([open_price, close_price], axis=1).min(axis=1)
    oc_max = pd.concat([open_price, close_price], axis=1).max(axis=1)
    _, ax1 = plt.subplots(figsize=(16, 9))
    ax1.set_title(ticker)
    candle_colors = [default_color(i, open_price, close_price) for i in bars['count']]
    ax1.bar(bars['count'], oc_max - oc_min, bottom=oc_min, color=candle_colors, linewidth=0)
    ax1.vlines(bars['count'], low, high, color=candle_colors, linewidth=1)
    ax1.xaxis.grid(False)
    ax1.xaxis.set_tick_params(which='major', length=3.0, direction='in', top='off')
    ax1.grid(False)
    time_format = '%d-%m-%Y'
    plt.xticks(bars['count'],
               [pd.to_datetime(str(bar_date)).strftime(time_format) for bar_date in bars.index.values],
               rotation='vertical')
    ax1.grid(True)

    plt.xticks(bars['count'],
               [pd.to_datetime(str(bar_date)).strftime(time_format) for bar_date in bars.index.values],
               rotation='vertical')  # .to_datetime().strftime(time_format)


# def plot_closes(ax, dates, closes, linestyle='-', marker='', color='black', handle_ylims=False, n_lines=1):
#     xticks = ax.get_xticklabels()
#     x = list(range(0, len(closes)))
#     if len(xticks) > 50:
#         xticks = [tick.get_text() for tick in xticks]
#         if dates[0] <= datetime.strptime(xticks[0], '%Y-%m-%d').date():
#             dates = [datetime.strftime(dt, '%Y-%m-%d') for dt in dates]
#             i = dates.index(xticks[0])
#             print(i)
#             x = list(range(-i, len(closes) - i))
#             # print(x)
#         else:
#             raise NotImplementedError
#     ax.plot(x, closes, color=color, linestyle=linestyle, marker=marker)
#     ax.set_xticks(list(range(len(dates))))
#     ax.set_xticklabels(dates, rotation='vertical', ha='right')
#     if handle_ylims:
#         min_y = min(closes)
#         max_y = max(closes)
#         ax.set_ylim(min_y - 0.02*min_y, max_y + 0.02*max_y)


# def plot_closes(ax, dates, closes, linestyle='-', marker='', color='black', handle_ylims=False, n_lines=1):
#     xticks = ax.get_xticklabels()
#     x = list(range(0, len(closes)))
#     if len(xticks) > 50:
#         xticks = [tick.get_text() for tick in xticks]
#         if dates[0] <= datetime.strptime(xticks[0], '%Y-%m-%d').date():
#             dates = [datetime.strftime(dt, '%Y-%m-%d') for dt in dates]
#             i = dates.index(xticks[0])
#             print(i)
#             x = list(range(-i, len(closes) - i))
#             # print(x)
#         else:
#             raise NotImplementedError
#     ax.plot(x, closes, color=color, linestyle=linestyle, marker=marker)
#     ax.set_xticks(list(range(len(dates))))
#     ax.set_xticklabels(dates, rotation='vertical', ha='right')
#     if handle_ylims:
#         min_y = min(closes)
#         max_y = max(closes)
#         ax.set_ylim(min_y - 0.02*min_y, max_y + 0.02*max_y)


def plot_closes(ax, dates, closes, linestyle='-', marker='', color='black', handle_ylims=False, n_lines=1):
    # xticks = ax.get_xticklabels()
    x = list(range(0, len(closes)))
    ax.plot(x, closes, color=color, linestyle=linestyle, marker=marker)
    ax.set_xticks(list(range(len(dates)))[::1])
    ax.set_xticklabels(dates[::1], rotation='vertical', ha='right')
    if handle_ylims:
        min_y = min(closes)
        max_y = max(closes)
        ax.set_ylim(min_y - 0.02*min_y, max_y + 0.02*max_y)


# def plot_closes_old(ax, dates, closes, linestyle='-', marker='', color='black', handle_ylims=False, n_lines=1):
#     # xticks = ax.get_xticklabels()
#     # x = list(range(0, len(closes)))
#     # if len(xticks) > 50:
#     #     xticks = [tick.get_text() for tick in xticks]
#     #     if dates[0] < datetime.strptime(xticks[0], '%Y-%m-%d').date():
#     #         dates = [datetime.strftime(dt, '%Y-%m-%d') for dt in dates]
#     #         i = dates.index(xticks[0])
#     #         print(i)
#     #         x = list(range(-i, len(closes) - i))
#     #     else:
#     #         raise NotImplementedError
#     ax.plot(dates, closes, color=color, linestyle=linestyle, marker=marker)
#     ax.set_xticks(list(range(len(dates))))
#     ax.set_xticklabels(dates, rotation='vertical', ha='right')
#     if handle_ylims:
#         min_y = min(closes)
#         max_y = max(closes)
#         ax.set_ylim(min_y - 0.02*min_y, max_y + 0.02*max_y)

# def plot_regression(ax):
#     pass


def plot_open_poses(ax, open_poses):
    # get_xticklabels
    # closes = list(ax.lines[-1].get_ydata())
    # dates = [xtick.get_text() for xtick in ax.get_xticklabels()]
    for op in open_poses:
        assert op.exit_dt is None
        assert op.exit_price is None
        assert op.pnl is None
        dt_op = op.entry_dt.strftime(format='%Y-%m-%d')
        for i, dt in enumerate(ax.get_xticklabels()):
            dt_bar = dt.get_text()
            if dt_op == dt_bar:
                ax.plot(i, op.entry_price, '^', color='lime')


def plot_closed_poses(ax, closed_poses):
    if type(closed_poses) is not list:
        closed_poses = [closed_poses]
    # get_xticklabels
    # closes = list(ax.lines[-1].get_ydata())
    # dates = [xtick.get_text() for xtick in ax.get_xticklabels()]
    for cp in closed_poses:
        assert cp.exit_dt is not None
        assert cp.exit_price is not None
        assert cp.pnl is not None
        # dt_op = cp.entry_dt
        # dt_cp = cp.exit_dt

        dt_op = cp.entry_dt.strftime(format='%Y-%m-%d')
        dt_cp = cp.exit_dt.strftime(format='%Y-%m-%d')
        x_op = None
        x_cp = None
        for i, dt in enumerate(ax.get_xticklabels()[:-1]):
            # idx = dt.get_position()[0]
            # dt_bar = datetime.strptime(dt.get_text(), format='%Y-%m-%d')
            # dt_nextbar = datetime.strptime(ax.get_xticklabels()[i + 1].get_text(), format='%Y-%m-%d')

            idx = i
            dt_bar = dt.get_text()

            # if dt_op.isin(pd.date_range(dt_bar, dt_nextbar)):
            if dt_op == dt_bar:
                x_op = idx
                ax.plot(idx, cp.entry_price, '^' if cp.amount > 0 else 'v', color='lime')
            # if dt_cp.isin(pd.date_range(dt_bar, dt_nextbar)):
            if dt_cp == dt_bar:
                x_cp = idx
                ax.plot(idx, cp.exit_price, 'v' if cp.amount > 0 else '^', color='red')
        if cp.pnl > 0:
            # print([x_op, x_cp], [cp.entry_price, cp.exit_price])
            ax.plot([x_op, x_cp], [cp.entry_price, cp.exit_price], 'g')
        else:
            ax.plot([x_op, x_cp], [cp.entry_price, cp.exit_price], color='red')


def plot_stops(ax, closed_poses):
    if type(closed_poses) is not list:
        closed_poses = [closed_poses]
    # get_xticklabels
    # closes = list(ax.lines[-1].get_ydata())
    # dates = [xtick.get_text() for xtick in ax.get_xticklabels()]
    for cp in closed_poses:
        assert cp.exit_dt is not None
        assert cp.exit_price is not None
        assert cp.pnl is not None
        # dt_op = cp.entry_dt
        # dt_cp = cp.exit_dt

        dt_op = cp.entry_dt.strftime(format='%Y-%m-%d')
        dt_cp = cp.exit_dt.strftime(format='%Y-%m-%d')
        x_op = None
        x_cp = None

        # plot_enabled = False
        stops = cp.ts_hist.copy()
        stop_dates = [stop[0].strftime(format='%Y-%m-%d') for stop in stops]

        for i, dt in enumerate(ax.get_xticklabels()[:-1]):
            # idx = dt.get_position()[0]
            # dt_bar = datetime.strptime(dt.get_text(), format='%Y-%m-%d')
            # dt_nextbar = datetime.strptime(ax.get_xticklabels()[i + 1].get_text(), format='%Y-%m-%d')

            idx = i
            dt_bar = dt.get_text()

            if dt_bar not in stop_dates:
                continue

            stop_idx = stop_dates.index(dt_bar)

            ax.plot(idx, stops[stop_idx][1], '.', color='r')

            # if dt_op.isin(pd.date_range(dt_bar, dt_nextbar)):
            # if dt_op == dt_bar:
            #     plot_enabled = True
                # x_op = idx
                # ax.plot(idx, cp.entry_price, '^', color='lime')
            # if dt_cp.isin(pd.date_range(dt_bar, dt_nextbar)):

            # if plot_enabled and len(stops) > 1:
            #     ax.plot(idx, stops.popleft()[0], '.', color='r')
            #
            # if dt_cp == dt_bar:
            #     plot_enabled = False
                # x_cp = idx
                # ax.plot(idx, cp.exit_price, 'v', color='red')

        # if cp.pnl > 0:
        #     ax.plot([x_op, x_cp], [cp.entry_price, cp.exit_price], 'g')
        # else:
        #     ax.plot([x_op, x_cp], [cp.entry_price, cp.exit_price], 'r')
