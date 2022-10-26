from matplotlib import pyplot as plt
import pandas as pd
import matplotlib.cm as cmx
import matplotlib.colors as colors


class Plot(object):

    def __init__(self, stock):
        self.stock = stock
        print(stock.weights_min_max)

    def plot_candles(self, show_levels=False, show_trend=False, show_vp=False, plot_macd=False, show_extremes=True,
                     show_bar_weights='', level_threshold=0.75):
        def default_color(index, open_price, close_price):
            return 'r' if open_price[index] > close_price[index] else 'g'

        open_price = self.stock.bars["open"]
        close_price = self.stock.bars["close"]
        low = self.stock.bars["low"]
        high = self.stock.bars["high"]
        oc_min = pd.concat([open_price, close_price], axis=1).min(axis=1)
        oc_max = pd.concat([open_price, close_price], axis=1).max(axis=1)
        _, ax1 = plt.subplots(figsize=(16, 9))
        ax1.set_title(self.stock.ticker)
        candle_colors = [default_color(i, open_price, close_price) for i in self.stock.bars['count']]
        ax1.bar(self.stock.bars['count'], oc_max - oc_min, bottom=oc_min, color=candle_colors, linewidth=0)
        ax1.vlines(self.stock.bars['count'], low, high, color=candle_colors, linewidth=1)
        ax1.xaxis.grid(False)
        ax1.xaxis.set_tick_params(which='major', length=3.0, direction='in', top='off')
        ax1.grid(False)
        time_format = '%d-%m-%Y'
        plt.xticks(self.stock.bars['count'],
                   [pd.to_datetime(str(bar_date)).strftime(time_format) for bar_date in self.stock.bars.index.values],
                   rotation='vertical')
        ax1.grid(True)

        # Plot macd
        if plot_macd:
            macd_line = self.stock.bars['macd_line']
            signal_line = self.stock.bars['signal_line']
            macd_hist = self.stock.bars['macd_hist']

            _, ax2 = plt.subplots(figsize=(16, 9))
            xticksdates = [pd.to_datetime(str(bar_date)).strftime(time_format) for bar_date in
                           self.stock.bars.index.values]
            ax2.plot(xticksdates, macd_line, color='green', lw=1, label='MACD Line(26,12)')
            ax2.plot(xticksdates, signal_line, color='purple', lw=1, label='Signal Line(9)')

            # set other parameters
            ax2.fill_between(xticksdates, macd_hist, color='gray', alpha=0.5, label='MACD Histogram')
            ax2.set(title='MACD(26,12,9)', ylabel='MACD')
            ax2.legend(loc='upper right')
            ax2.grid(True)

        # print(self.stock.extremes_low)
        if show_extremes:
            for ext_h in self.stock.extremes_high:
                ax1.plot([ext_h["count"] - 2, ext_h["count"] + 2], [ext_h["high"], ext_h["high"]], 'g--')
            for ext_l in self.stock.extremes_low:
                ax1.plot([ext_l["count"] - 2, ext_l["count"] + 2], [ext_l["low"], ext_l["low"]], 'r--')

        plt.xticks(self.stock.bars['count'],
                   [pd.to_datetime(str(bar_date)).strftime(time_format) for bar_date in self.stock.bars.index.values],
                   rotation='vertical')  # .to_datetime().strftime(time_format)

        if show_levels:
            for level in self.stock.levels:
                if level.weight <= level_threshold * self.stock.weights_min_max['level_weight'][1]: continue
                # PLASMA Color map:
                scalarMap = cmx.ScalarMappable(norm=colors.Normalize(vmin=self.stock.weights_min_max['level_weight'][0],
                                                                     vmax=self.stock.weights_min_max['level_weight'][
                                                                         1]),
                                               cmap=plt.get_cmap("cool"))
                ax1.plot(level.bars['count'],
                         [level.price for _ in range(level.bars.shape[0])],
                         '--',
                         color=scalarMap.to_rgba(level.weight))

                # ax1.text(level.bars['count'][0], level.price - level.price * 0.005, str(round(level.weight, 2)))

        if show_bar_weights is not '':
            count_to_weight = {k: [] for k in self.stock.bars['count']}
            for level in self.stock.levels:
                if level.weight <= level_threshold * self.stock.weights_min_max['level_weight'][1]: continue
                weights = level.get_bars_weight(show_bar_weights)
                for _, bar in weights.iterrows():
                    count_to_weight[bar['count']].append((bar['weight'], bar['touch']))
            for key, value in count_to_weight.items():
                for weight in value:
                    bar = self.stock.bars.loc[self.stock.bars['count'] == key]
                    if abs(weight[0]) > -1:
                        ax1.text(key, bar[weight[1]], str(round(weight[0], 3)))

        if show_trend:
            ax1.plot(self.stock.bars['count'][1:], self.stock.bars['super_trend'][1:])

        if show_vp:
            if self.stock.resample:
                self.stock.vp.calculate(self.stock.bars_unsampled, self.stock.start_date)
            else:
                self.stock.vp.calculate(self.stock.bars, self.stock.start_date)

            self.stock.vp.plot(ax1)
        plt.show()

