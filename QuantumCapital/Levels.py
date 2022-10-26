import pandas as pd
import numpy as np


class Level(object):
    # bars - list of tuples, where key is Bar object and value is touch type ('open', 'close', 'high', 'low')
    # score - confidence level
    def __init__(self, columns, price):
        self.price = price
        self.weight = 0.0
        self.weight_coeffs = {"fluctuating_weight": 3, "limit_weight": 1, "tail_weight": 1, "gap_weight": 2,
                              "paranormal_weight": 1, "trend_weight": 3, "quantity_weight": 1}
        self.bars = pd.DataFrame(columns=columns + ["touch"] + list(self.weight_coeffs.keys()))
        self.bars.set_index("date", inplace=True)
        self.is_active = False

    def add_bar(self, bar, df):
        bar = bar.assign(**{}.fromkeys(list(self.weight_coeffs.keys()), 0.0))
        self.bars = self.bars.append(bar)
        self._update_last_bar_weights(df)
        # with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        #     print(self.bars)
        #     print()
        if self.bars.shape[0] > 1:
            self.is_active = True
        return self.bars.iloc[-1][list(self.weight_coeffs.keys())]

    def get_bars_weight(self, weight_name):
        # print(self.bars[[weight_name, 'count', 'touch']])
        # print('-')
        # print(self.bars[[weight_name, 'count', 'touch']]. \
        #     drop_duplicates(subset=[weight_name, 'count']))
        # print()
        return self.bars[[weight_name, 'count', 'touch']]. \
            rename(columns={weight_name: 'weight'})


    def increase_trend_weight(self, weight, bars, touch):
        condition = [x and y for x, y in
                     zip(self.bars.index.isin(bars.index), list(self.bars["touch"] == touch))]
        self.bars.loc[condition, 'trend_weight'] = weight

    '''
        Calculate levels weights
    '''

    def _update_last_bar_weights(self, df):
        self._update_quantity_weight()
        self._last_bar_tail_longness()
        self._last_bar_paranormality()
        self._last_bar_gap(df)
        self._last_bar_fluctuating(df)
        self._check_limit()

        # date, weight = self._last_bar_fluctuating()
        # self.set_bar_weight(date, 'fluctuating_weight', weight)
        #
        # self.update_weight()

    def _norm_weight(self, value, min, max):
        normalized = (value - min) / (max - min) if (value - min) / (max - min) > 0.0 and not \
            np.isinf((value - min) / (max - min)) and not np.isnan((value - min) / (max - min)) else 0.0
        normalized = 1.0 if normalized > 1 else normalized
        return normalized

    def calculate_level_weight(self, norm_weights):
        flat_limits = [item for sublist in list(norm_weights.values()) for item in sublist]
        if np.inf in flat_limits or -np.inf in flat_limits:
            self.weight = 0.0
            temp_bar = self.bars.iloc[-1].copy()
            temp_bar.loc['level_weight'] = self.weight
            return temp_bar[['level_weight']]
        bar_weights = self.bars[list(self.weight_coeffs.keys())]
        for col_name in list(bar_weights.columns.values):
            bar_weights[col_name] = bar_weights[col_name].apply(
                lambda x: self.weight_coeffs[col_name] * self._norm_weight(x, norm_weights[col_name][0],
                                                                           norm_weights[col_name][1]))

        # with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        #     print(norm_weights)
        self.weight = self.bars.shape[0] + bar_weights.sum().sum()
        # print(self.weight)
        # with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            # print(bar_weights)
        # print(self.weight)
        # print()
        temp_bar = self.bars.iloc[-1]
        temp_bar.loc['level_weight'] = self.weight
        return temp_bar[['level_weight']]

    '''
        Calculate different types of weights
    '''

    def _update_quantity_weight(self):
        last_bar = self.bars.iloc[-1]
        condition = [x and y for x, y in
                     zip(self.bars.index == last_bar.name, list(self.bars["touch"] == last_bar['touch']))]
        # with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        repeated_touches = self.bars[[x for x in list(self.bars['touch'] == last_bar['touch'])]].shape[0]
        self.bars.loc[condition, ['quantity_weight']] = 1 * repeated_touches if last_bar['touch'] in ['high', 'low'] else 0.5 * repeated_touches

    def _last_bar_tail_longness(self):
        last_bar = self.bars.iloc[-1]

        bars_ochl = last_bar[['open', 'close', 'high', 'low']].values
        sorted_idxs = np.argsort(bars_ochl)
        tail = 0.0
        if last_bar['touch'] == 'high':
            # High minus closest point to high
            closest = bars_ochl[sorted_idxs[2]]
            tail = bars_ochl[2] - closest

        elif last_bar['touch'] == 'low':
            # Closest point to high minus low
            closest = bars_ochl[sorted_idxs[1]]
            tail = closest - bars_ochl[3]

        condition = [x and y for x, y in
                     zip(self.bars.index == last_bar.name, list(self.bars["touch"] == last_bar['touch']))]
        self.bars.loc[condition, ['tail_weight']] = tail / last_bar['ATR']


    def _check_limit(self):
        min_weight = 0.01
        weight = 0.01  # default weight
        last_2 = self.bars.tail(2)
        last_bar = self.bars.iloc[-1]
        date = last_bar.index
        if last_2.shape[0] == 2:
            difference = last_2.iloc[1]['count'] - last_2.iloc[0]['count']
            prev_weight = last_2.iloc[0]['limit_weight']
            if difference != 0:  # check whether these bars are not the same (e.g. bar.high = bar.open and both generate one level)
                if abs(last_bar['high'] - self.price) < last_bar['high'] * 0.0004:
                    if difference == 1:  # if these bars are consecutive days
                        if last_bar['high'] <= self.price:
                            weight = 3 * prev_weight
                        else:
                            weight = 2 * prev_weight
                    else:
                        temp = prev_weight / (2 ** (difference - 1))
                        if temp >= weight:
                            weight = temp
                elif abs(last_bar['low'] - self.price) < last_bar['low'] * 0.0004:
                    if difference == 1:
                        if last_bar['low'] >= self.price:
                            weight = 3 * prev_weight
                        else:
                            weight = 2 * prev_weight
                    else:
                        temp = prev_weight / (2 ** (difference - 1))
                        if temp >= weight:
                            weight = temp
            else:
                weight = min_weight
        else:
            weight = min_weight

        condition = [x and y for x, y in
                     zip(self.bars.index == last_bar.name, list(self.bars["touch"] == last_bar['touch']))]
        self.bars.loc[condition, ['limit_weight']] = weight
        # print(self.bars)

        # calc score for limit weights


    def _last_bar_paranormality(self):
        min_weight = 1.0
        last_bar = self.bars.iloc[-1]
        full_body = last_bar['high'] - last_bar['low']
        weight = full_body / (1.5 * last_bar['ATR'])
        condition = [x and y for x, y in
                     zip(self.bars.index == last_bar.name, list(self.bars["touch"] == last_bar['touch']))]
        if (last_bar['close'] - last_bar['open'] > 0 and last_bar['touch'] == 'high') or \
                (last_bar['close'] - last_bar['open'] < 0 and last_bar['touch'] == 'low'):
            self.bars.loc[condition, ['paranormal_weight']] = weight if weight > min_weight else min_weight
        else:
            self.bars.loc[condition, ['paranormal_weight']] = min_weight

    def _last_bar_gap(self, df):
        min_weight = 0.0
        last_bar = self.bars.iloc[-1]
        condition = [x and y for x, y in
                     zip(self.bars.index == last_bar.name, list(self.bars["touch"] == last_bar['touch']))]
        if last_bar["touch"] in ["open", "close"] or df.shape[0] < 2:
            self.bars.loc[condition, 'gap_weight'] = min_weight
            return
        prev_last_bar = df.iloc[df.index.get_loc(last_bar.name) - 1]
        hl_prev = prev_last_bar[['high', 'low']].values
        hl_cur = last_bar[['high', 'low']].values

        # Generating list of tuples  of the format:
        # [(day, value), ... ];  day can take this values: p or c, previous or current correspondingly
        # value is the high or low price value
        points = [('p', point) for point in hl_prev] + [('c', point) for point in hl_cur]
        # Sorting the array by value:
        points = sorted(points, key=lambda x: x[1])

        if points[0][0] != points[1][0]:
            self.bars.loc[condition, 'gap_weight'] = min_weight
            return
        if (points[0][0] == 'p' and self.price == last_bar["high"]) or \
                (points[0][0] == 'c' and self.price == last_bar["low"]):
            weight = (points[2][1] - points[1][1]) / (1.5 * prev_last_bar['ATR'])
            self.bars.loc[condition, 'gap_weight'] = weight
            return
        else:
            self.bars.loc[condition, 'gap_weight'] = min_weight
            return


    def _last_bar_fluctuating(self, df, num_bars=5):
        min_weight = 0.0
        last_bar = self.bars.iloc[-1]
        L = self.bars.loc[[last_bar.name]].shape[0]
        if L > 1: return
        if df.shape[0] < num_bars:
            self.bars.loc[last_bar.name, 'fluctuating_weight'] = 0.0
            return
        loc = df.index.get_loc(last_bar.name)
        bars = df.iloc[loc - num_bars:loc + 1]
        bars_ochl = bars[['open', 'close', 'high', 'low']].values

        # Check for high and low data
        in_level = []
        last_idx = 0
        for i, bars_ochl in enumerate(np.flip(bars_ochl, axis=0)):
            # Sorted close and open values for each day
            low, high = sorted(bars_ochl[:2])
            if high >= self.price >= low:
                if i - last_idx <= 1:
                    in_level.append(bars_ochl[:2])
                    last_idx = i
                else:
                    break

        if len(in_level) < 0.6 * (num_bars + 1):
            self.bars.loc[last_bar.name, 'fluctuating_weight'] = 0.0
            return
        """
            Check fluctuations for open and close
        """
        weight = min_weight
        for j in range(2):
            prev_var = in_level[0][j]
            num_change = 0.0
            last_change = 0
            for cur_var in [vars[j] for vars in in_level[1:]]:
                cur_change = np.sign(cur_var - prev_var)
                if cur_change != last_change: num_change += 1
                last_change = cur_change
                prev_var = cur_var
            change_rate = num_change / (len(in_level) - 1)
            weight += change_rate / 2

        self.bars.loc[last_bar.name, 'fluctuating_weight'] = weight