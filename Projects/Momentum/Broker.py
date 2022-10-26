import sys

import copy
from datetime import date, datetime, timedelta

import pandas as pd

from Order import Order
from Position import Position
from Portfolio import Portfolio

import config
from QuantumCapital import constants
from QuantumCapital.DBManager import DBManager


DB_USERNAME = config.DB_USERNAME
DB_PASS = config.DB_PASS


class Broker:
    def __init__(self, logger=None, start_dt=None, step=1, min_commission=1.0, commission_rate=0.005):
        # Fixed values\
        if logger is not None:
            self.logger = logger
        else:
            self.logger = None

        # self.dbm = DBManager(DB_USERNAME, DB_PASS, 'ODS')

        self.min_commission = min_commission
        self.commission = commission_rate
        self.step = step

        self.close_hist = pd.read_parquet('rus_close_201120.parquet')
        self.volume_hist = pd.read_parquet('rus_volume_201120.parquet')


        # self.close_hist = pd.read_parquet('SPX_price_230920.parquet')
        # self.volume_hist = pd.read_parquet('SPX_volume_230920.parquet')

        self.except_tickers = [col for col in list(self.close_hist.columns) if '.' in col and 'date' not in col]

        # Current state values
        self.first_dt, self.last_dt = self._get_min_max_date()
        if start_dt is not None:
            self.first_dt = start_dt

        # temp
        # self.first_dt = self.first_dt + timedelta(375)
        self.current_dt = self.first_dt
        # self.last_dt = self.current_dt + timedelta(300)

        print('dates range', self.current_dt, self.last_dt)
        self.last_indexing_dt = datetime(2000, 1, 1)
        self.current_tickers = self._current_tickers()

        assert len(self.current_tickers) > 0, 'No given year data'

        if self.current_dt.weekday() >= 5:
            self.current_dt += timedelta(7 - self.current_dt.weekday())
        self.current_bars = self._get_curr_bars()
        while not self.current_bars[0]:
            self.current_dt += timedelta(1)
            self.current_bars = self._get_curr_bars()

    def _get_min_max_date(self):
        s = pd.Series([])
        for i in range(0, self.close_hist.shape[1] // 100, 2):
            dts = pd.to_datetime(self.close_hist.iloc[:, i], errors='coerce')
            dts.dropna(inplace=True)
            s = s.append(dts)
        return s.min(), s.max()

    def _get_range(self, dt):
        if 1 <= dt.month <= 6 or \
                (dt.month == 7 and dt.day == 1):
            start_dt = datetime(dt.year - 1, 7, 2, 0, 0, 0)
            end_dt = datetime(dt.year, 7, 1, 23, 59, 59)
        else:
            start_dt = datetime(dt.year, 7, 2, 0, 0, 0)
            end_dt = datetime(dt.year + 1, 7, 1, 23, 59, 59)
        return start_dt, end_dt

    # def _get_range(self, dt, half=False):
    #     # if 1 <= dt.month <= 6 or \
    #     #         (dt.month == 7 and dt.day == 1):
    #     #     start_dt = datetime(dt.year - 1, 7, 2, 0, 0, 0)
    #     #     end_dt = datetime(dt.year, 7, 1, 23, 59, 59)
    #     # else:
    #     #     start_dt = datetime(dt.year, 7, 2, 0, 0, 0)
    #     #     end_dt = datetime(dt.year + 1, 7, 1, 23, 59, 59)
    #     if not half:
    #         if 1 <= dt.month <= 6:
    #             start_dt = datetime(dt.year - 1, 7, 1, 0, 0, 0)
    #             end_dt = datetime(dt.year, 6, 30, 23, 59, 59)
    #         else:
    #             start_dt = datetime(dt.year, 7, 1, 0, 0, 0)
    #             end_dt = datetime(dt.year + 1, 6, 30, 23, 59, 59)
    #     else:
    #         if 1 <= dt.month <= 6:
    #             start_dt = datetime(dt.year - 1, 7, 1, 0, 0, 0)
    #             end_dt = datetime(dt.year - 1, 12, 31, 23, 59, 59)
    #         else:
    #             start_dt = datetime(dt.year, 7, 1, 0, 0, 0)
    #             end_dt = datetime(dt.year, 12, 31, 23, 59, 59)
    #     return start_dt, end_dt

    def _current_tickers(self):
        start_dt, end_dt = self._get_range(self.current_dt)
        if start_dt > self.last_indexing_dt:
            print('Reindexing...')
            current_tickers = []
            for i in range(0, self.close_hist.shape[1], 2):  # for SPX
                df = self.close_hist.iloc[:, i:i+2]
                ticker = df.columns[1]
                df = df.iloc[:, 0]
                if sum(df.isin(pd.date_range(start_dt, end_dt))) > 0 and ticker not in self.except_tickers:
                    current_tickers.append(ticker)
            self.last_indexing_dt = start_dt
            print('Number of tickers:', len(current_tickers))
            if self.logger: self.logger.info(f'Reindex: {len(current_tickers)}')
            return current_tickers
        else:
            return self.current_tickers
    
    def _get_curr_bars(self):
        # print('get curr bars')
        closes = {}
        zero_vol_cnt = 0
        for i, ticker in enumerate(self.current_tickers):
            i = self.close_hist.columns.get_loc(ticker)
            df_c = self.close_hist.iloc[:, i-1:i+1]
            # print(self.current_dt)
            j = self.volume_hist.columns.get_loc(ticker)
            df_v = self.volume_hist.iloc[:, j-1:j+1]
            # print(i, ticker)
            close = df_c[df_c.iloc[:, 0].isin(pd.date_range(
                        datetime(self.current_dt.year, self.current_dt.month, self.current_dt.day, 0, 0, 0),
                        datetime(self.current_dt.year, self.current_dt.month, self.current_dt.day, 23, 59, 59)))]
            # if ticker == 'uspi':
            #     print('1st try')
            #     print(close)
            # print(close)
            volume = df_v[df_v.iloc[:, 0].isin(pd.date_range(
                        datetime(self.current_dt.year, self.current_dt.month, self.current_dt.day, 0, 0, 0),
                        datetime(self.current_dt.year, self.current_dt.month, self.current_dt.day, 23, 59, 59)))]
            if volume.shape[0] == 1:
                volume = float(volume.iloc[0, 1])
            elif volume.shape[0] == 0:
                volume = 0
                zero_vol_cnt += 1
            else:
                raise Exception
            if close.shape[0] == 1:
                close = float(close.iloc[0, 1])
            elif close.shape[0] == 0:
                start_dt, end_dt = self._get_range(self.current_dt)
                start_dt = datetime(start_dt.year - 1, start_dt.month, 1)
                close = df_c[df_c.iloc[:, 0].isin(pd.date_range(start_dt, end_dt))]
                # if ticker == 'uspi':
                #     print('2nd try')
                #     print(start_dt, end_dt)
                #     print(close)
                if close.iloc[0, 0] <= self.current_dt:
                    close = close[close.iloc[:, 0].isin(pd.date_range(start_dt, self.current_dt))]
                    close = float(close.iloc[-1, 1])
                    # if ticker == 'uspi':
                    #     print('gooood', close)
                else:
                    # if ticker == 'uspi':
                    #     print('VOT ZDES NONE')
                    close = None
                    volume = None
            else:
                raise Exception
            closes[ticker] = close, volume
        if zero_vol_cnt == len(self.current_tickers):
            # for ticker in closes:
            #     closes[ticker] = None, None
            return False, {}
        return True, closes

    def get_curr_bars(self):
        return self.current_bars[1]

    def _subset_sum(self, numbers, n, x, indices):
        # Base Cases
        if x == 0:
            return True
        if n == 0 and x != 0:
            return False
        # If last element is greater than x, then ignore it
        if numbers[n - 1][1] > x:
            return self._subset_sum(numbers, n - 1, x, indices)
        # else, check if x can be obtained by any of the following
        # (a) including the last element
        found = self._subset_sum(numbers, n - 1, x, indices)
        if found:
            return True
        # (b) excluding the last element
        indices.insert(0, numbers[n - 1][0])
        found = self._subset_sum(numbers, n - 1, x - numbers[n - 1][1], indices)
        if not found:
            indices.pop(0)
        return found

    def make_order(self, portfolio: Portfolio, order: Order):
        assert self.current_bars[0], 'Impossible to make order, not RTH'
        price = self.current_bars[1][order.ticker][0]

        if order.dir == 'buy':
            if order.kind == 'market':
                comm = max(self.min_commission, order.amount * self.commission)
                position = Position(self.current_dt,
                                    price + comm / order.amount,  # entry price
                                    order.ticker,
                                    order.amount)
                portfolio.cash -= position.amount * position.entry_price
                # assert portfolio.cash >= 0, 'Margin CALL!!!'
                portfolio.open_positions.append(position)
            else:
                raise Exception('Only market order is implemented')
        else:
            if order.kind == 'market':
                op_full_idx = None
                op_partial_id_amnt = []
                for i, open_pose in enumerate(portfolio.open_positions):
                    if open_pose.ticker == order.ticker:
                        if open_pose.amount == order.amount:
                            op_full_idx = i
                            break
                        else:
                            op_partial_id_amnt.append((i, open_pose.amount))
                assert op_full_idx is not None or len(op_partial_id_amnt) != 0
                if op_full_idx is not None:
                    assert portfolio.open_positions[op_full_idx].amount == order.amount

                    position = copy.copy(portfolio.open_positions[op_full_idx])
                    del portfolio.open_positions[op_full_idx]

                    comm = max(self.min_commission, order.amount * self.commission)
                    position.exit_dt = self.current_dt
                    position.exit_price = price - comm / order.amount
                    position.pnl = position.amount * (position.exit_price - position.entry_price)
                    portfolio.cash += position.amount * position.exit_price
                    portfolio.closed_positions.append(position)
                else:
                    ops_amount_sum = sum([id_amnt[1] for id_amnt in op_partial_id_amnt])
                    assert ops_amount_sum >= order.amount

                    indexes = []
                    found = self._subset_sum(op_partial_id_amnt, len(op_partial_id_amnt), order.amount, indexes)
                    if found:
                        for i in indexes:
                            position = copy.copy(portfolio.open_positions[i])
                            comm = max(self.min_commission, order.amount * self.commission)
                            position.exit_dt = self.current_dt
                            position.exit_price = price - comm / order.amount
                            position.pnl = position.amount * (position.exit_price - position.entry_price)
                            portfolio.cash += position.amount * position.exit_price
                            portfolio.closed_positions.append(position)
                        for i in reversed(indexes):
                            del portfolio.open_positions[i]
                    else:
                        op_partial_id_amnt.sort(key=lambda x: x[1])
                        curr_sum = 0
                        i = 0
                        while curr_sum < order.amount:
                            curr_sum += op_partial_id_amnt[i][1]
                            i += 1
                        indexes = op_partial_id_amnt[:i]

                        to_close = order.amount
                        for i in indexes[:-1]:
                            position = copy.copy(portfolio.open_positions[i[0]])
                            comm = max(self.min_commission, order.amount * self.commission)
                            position.exit_dt = self.current_dt
                            position.exit_price = price - comm / order.amount
                            position.pnl = position.amount * (position.exit_price - position.entry_price)
                            portfolio.cash += position.amount * position.exit_price
                            portfolio.closed_positions.append(position)
                            to_close -= position.amount

                        position = copy.copy(portfolio.open_positions[indexes[-1][0]])
                        position.amount = to_close
                        comm = max(self.min_commission, order.amount * self.commission)

                        position.exit_dt = self.current_dt
                        position.exit_price = price - comm / order.amount
                        position.pnl = position.amount * (position.exit_price - position.entry_price)
                        portfolio.cash += position.amount * position.exit_price
                        portfolio.open_positions[indexes[-1][0]].amount -= to_close
                        portfolio.closed_positions.append(position)

                        indexes = sorted([i for i, _ in indexes[:-1]], reverse=True)

                        for i in indexes:
                            del portfolio.open_positions[i]
            else:
                raise Exception('Only market order is implemented')

    def _update_market_val(self, portfolio: Portfolio):
        assert self.current_bars[0], 'Not RTH!'
        market_val = portfolio.cash
        for op in portfolio.open_positions:
            price = self.current_bars[1][op.ticker][0]
            comm = max(self.min_commission, op.amount * self.commission)
            # if price is None:
            #     print('hey')
            #     print(op)
            value = price * op.amount - comm
            # pnl = op.amount * (price - op.entry_price)
            market_val += value
        portfolio.market_value = market_val

    def _close_absent_tickers(self, portfolio: Portfolio, tickers):
        assert len(set(tickers)) == len(tickers)

        for ticker in tickers:
            amount = portfolio.number_of_shares_open(ticker)
            if amount > 0:
                self.make_order(portfolio, Order(ticker, 'market', 'sell', amount, 'shares'))

        # ops_to_remove = []
        # for ticker in tickers:
        #     for j, position in enumerate(portfolio.open_positions):
        #         if position.ticker == ticker:
        #             comm = max(self.min_commission, position.amount * self.commission)
        #             start_dt, end_dt = self._get_range(date(self.current_dt.year - 1,
        #                                                     self.current_dt.month,
        #                                                     self.current_dt.day))
        #             i = self.close_hist.columns.get_loc(ticker)
        #             df = self.close_hist.iloc[:, i-1:i+1]
        #             # df = df[df.iloc[0]]
        #             close = df[df.iloc[:, 0].isin(pd.date_range(start_dt, end_dt))]
        #             close = float(close.iloc[-1, 1])
        #
        #             position.exit_dt = end_dt
        #             position.exit_price = close - comm / position.amount
        #             position.pnl = position.amount * (position.exit_price - position.entry_price)
        #             portfolio.cash += position.amount * position.exit_price
        #             closed_position = copy.copy(position)
        #             portfolio.closed_positions.append(closed_position)
        #             ops_to_remove.append(j)
        # for i in ops_to_remove:
        #     del portfolio.open_positions[i]

    def go_next_rth_day(self, portfolio: Portfolio):
        self.current_dt += timedelta(self.step)
        if self.current_dt.weekday() >= 5:
            self.current_dt += timedelta(7 - self.current_dt.weekday())

        new_tickers = self._current_tickers()
        absent_tickers = list(set(self._current_tickers()) - set(new_tickers))
        self._close_absent_tickers(portfolio, absent_tickers)
        self.current_tickers = new_tickers
        self.current_bars = self._get_curr_bars()  # Monday
        while not self.current_bars[0]:
            self.current_dt += timedelta(1)
            self.current_bars = self._get_curr_bars()
        self._update_market_val(portfolio)
        if self.current_dt == self.last_dt:
            return True
        return False


if __name__ == '__main__':
    broker = Broker()
    # for i in range()
    # bars = broker.get_curr_bars()
    # print(bars)
