import numpy as np
import torch
import sys, os

import copy
from datetime import date, datetime, timedelta

import pandas as pd

sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from Order import Order
from Position import Position
from Portfolio import Portfolio

import config
# from QuantumCapital import constants
# from QuantumCapital.DBManager import DBManager

from DataParsers.DataEngine import DataEngine

DB_USERNAME = config.DB_USERNAME
DB_PASS = config.DB_PASS


class Broker:
    def __init__(self,
                 data,
                 n_dots,
                 step=1,
                 min_commission=1.0,
                 commission_rate=0.005,
                 slippage=True,
                 slippage_perc=0.2,
                 reindex=True,
                 return_vols=True,
                 check_stops=False,
                 precise_take_stop=False,
                 logger=None):
        assert n_dots >= step
        if precise_take_stop is True:
            assert check_stops is True
        # self.dbm = DBManager(DB_USERNAME, DB_PASS, 'ODS')

        # Fixed values
        self.logger = logger
        self.reindex = reindex
        self.return_vols = return_vols
        self.check_stops = check_stops
        self.precise_take_stop = precise_take_stop

        self.min_commission = min_commission
        self.commission = commission_rate

        self.slippage = slippage
        self.slippage_perc = slippage_perc

        self.step = step
        self.n_dots = n_dots
        # self.data = pd.read_parquet('../../Models_&_Files/Bloom_Files/rus_aligned_df_140721_280721.parquet')
        self.data = data

        self.price_cols = [col for col in self.data.columns if 'price' in col]
        self.volume_cols = [col for col in self.data.columns if 'volume' in col]
        self.rdate_cols = [col for col in self.data.columns if 'rep_date' in col]
        self.dates = self.data.index.tolist()
        self.first_dt, self.last_dt = self._get_min_max_date()

        # Current state values

        if torch.cuda.is_available():
            self.gpu = torch.device("cuda:0")
        else:
            raise Exception('No GPU')

        # temp
        # self.first_dt = self.first_dt + timedelta(375)
        self.current_dt_i = n_dots - 1
        if not self._get_range(self.dates[self.current_dt_i], half=True)[0].year == self.dates[self.current_dt_i].year:
            y = self.dates[self.current_dt_i].year
            for i, d in enumerate(self.dates):
                if d > datetime(y, 7, 1, 0, 0, 0, 0):
                    self.current_dt_i = i - 1
                    break

        # self.last_dt = self.current_dt + timedelta(300)
        if self.logger:
            self.logger.info(f'Broker data dt range {self.dates[self.current_dt_i]} - {self.last_dt}')
        self.last_indexing_dt = datetime(2000, 1, 1)
        self.current_tickers = self._current_tickers()

        assert len(self.current_tickers) > 0, 'No given year data'

        self._get_curr_bars()

        if self.precise_take_stop:
            self.data_parser = DataEngine(11)
        else:
            self.data_parser = None

    def _get_min_max_date(self):
        return self.dates[self.n_dots - 1], self.dates[-1]

    def _get_range(self, dt, half=False):
        # if 1 <= dt.month <= 6 or \
        #         (dt.month == 7 and dt.day == 1):
        #     start_dt = datetime(dt.year - 1, 7, 2, 0, 0, 0)
        #     end_dt = datetime(dt.year, 7, 1, 23, 59, 59)
        # else:
        #     start_dt = datetime(dt.year, 7, 2, 0, 0, 0)
        #     end_dt = datetime(dt.year + 1, 7, 1, 23, 59, 59)
        if not half:
            if 1 <= dt.month <= 6:
                start_dt = datetime(dt.year - 1, 7, 1, 0, 0, 0)
                end_dt = datetime(dt.year, 6, 30, 23, 59, 59)
            else:
                start_dt = datetime(dt.year, 7, 1, 0, 0, 0)
                end_dt = datetime(dt.year + 1, 6, 30, 23, 59, 59)
        else:
            if 1 <= dt.month <= 6:
                start_dt = datetime(dt.year - 1, 7, 1, 0, 0, 0)
                end_dt = datetime(dt.year - 1, 12, 31, 23, 59, 59)
            else:
                start_dt = datetime(dt.year, 7, 1, 0, 0, 0)
                end_dt = datetime(dt.year, 12, 31, 23, 59, 59)
        return start_dt, end_dt

    def _current_tickers(self):
        if self.reindex:
            start_dt, end_dt = self._get_range(self.dates[self.current_dt_i], half=True)
        else:
            start_dt, end_dt = self.dates[0].date(), self.dates[-1].date()
            start_dt, end_dt = datetime(start_dt.year, start_dt.month, start_dt.day), datetime(end_dt.year, end_dt.month, end_dt.day)
        if start_dt > self.last_indexing_dt:
            data = self.data[self.data.index.isin(pd.date_range(start_dt, end_dt))][self.price_cols].\
                dropna(axis=1, how='all')
            self.last_indexing_dt = start_dt
            if self.logger:
                self.logger.info(f'Reindexed new tickers: {data.shape[1]}')
            current_tickers = ['_'.join(col.split('_')[:-1]) for col in data.columns]
            # present_tickers = self.dbm.select_df(f'select * from bars_1_day_1russel_presence where yr={start_dt.year}')

            return current_tickers
        else:
            return self.current_tickers

    # def _find_date_idx(self, dt: date, l: int, r: int):
    #
    #     --

    def _get_curr_bars(self):
        price_df = self.data.iloc[self.current_dt_i - self.n_dots + 1:self.current_dt_i + 1][
            [ticker + '_price' for ticker in self.current_tickers]]
        self.curr_price_df = torch.from_numpy(price_df.to_numpy()).to(self.gpu)

        if self.return_vols:
            volume_df = self.data.iloc[self.current_dt_i - self.n_dots + 1:self.current_dt_i + 1][
                [ticker + '_volume' for ticker in self.current_tickers]]
            self.curr_volume_df = torch.from_numpy(volume_df.to_numpy()).to(self.gpu)
        else:
            self.curr_volume_df = None

    def get_curr_bars(self):
        return self.current_tickers, self.curr_price_df, self.curr_volume_df

    # def _subset_sum(self, numbers, n, x, indices, printsteps=False):
    #     if printsteps:
    #         self.logger.info(f'{numbers}, {n}, {x}, {indices}')
    #     # Base Cases
    #     if x == 0:
    #         return True
    #     if n == 0 and x != 0:
    #         return False
    #     # If last element is greater than x, then ignore it
    #     if numbers[n - 1][1] > x:
    #         return self._subset_sum(numbers, n - 1, x, indices, printsteps=printsteps)
    #     # else, check if x can be obtained by any of the following
    #     # (a) including the last element
    #     found = self._subset_sum(numbers, n - 1, x, indices, printsteps=printsteps)
    #     if found:
    #         return True
    #     # (b) excluding the last element
    #     indices.insert(0, numbers[n - 1][0])
    #     found = self._subset_sum(numbers, n - 1, x - numbers[n - 1][1], indices, printsteps=printsteps)
    #     if not found:
    #         indices.pop(0)
    #     return found

    def _subset_sum(self, numbers, n, x, indices, printsteps=False):
        if printsteps:
            self.logger.info(f'{numbers}, {n}, {x}, {indices}')
        # Base Cases
        if x == 0:
            return True
        if n == 0 and x != 0:
            return False
        # If last element is greater than x, then ignore it
        if numbers[n - 1][1] > x:
            return self._subset_sum(numbers, n - 1, x, indices, printsteps=printsteps)
        # else, check if x can be obtained by any of the following

        # (a) including the last element
        indices.insert(0, numbers[n - 1][0])
        found = self._subset_sum(numbers, n - 1, x - numbers[n - 1][1], indices, printsteps=printsteps)
        if not found:
            indices.pop(0)
        else:
            return True

        # (b) excluding the last element
        found = self._subset_sum(numbers, n - 1, x, indices, printsteps=printsteps)
        if found:
            return True

        return found

    def make_order(self, portfolio: Portfolio, order: Order, set_date_idx=None, set_price=None, stop_trigger=False,
                   take_trigger=False, precise_take_stop=False):
        # print(order)
        # assert self.current_bars[0], 'Impossible to make order, not RTH'
        # price = self.current_bars[1][order.ticker][0]
        i = self.current_tickers.index(order.ticker)
        price = self.curr_price_df[-1, i]
        current_dt_i = self.current_dt_i

        if set_date_idx:
            assert set_date_idx <= self.current_dt_i
            price = self.curr_price_df[-1 - (self.current_dt_i - set_date_idx), i]
            current_dt_i = set_date_idx

        if set_price:
            price = float(set_price)

        if order.dir == 'buy':
            assert order.amount > 0
            if order.kind in ('market', 'market_TS'):
                comm = max(self.min_commission, order.amount * self.commission)
                if self.slippage:
                    entry_price = price + price * 0.01 * self.slippage_perc
                else:
                    entry_price = price
                entry_price = entry_price + comm / order.amount
                position = Position(self.dates[current_dt_i],
                                    entry_price if type(entry_price) is float else float(entry_price.cpu().numpy()),  # entry price
                                    order.ticker,
                                    order.amount,
                                    stop=order.ts if order.kind == 'market_TS' else None,
                                    info=order.info)
                cash_diff = position.amount * position.entry_price
                cash_diff = cash_diff if type(cash_diff) is float else float(cash_diff.cpu().numpy())
                portfolio.cash = [self.dates[self.current_dt_i], portfolio.cash - cash_diff]
                # assert portfolio.cash >= 0, 'Margin CALL!!!'
                portfolio.open_positions.append(position)
            else:
                raise Exception('Only market type order is implemented')
        elif order.dir == 'sell':
            assert order.amount > 0
            if order.kind == 'market':
                op_full_idx = None
                op_partial_id_amnt = []
                for i, open_pose in enumerate(portfolio.open_positions):
                    if open_pose.ticker == order.ticker:
                        if open_pose.amount == order.amount:
                            op_full_idx = i
                            break
                        elif open_pose.amount > 0:
                            op_partial_id_amnt.append((i, open_pose.amount))
                        elif open_pose.amount < 0:
                            continue
                        else:
                            raise Exception('0 amount open pose')
                assert op_full_idx is not None or len(op_partial_id_amnt) != 0
                if op_full_idx is not None:
                    assert portfolio.open_positions[op_full_idx].amount == order.amount

                    position = copy.deepcopy(portfolio.open_positions[op_full_idx])
                    del portfolio.open_positions[op_full_idx]

                    comm = max(self.min_commission, order.amount * self.commission)
                    position.exit_dt = self.dates[current_dt_i].date()
                    if self.slippage:
                        exit_price = price - price * 0.01 * self.slippage_perc
                    else:
                        exit_price = price
                    exit_price = exit_price - comm / order.amount

                    position.exit_price = exit_price
                    position.take_triggered = take_trigger
                    position.stop_triggered = stop_trigger
                    position.precise_take_stop = precise_take_stop
                    # position.pnl = position.amount * (position.exit_price - position.entry_price)
                    cash_diff = position.amount * position.exit_price
                    cash_diff = cash_diff if isinstance(cash_diff, (float, np.floating)) else float(cash_diff.cpu().numpy())

                    portfolio.cash = [self.dates[self.current_dt_i], portfolio.cash + cash_diff]
                    portfolio.closed_positions.append(position)
                else:
                    ops_amount_sum = sum([id_amnt[1] for id_amnt in op_partial_id_amnt])
                    assert ops_amount_sum >= order.amount

                    indexes = []
                    found = self._subset_sum(op_partial_id_amnt, len(op_partial_id_amnt), order.amount, indexes)
                    if found:
                        for i in indexes:
                            position = copy.deepcopy(portfolio.open_positions[i])
                            comm = max(self.min_commission, order.amount * self.commission)
                            position.exit_dt = self.dates[current_dt_i].date()
                            if self.slippage:
                                exit_price = price - price * 0.01 * self.slippage_perc
                            else:
                                exit_price = price
                            exit_price = exit_price - comm / order.amount
                            position.exit_price = exit_price
                            position.take_triggered = take_trigger
                            position.stop_triggered = stop_trigger
                            position.precise_take_stop = precise_take_stop
                            # position.pnl = position.amount * (position.exit_price - position.entry_price)
                            cash_diff = position.amount * position.exit_price
                            cash_diff = cash_diff if type(cash_diff) is float else float(cash_diff.cpu().numpy())
                            portfolio.cash = [self.dates[self.current_dt_i], portfolio.cash + cash_diff]
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
                            position = copy.deepcopy(portfolio.open_positions[i[0]])
                            comm = max(self.min_commission, order.amount * self.commission)
                            position.exit_dt = self.dates[current_dt_i].date()
                            if self.slippage:
                                exit_price = price - price * 0.01 * self.slippage_perc
                            else:
                                exit_price = price
                            exit_price = exit_price - comm / order.amount
                            position.exit_price = exit_price
                            position.take_triggered = take_trigger
                            position.stop_triggered = stop_trigger
                            position.precise_take_stop = precise_take_stop
                            # position.pnl = position.amount * (position.exit_price - position.entry_price)
                            cash_diff = position.amount * position.exit_price
                            cash_diff = cash_diff if type(cash_diff) is float else float(cash_diff.cpu().numpy())
                            portfolio.cash = [self.dates[self.current_dt_i], portfolio.cash + cash_diff]
                            portfolio.closed_positions.append(position)
                            to_close -= position.amount

                        position = copy.deepcopy(portfolio.open_positions[indexes[-1][0]])
                        position.amount = to_close
                        comm = max(self.min_commission, order.amount * self.commission)

                        position.exit_dt = self.dates[current_dt_i].date()
                        if self.slippage:
                            exit_price = price - price * 0.01 * self.slippage_perc
                        else:
                            exit_price = price
                        exit_price = exit_price - comm / order.amount
                        position.exit_price = exit_price
                        position.take_triggered = take_trigger
                        position.stop_triggered = stop_trigger
                        position.precise_take_stop = precise_take_stop
                        # position.pnl = position.amount * (position.exit_price - position.entry_price)
                        cash_diff = position.amount * position.exit_price
                        cash_diff = cash_diff if type(cash_diff) is float else float(cash_diff.cpu().numpy())
                        portfolio.cash = [self.dates[self.current_dt_i], portfolio.cash + cash_diff]
                        portfolio.open_positions[indexes[-1][0]].amount -= to_close
                        portfolio.closed_positions.append(position)

                        indexes = sorted([i for i, _ in indexes[:-1]], reverse=True)

                        for i in indexes:
                            del portfolio.open_positions[i]
            else:
                raise Exception('Only market order is implemented')
        elif order.dir == 'short':
            assert order.amount < 0
            if order.kind in ('market', 'market_TS'):
                comm = max(self.min_commission, -order.amount * self.commission)
                if self.slippage:
                    entry_price = price - price * 0.01 * self.slippage_perc
                else:
                    entry_price = price
                entry_price = entry_price - comm / order.amount
                position = Position(self.dates[current_dt_i],
                                    entry_price if type(entry_price) is float else float(entry_price.cpu().numpy()),
                                    # entry price
                                    order.ticker,
                                    order.amount,
                                    stop=order.ts if order.kind == 'market_TS' else None,
                                    info=order.info)
                cash_diff = position.amount * position.entry_price
                cash_diff = cash_diff if type(cash_diff) is float else float(cash_diff.cpu().numpy())
                portfolio.cash = [self.dates[self.current_dt_i], portfolio.cash - cash_diff]
                # assert portfolio.cash >= 0, 'Margin CALL!!!'
                portfolio.open_positions.append(position)
            else:
                raise Exception('Only market type order is implemented')
        elif order.dir == 'cover':
            assert order.amount < 0
            if order.kind == 'market':
                op_full_idx = None
                op_partial_id_amnt = []
                for i, open_pose in enumerate(portfolio.open_positions):
                    if open_pose.ticker == order.ticker:
                        # if open_pose.amount == order.amount:
                        #     op_full_idx = i
                        #     break
                        # else:
                        #     op_partial_id_amnt.append((i, open_pose.amount))

                        if open_pose.amount == order.amount:
                            op_full_idx = i
                            break
                        elif open_pose.amount < 0:
                            op_partial_id_amnt.append((i, open_pose.amount))
                        elif open_pose.amount > 0:
                            continue
                        else:
                            raise Exception('0 amount open pose')
                assert op_full_idx is not None or len(op_partial_id_amnt) != 0
                if op_full_idx is not None:
                    assert portfolio.open_positions[op_full_idx].amount == order.amount

                    position = copy.deepcopy(portfolio.open_positions[op_full_idx])
                    del portfolio.open_positions[op_full_idx]

                    comm = max(self.min_commission, order.amount * self.commission)
                    position.exit_dt = self.dates[current_dt_i].date()
                    if self.slippage:
                        exit_price = price + price * 0.01 * self.slippage_perc
                    else:
                        exit_price = price
                    exit_price = exit_price + comm / order.amount
                    position.exit_price = exit_price
                    # position.pnl = position.amount * (position.exit_price - position.entry_price)
                    cash_diff = position.amount * position.exit_price
                    cash_diff = cash_diff if isinstance(cash_diff, (float, np.floating)) else float(cash_diff.cpu().numpy())

                    portfolio.cash = [self.dates[self.current_dt_i], portfolio.cash + cash_diff]
                    portfolio.closed_positions.append(position)
                else:
                    ops_amount_sum = sum([id_amnt[1] for id_amnt in op_partial_id_amnt])
                    assert ops_amount_sum <= order.amount

                    indexes = []
                    found = self._subset_sum([(id_amnt[0], -id_amnt[1]) for id_amnt in op_partial_id_amnt],
                                             len(op_partial_id_amnt), -order.amount, indexes)
                    if found:
                        for i in indexes:
                            position = copy.deepcopy(portfolio.open_positions[i])
                            comm = max(self.min_commission, order.amount * self.commission)
                            position.exit_dt = self.dates[current_dt_i].date()
                            if self.slippage:
                                exit_price = price + price * 0.01 * self.slippage_perc
                            else:
                                exit_price = price
                            exit_price = exit_price + comm / order.amount
                            position.exit_price = exit_price
                            # position.pnl = position.amount * (position.exit_price - position.entry_price)
                            cash_diff = position.amount * position.exit_price
                            cash_diff = cash_diff if type(cash_diff) is float else float(cash_diff.cpu().numpy())
                            portfolio.cash = [self.dates[self.current_dt_i], portfolio.cash + cash_diff]
                            portfolio.closed_positions.append(position)
                        for i in reversed(indexes):
                            del portfolio.open_positions[i]
                    else:
                        op_partial_id_amnt.sort(key=lambda x: x[1], reverse=True)  # # # # # # # # #
                        curr_sum = 0
                        i = 0
                        while curr_sum > order.amount:
                            curr_sum += op_partial_id_amnt[i][1]
                            i += 1
                        indexes = op_partial_id_amnt[:i]

                        to_close = order.amount
                        for i in indexes[:-1]:
                            position = copy.deepcopy(portfolio.open_positions[i[0]])
                            comm = max(self.min_commission, order.amount * self.commission)
                            position.exit_dt = self.dates[current_dt_i].date()
                            if self.slippage:
                                exit_price = price + price * 0.01 * self.slippage_perc
                            else:
                                exit_price = price
                            exit_price = exit_price + comm / order.amount
                            position.exit_price = exit_price
                            # position.pnl = position.amount * (position.exit_price - position.entry_price)
                            cash_diff = position.amount * position.exit_price
                            cash_diff = cash_diff if type(cash_diff) is float else float(cash_diff.cpu().numpy())
                            portfolio.cash = [self.dates[self.current_dt_i], portfolio.cash + cash_diff]
                            portfolio.closed_positions.append(position)
                            to_close -= position.amount

                        position = copy.deepcopy(portfolio.open_positions[indexes[-1][0]])
                        position.amount = to_close
                        comm = max(self.min_commission, order.amount * self.commission)

                        position.exit_dt = self.dates[current_dt_i].date()
                        if self.slippage:
                            exit_price = price + price * 0.01 * self.slippage_perc
                        else:
                            exit_price = price
                        exit_price = exit_price + comm / order.amount
                        position.exit_price = exit_price
                        # position.pnl = position.amount * (position.exit_price - position.entry_price)
                        cash_diff = position.amount * position.exit_price
                        cash_diff = cash_diff if type(cash_diff) is float else float(cash_diff.cpu().numpy())
                        portfolio.cash = [self.dates[self.current_dt_i], portfolio.cash + cash_diff]
                        portfolio.open_positions[indexes[-1][0]].amount -= to_close
                        portfolio.closed_positions.append(position)

                        indexes = sorted([i for i, _ in indexes[:-1]], reverse=True)

                        for i in indexes:
                            del portfolio.open_positions[i]
            else:
                raise Exception('Only market order is implemented')
        else:
            raise Exception('No such order type')

    def _update_market_val(self, portfolio: Portfolio):
        # assert self.current_bars[0], 'Not RTH!'
        market_val = portfolio.cash
        for op in portfolio.open_positions:
            # price = self.current_bars[1][op.ticker][0]
            i = self.current_tickers.index(op.ticker)
            price = self.curr_price_df[-1, i]
            comm = max(self.min_commission, op.amount * self.commission)
            if op.amount > 0:
                price = float(price.cpu().numpy())
                if self.slippage:
                    price -= price * 0.01 * self.slippage_perc
                value = price * op.amount - comm
            elif op.amount < 0:
                price = float(price.cpu().numpy())
                if self.slippage:
                    price += price * 0.01 * self.slippage_perc
                value = price * op.amount + comm
            else:
                raise Exception
            # pnl = op.amount * (price - op.entry_price)
            market_val += value
        portfolio.market_value = [self.dates[self.current_dt_i], market_val]

    def get_long_positions_market_val(self, portfolio: Portfolio):
        market_val = 0
        for op in portfolio.open_positions:
            if op.amount < 0:
                continue
            # price = self.current_bars[1][op.ticker][0]
            i = self.current_tickers.index(op.ticker)
            price = self.curr_price_df[-1, i]
            comm = max(self.min_commission, op.amount * self.commission)
            value = float(price.cpu().numpy()) * op.amount - comm
            # pnl = op.amount * (price - op.entry_price)
            market_val += value
        return market_val

    def _close_absent_tickers(self, portfolio: Portfolio, tickers):
        assert len(set(tickers)) == len(tickers)

        for ticker in tickers:
            amount = portfolio.number_of_shares_open(ticker)
            if amount > 0:
                self.make_order(portfolio, Order(ticker, 'market', 'sell', amount, 'shares'))
            elif amount < 0:
                self.make_order(portfolio, Order(ticker, 'market', 'cover', amount, 'shares'))

    def set_stop(self, portfolio: Portfolio, ticker: str, price: float):
        portfolio.assert_orders_onesided()
        for pos in portfolio.open_positions:
            if pos.ticker != ticker:
                continue
            assert pos.amount != 0
            is_long = True if pos.amount > 0 else False

        if is_long:
            stop_price = float('-inf')
            for pos in portfolio.open_positions:
                if pos.ticker != ticker:
                    continue
                stop_price = max(stop_price, pos.stop) if pos.stop is not None else stop_price
            price = float(max(stop_price, price))

            for pos in portfolio.open_positions:
                if pos.ticker != ticker:
                    continue
                pos.stop = [self.dates[self.current_dt_i], price]
        else:
            stop_price = float('inf')
            for pos in portfolio.open_positions:
                if pos.ticker != ticker:
                    continue
                stop_price = min(stop_price, pos.stop) if pos.stop is not None else stop_price
            price = float(min(stop_price, price))

            for pos in portfolio.open_positions:
                if pos.ticker != ticker:
                    continue
                pos.stop = [self.dates[self.current_dt_i], price]

    def _close_stop_positions(self, portfolio: Portfolio):
        portfolio.assert_orders_onesided()
        tickers_stops = {}
        for i, pos in enumerate(portfolio.open_positions):
            if pos.stop is not None:
                assert pos.stop == pos.stop_hist[-1][1]
                if pos.ticker not in tickers_stops:
                    tickers_stops[pos.ticker] = (pos.stop, pos.amount)
                else:
                    assert tickers_stops[pos.ticker][0] == pos.stop
        tickers_triggered = set()
        for ticker in tickers_stops:
            curr_tickers, price_df, volume_df = self.get_curr_bars()
            ticker_idx = curr_tickers.index(ticker)
            price_hist = price_df[-self.step:, ticker_idx]
            is_long = True if tickers_stops[ticker][1] > 0 else False

            precise_stop = False
            if self.precise_take_stop:
                bars = self.data_parser.get_bars(ticker, self.dates[self.current_dt_i - self.step + 1], self.dates[self.current_dt_i], False, which='db')
                lows_hist = None
                if bars is not None and bars.shape[0] == self.step:
                    ratios = price_hist / torch.from_numpy(bars.close.to_numpy()).to(self.gpu)
                    ratios_mean = ratios.mean()
                    ratios_std = ratios.std()
                    perc_diff = ratios_std / ratios_mean
                    if perc_diff <= 0.01:
                        lows_hist = ratios_mean * torch.from_numpy(bars.low.to_numpy() if is_long else bars.high.to_numpy()).to(self.gpu)
                if lows_hist is not None:
                    price_hist = lows_hist
                    precise_stop = True
            if is_long:
                stop_idx = (price_hist <= tickers_stops[ticker][0]).nonzero(as_tuple=True)[0]
            else:
                stop_idx = (price_hist >= tickers_stops[ticker][0]).nonzero(as_tuple=True)[0]
            if stop_idx.shape[0] > 0:
                stop_idx = int(stop_idx[0].cpu().numpy())
                dt_idx = self.current_dt_i - (self.step - 1 - stop_idx)
                self.make_order(portfolio, Order(ticker,
                                                 'market',
                                                 'sell' if is_long else 'cover',
                                                 portfolio.number_of_shares_open(ticker),
                                                 'shares'),
                                set_date_idx=dt_idx,
                                set_price=tickers_stops[ticker],
                                stop_trigger=True,
                                precise_take_stop=precise_stop)
                tickers_triggered.add(ticker)
        return tickers_triggered

    # def _print_spec_pose(self, portfolio):
    #     for pos in portfolio.closed_positions:
    #         if pos.ticker == 'CTLM' and pos.entry_price == 5.765 and pos.amount == 2052:
    #             print(pos)

    def set_take(self, portfolio: Portfolio, ticker: str, price: float):
        portfolio.assert_orders_onesided()
        for pos in portfolio.open_positions:
            if pos.ticker != ticker:
                continue
            pos.take = price

    def _close_take_positions(self, portfolio: Portfolio):
        portfolio.assert_orders_onesided()
        tickers_takes = {}
        for i, pos in enumerate(portfolio.open_positions):
            if pos.take is not None:
                assert pos.take == pos.take_hist[-1][1]
                if pos.ticker not in tickers_takes:
                    tickers_takes[pos.ticker] = (pos.take, pos.amount)
                else:
                    assert tickers_takes[pos.ticker][0] == pos.take
            else:
                raise Exception(f'No stop for ticker {pos.ticker}')
        tickers_triggered = set()
        for ticker in tickers_takes:
            curr_tickers, price_df, volume_df = self.get_curr_bars()
            ticker_idx = curr_tickers.index(ticker)
            price_hist = price_df[-self.step:, ticker_idx]
            is_long = True if tickers_takes[ticker][1] > 0 else False

            precise_stop = False
            if self.precise_take_stop:
                bars = self.data_parser.get_bars(ticker, self.dates[self.current_dt_i - self.step + 1],
                                                 self.dates[self.current_dt_i], False, which='db')
                highs_hist = None
                if bars is not None and bars.shape[0] == self.step:
                    ratios = price_hist / torch.from_numpy(bars.close.to_numpy()).to(self.gpu)
                    ratios_mean = ratios.mean()
                    ratios_std = ratios.std()
                    perc_diff = ratios_std / ratios_mean
                    if perc_diff <= 0.01:
                        highs_hist = ratios_mean * torch.from_numpy(bars.high.to_numpy() if is_long else bars.low.to_numpy()).to(self.gpu)
                if highs_hist is not None:
                    price_hist = highs_hist
                    precise_stop = True
            if is_long:
                stop_idx = (price_hist >= tickers_takes[ticker][0]).nonzero(as_tuple=True)[0]
            else:
                stop_idx = (price_hist <= tickers_takes[ticker][0]).nonzero(as_tuple=True)[0]
            if stop_idx.shape[0] > 0:
                stop_idx = int(stop_idx[0].cpu().numpy())
                dt_idx = self.current_dt_i - (self.step - 1 - stop_idx)
                self.make_order(portfolio, Order(ticker,
                                                 'market',
                                                 'sell' if is_long else 'cover',
                                                 portfolio.number_of_shares_open(ticker),
                                                 'shares'),
                                set_date_idx=dt_idx,
                                set_price=tickers_takes[ticker],
                                take_trigger=True,
                                precise_take_stop=precise_stop)
                tickers_triggered.add(ticker)
        return tickers_triggered

    def go_next_rth_day(self, portfolios: list):
        # self._print_spec_pose(portfolios[0])

        self.current_dt_i += self.step
        # self.current_dt += timedelta(self.step)
        # if self.current_dt.weekday() >= 5:
        #     self.current_dt += timedelta(7 - self.current_dt.weekday())

        stop_tickers = set()

        new_tickers = self._current_tickers()
        absent_tickers = list(set(self._current_tickers()) - set(new_tickers))
        for portfolio in portfolios:
            self._close_absent_tickers(portfolio, absent_tickers)
        self.current_tickers = new_tickers
        self._get_curr_bars()
        for portfolio in portfolios:
            if self.check_stops:
                stop_tickers.update(self._close_stop_positions(portfolio))
            self._update_market_val(portfolio)
        if self.current_dt_i + self.step > len(self.dates) - 1:
            for portfolio in portfolios:
                self._close_absent_tickers(portfolio, self.current_tickers)
            del self.data_parser
            return True, stop_tickers
        return False, stop_tickers


if __name__ == '__main__':
    # broker = Broker(125)
    # for i in range()
    # bars = broker.get_curr_bars()
    df = pd.read_parquet('../Models_&_Files/Bloom_Files/rus_aligned_df_140721_280721.parquet')
    print(df)
    broker = Broker(df, 125)
