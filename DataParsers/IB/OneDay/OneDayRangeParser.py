import sys

import numpy as np
import pandas as pd
import pytz

from datetime import datetime, timedelta, date

from ib_insync import Stock, IB, util


sys.path.append('../../..')
from QuantumCapital import constants


class OneDayRangeParser:
    def __init__(self, dbm, ib, logger):
        self.dbm = dbm
        self.ib = ib
        self.logger = logger

    def parse_range(self, ticker: str, start_date: date, end_date: date):
        """
        :param ticker:
        :param start_date:
        :param end_date: inclusive
        :return: success
        """
        start_date = datetime(start_date.year, start_date.month, start_date.day, 0, 0, 0)
        end_date = datetime(end_date.year, end_date.month, end_date.day, 23, 59, 59)
        if (end_date - start_date).days == -1:
            return False, None, None
        elif (end_date - start_date).days < -1:
            raise Exception('Unknown case')

        if not self.dbm.check_if_table_exists(constants.DAY_BAR_NAMING + ticker.lower()):
            self.dbm.create_table(constants.DAY_BAR_NAMING + ticker.lower(),
                                  {'dt': 'timestamp with time zone primary key',
                                   'open': 'real',
                                   'high': 'real',
                                   'low': 'real',
                                   'close': 'real',
                                   'volume': 'real',
                                   'volume_bloom': 'real',
                                   'volume_bloom_inf': 'real'})
            self.logger.info(f'Created table {constants.DAY_BAR_NAMING + ticker.lower()}')
        dt_ranges = []
        if (end_date - start_date).days > 365:
            st_dt = start_date
            ed_dt = datetime(start_date.year, start_date.month, start_date.day, 23, 59, 59) + timedelta(365)
            while ed_dt < end_date:
                dt_ranges.append((st_dt, ed_dt))
                st_dt = ed_dt + timedelta(1)
                st_dt = datetime(st_dt.year, st_dt.month, st_dt.day, 0, 0, 0)
                ed_dt = datetime(st_dt.year, st_dt.month, st_dt.day, 23, 59, 59) + timedelta(365)
            dt_ranges.append((st_dt, end_date))
        else:
            dt_ranges.append((datetime(start_date.year, start_date.month, start_date.day, 0, 0, 0),
                              datetime(end_date.year, end_date.month, end_date.day, 23, 59, 59)))
        inserted_data = False
        last_date = None
        bars_amnt = 0
        for dt_range in dt_ranges:
            success, bars = self.get_chunk(ticker, dt_range[0], dt_range[1])
            if not success:
                break
            bars = bars[['date', 'open', 'high', 'low', 'close', 'volume']]
            bars.columns = ['dt', 'open', 'high', 'low', 'close', 'volume']
            bars['dt'] = pd.to_datetime(bars['dt'], format='%Y-%m-%d %H:%M:%S')
            bars['dt'] = bars['dt'].dt.tz_localize(tz='US/Eastern')
            bars['dt'] = bars['dt'] + timedelta(hours=16)
            # with pd.option_context('display.max_rows', None, 'display.max_columns',
            #                        None):  # more options can be specified also
            #     print(bars['dt'])
            bars['volume_bloom'] = np.NAN
            bars['volume_bloom_inf'] = np.NAN

            self.dbm.insert_df_simple(bars, constants.DAY_BAR_NAMING + ticker.lower())
            inserted_data = True
            last_date = bars.dt.iloc[-1]
            bars_amnt += bars.shape[0]
        return inserted_data, last_date, bars_amnt

    def get_chunk(self, ticker: str, start_date: datetime, end_date: datetime):
        start_date = datetime(start_date.year,
                              start_date.month,
                              start_date.day,
                              0,  # start_date.hour,
                              0,  # start_date.minute,
                              0,  # start_date.second,
                              0,  # start_date.microsecond,
                              pytz.timezone('America/New_York'))
        end_date = datetime(end_date.year,
                            end_date.month,
                            end_date.day,
                            23,  # end_date.hour,
                            59,  # end_date.minute,
                            59,  # end_date.second,
                            0,   # end_date.microsecond,
                            pytz.timezone('America/New_York'))
        delta = (end_date - start_date).days
        assert delta <= 365
        delta = min(365, delta + 10)
        bars = self.ib.reqHistoricalData(Stock(ticker.upper().replace('_', ' '), 'SMART/ISLAND', 'USD'),
                                         endDateTime=end_date,
                                         durationStr=f'{delta} D',
                                         barSizeSetting='1 day',
                                         whatToShow='TRADES',
                                         useRTH=True,
                                         formatDate=2,
                                         timeout=30)
        bars = util.df(bars)
        if bars is None:
            return False, None
        bars = bars[(bars.date >= start_date.date()) & (bars.date <= end_date.date())]
        bars.reset_index(drop=True, inplace=True)
        return bars.shape[0] != 0, bars


# if __name__ == '__main__':
#     parser = OneDayRangeParser()
#     parser.parse_range('AA', datetime(2018, 1, 1), datetime(2021, 7, 1))
