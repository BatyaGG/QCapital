import datetime

import pytz
from datetime import date, timedelta
import sys

import pandas as pd
import pandas_market_calendars as mcal
from ib_insync import Stock, IB, util, Client

sys.path.append('../..')
import config
from QuantumCapital import constants
from QuantumCapital.DBManager import DBManager
from QuantumCapital.TGManager import TGManager


SERVER_IP = config.SERVER_IP
IB_GW_PORT = config.IB_GW_PORT
IB_TWS_PORT = config.IB_TWS_PORT

DB_USERNAME = config.DB_USERNAME
DB_PASS = config.DB_PASS

QC_TOKEN = constants.QC_TOKEN
TWS_USR_ID = constants.TWS_USER_ID
ML_GROUP_ID = constants.QC_ML_GROUP_ID
LOGS_GROUP_ID = constants.QC_BOT_LOGS_GROUP_ID


class DataEngine:
    def __init__(self, client_id):
        self.dbm = DBManager(DB_USERNAME, DB_PASS, 'BARS_DAY')
        tgm = TGManager(QC_TOKEN, TWS_USR_ID, ML_GROUP_ID, LOGS_GROUP_ID)
        while True:
            try:
                for port in (IB_GW_PORT, IB_TWS_PORT):
                    try:
                        self.ib = IB()
                        self.ib.connect(SERVER_IP, port, clientId=client_id, readonly=True, timeout=90)
                        break
                    except:
                        if port == IB_GW_PORT:
                            continue
                        raise Exception()
                break
            except:
                tgm.send('Could not launch DataEngine\nWaiting for IB gateway re-login...', 'logs_group')
                tgm.tws_alert_block()

    def _get_stock_data(self, tick, time_span='1 D', barSize='1 day', asof=''):

        contract = Stock(tick.upper(), 'SMART/ISLAND', 'USD')
        bars = self.ib.reqHistoricalData(
            contract,
            endDateTime=asof,
            durationStr=time_span,
            barSizeSetting=barSize,
            whatToShow='TRADES',
            useRTH=True,
            formatDate=2,
            timeout=60)
        if util.df(bars) is not None:
            idf = util.df(bars).set_index('date')
            return idf

    @staticmethod
    def _hour_add(x):
        return x.replace(hour=16)

    def get_bars(self, ticker: str, start_date: date, end_date: date, local_time: bool, which: str = 'best'):
        assert which in ('best', 'ib', 'db')
        nyse = mcal.get_calendar('NYSE')
        r_days = nyse.valid_days(start_date=start_date.strftime('%Y-%m-%d'),
                                 end_date=end_date.strftime('%Y-%m-%d'))
        if which in ('best', 'db'):
            start_date = r_days[0].date()
            end_date = min(r_days[-1].date(), date.today() - timedelta(1))
            try:
                df = self.dbm.select_df(f'select * from bars_1_day_{ticker.lower()} '
                                        f"where dt between \'{start_date.strftime('%Y-%m-%d')} 00:00:00+00:00\'"
                                        f"and \'{end_date.strftime('%Y-%m-%d')} 23:59:59+00:00\'")
                db_start_date = df.iloc[0]['dt'].date()
                db_end_date = df.iloc[-1]['dt'].date()
                df['dt'] = df['dt'].dt.tz_convert('Asia/Almaty')
            except:
                db_start_date = None
                db_end_date = None
                df = None
            if start_date == db_start_date and end_date == db_end_date:
                print('fetched from DB')
                if not local_time:
                    df['dt'] = df['dt'].dt.tz_convert('America/New_York')
                return df[['dt', 'open', 'high', 'low', 'close', 'volume']]
        if which in ('best', 'ib'):
            df = self._get_stock_data(ticker, time_span=f'{len(r_days)} D', asof=end_date)
            if df is not None:
                df['dt'] = df.index
                df.index = list(range(df.shape[0]))

                df['dt'] = pd.to_datetime(df['dt'])
                df['dt'] = df['dt'].dt.tz_localize('America/New_York') + pd.Timedelta(hours=16)
                if local_time:
                    df['dt'] = df['dt'].apply(self._hour_add)
                    df['dt'] = df['dt'].dt.tz_convert('Asia/Almaty')
                print(f'fetched from IB')
                return df[['dt', 'open', 'high', 'low', 'close', 'volume']]
            print('No data in IB')

    def __del__(self):
        del self.dbm
        self.ib.disconnect()


if __name__ == '__main__':
    de = DataEngine(10)
    df = de.get_bars('NVDA', date(2021, 1, 18), date(2021, 9, 28), False)
    print(df)

    # -4 - 16:00
