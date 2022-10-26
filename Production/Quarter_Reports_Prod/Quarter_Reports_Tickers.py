#!/usr/bin/env python3

import logging
import sys, os
import csv
import requests

from datetime import timedelta, datetime

import pandas as pd
import pandas_market_calendars as mcal

sys.path.append('../..')
import config
from QuantumCapital import constants
from QuantumCapital.DBManager import DBManager
from QuantumCapital.TGManager import TGManager


# CONSTANTS
SCRIPT_NAME = 'Quarter Report Tickers'
LOG_FILE_PATH = 'Logs/Quarter_Reports_Tickers.log'  # Path for log file


QC_TOKEN = constants.QC_TOKEN
TWS_USR_ID = constants.TWS_USER_ID
ML_GROUP_ID = constants.QC_ML_GROUP_ID
LOGS_GROUP_ID = constants.QC_BOT_LOGS_GROUP_ID

DB_USERNAME = config.DB_USERNAME
DB_PASS = config.DB_PASS


# Configuring logging rules
a = logging.Logger.manager.loggerDict  # Disabling other loggers
for k in a:
    a[k].disabled = True
    
for handler in logging.root.handlers[:]:  # Disabling root logger handlers
    logging.root.removeHandler(handler)


rfh = logging.handlers.RotatingFileHandler(
    filename=LOG_FILE_PATH,
    mode='a',
    maxBytes=5*1024*1024,
    backupCount=2,
    encoding=None,
    delay=0
)

logging.basicConfig(handlers=[rfh],
    # filename=log_file_path,
                    format='%(asctime)s %(levelname)s [%(funcName)s]: %(message)s',
                    datefmt='%Y-%m-%d | %H:%M:%S',
                    level=logging.DEBUG)

# logging.getLogger().setLevel(logging.DEBUG)  # If you want logger write in console
logging.info('\n\n_______________________________________________________________________________________________________')


def intersection(lst1, lst2):
    lst3 = [value for value in lst1 if value in lst2]
    return list(set(lst3))


def get_edates_between(start, end):
    CSV_URL = 'https://www.alphavantage.co/query?function=EARNINGS_CALENDAR&horizon=3month&apikey=HIPEKPCQ0Z5WWNXF'

    with requests.Session() as s:
        download = s.get(CSV_URL)
        decoded_content = download.content.decode('utf-8')
        cr = csv.reader(decoded_content.splitlines(), delimiter=',')
        df = pd.DataFrame(cr)
        df.columns = df.iloc[0]
        df = df.iloc[1:]
        df = df[['symbol', 'reportDate']]
        df['reportDate'] = pd.to_datetime(df.reportDate)
        df = df[df.reportDate.isin(pd.date_range(start.date(), end.date()))]
    return df


def get_traded_vol_divided(window=31):
    tickers = dbm_barsday.select_df(f'select * from {constants.DAY_BAR_RUSSEL_PRESENCE_TABLE} '
                            f'order by yr desc limit 1').iloc[0].present
    small_cap = []
    large_cap = []
    for ticker in tickers:
        bars = dbm_barsday.select_df(f'select close, volume '
                             f'from {constants.DAY_BAR_NAMING + ticker} '
                             f'order by dt desc limit {window}')
        bars['price_vol'] = bars.close * bars.volume
        if (bars.close * bars.volume).median() < 100000:
            small_cap.append(ticker)
        else:
            large_cap.append(ticker)
    return large_cap, small_cap


def main():
    large_tickers, small_tickers = get_traded_vol_divided()

    today = datetime.today()
    logging.info('Current DT: %s', today)

    nyse = mcal.get_calendar('NYSE')
    r_days = nyse.valid_days(start_date=(today + timedelta(14)).strftime('%Y-%m-%d'),
                             end_date=(today + timedelta(25)).strftime('%Y-%m-%d'))[:2]

    start = r_days[0].to_pydatetime()
    end = r_days[-1].to_pydatetime()

    logging.debug("""Report dates inclusive:
                     Start: %s
                     End: %s""", start, end)

    earnings = get_edates_between(start, end)

    reporting_small_tickers = set(small_tickers) & set(earnings.symbol)
    reporting_large_tickers = set(large_tickers) & set(earnings.symbol)

    logging.debug('Len of large tickers list is: %s', len(reporting_large_tickers))
    logging.debug('Len of small tickers list is: %s', len(reporting_small_tickers))

    small_df = pd.DataFrame(reporting_small_tickers)
    large_df = pd.DataFrame(reporting_large_tickers)

    small_df.columns = ['ticker']
    large_df.columns = ['ticker']

    small_df.to_csv('reporting_small_tickers.csv', index=False)
    large_df.to_csv('reporting_large_tickers.csv', index=False)

    with open('reporting_small_tickers.csv', 'rb') as file:
        tgm.send_file(file, 'logs_group')
    with open('reporting_large_tickers.csv', 'rb') as file:
        tgm.send_file(file, 'logs_group')

    dbm_dwh.truncate_table('reporting_small_tickers')
    dbm_dwh.insert_df_simple(small_df, 'reporting_small_tickers')

    dbm_dwh.truncate_table('reporting_large_tickers')
    dbm_dwh.insert_df_simple(large_df, 'reporting_large_tickers')
    dbm_dwh.commit()
    logging.info('Saved tickers to DB and files.csv')
    sys.exit()


if __name__ == '__main__':  # This is for handling global errors
    logging.info('Reserved PID: %s', os.getpid())
    dbm_barsday = DBManager(DB_USERNAME, DB_PASS, 'BARS_DAY')
    dbm_dwh = DBManager(DB_USERNAME, DB_PASS, 'DWH')
    logging.info('Connected to DB successfully')
    tgm = TGManager(QC_TOKEN, TWS_USR_ID, ML_GROUP_ID, LOGS_GROUP_ID)
    logging.info('Connected to TG bot successfully')
    try:
        main()
    except Exception as e:
        dbm_dwh.rollback()
        tgm.send(SCRIPT_NAME + ': ' + repr(e), 'logs_group')
        logging.error('Error at %s', 'division', exc_info=e)
    logging.info('end of script___________________________________________________')
