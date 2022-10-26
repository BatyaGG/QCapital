#!/usr/bin/env python3

import sys
import os
import pickle as pk
import time
import logging
import logging.config
import pytz
from datetime import datetime, timedelta, date
from datetime import time as time_obj
from fractions import Fraction

import psycopg2 as pg
import pandas as pd
import pandas_market_calendars as mcal
import telegram
from ib_insync import Stock, IB, util

from OneDayRangeParser import OneDayRangeParser


sys.path.append('../..')
import config
from QuantumCapital import constants
from QuantumCapital.DBManager import DBManager
from QuantumCapital.TGManager import TGManager

# CONSTANTS
SCRIPT_NAME = 'One Day Bar Parser'
LOG_FILE_PATH = '../Logs/One_day_bars_russel.log'

QC_TOKEN = constants.QC_TOKEN
TWS_USR_ID = constants.TWS_USER_ID
ML_GROUP_ID = constants.QC_ML_GROUP_ID
LOGS_GROUP_ID = constants.QC_BOT_LOGS_GROUP_ID

SERVER_IP = config.SERVER_IP
IB_GW_PORT = config.IB_GW_PORT
IB_TWS_PORT = config.IB_TWS_PORT

DB_USERNAME = config.DB_USERNAME
DB_PASS = config.DB_PASS

# Configuring logging rules -------------------------------------------------------------- /
a = logging.Logger.manager.loggerDict  # Disabling other loggers
for k in a:
    a[k].disabled = True

for handler in logging.root.handlers[:]:  # Disabling root logger handlers
    logging.root.removeHandler(handler)

rfh = logging.handlers.RotatingFileHandler(
    filename=LOG_FILE_PATH,
    mode='a',
    maxBytes=2*1024*1024,
    backupCount=2,
    encoding=None,
    delay=0
)

logging.basicConfig(handlers=[rfh],
    # filename=LOG_FILE_PATH,
                    format='%(asctime)s %(levelname)s [%(funcName)s]: %(message)s',
                    datefmt='%Y-%m-%d | %H:%M:%S',
                    level=logging.ERROR)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

requests_logger = logging.getLogger('requests')
requests_logger.setLevel(logging.ERROR)

handler = logging.StreamHandler()
handler.setLevel(logging.ERROR)
logger.addHandler(handler)
requests_logger.addHandler(handler)

# logging.getLogger().setLevel(logging.DEBUG)  # If you want logger write in console
logger.info(
    '\n\n_______________________________________________________________________________________________________')
# --------------------------------------------------------------------------------------- /


def get_russel_year(today: datetime):
    if 7 <= today.month <= 12:
        return today.year
    else:
        return today.year - 1


def ticker_name_parser(ticker: str) -> str:
    return ticker.split()[0].upper()


def get_curr_year_tickers(dbm, year):
    tickers = dbm.select_df('select tickers '
                            'from rus_tickers_historical '
                            f'where yr = {year} order by tickers')
    tickers = tickers.iloc[0].tickers
    tickers = [ticker_name_parser(ticker) for ticker in tickers]
    tickers = [ticker.replace('/', '_') for ticker in tickers]
    return tickers


def ticker_exists_in_db(dbm, ticker):
    dict_val = dbm.select_df(f'select ticker from {constants.DAY_BAR_DICTIONARY_TABLE} where ticker = \'{ticker.upper()}\'')
    db_info_val = dbm.select_df("select table_name from information_schema.tables "
                                f"where table_schema=\'public\' and table_name=\'{constants.DAY_BAR_NAMING + ticker.lower()}\'")
    # print(dict_val)
    # print(f'select ticker from {constants.DAY_BAR_DICTIONARY_TABLE} where ticker = \'{ticker.upper()}\'')
    # print(db_info_val)
    if dict_val.shape[0] == 1 and db_info_val.shape[0] == 1:
        return True
    elif dict_val.shape[0] == 0 and db_info_val.shape[0] == 0:
        return False
    else:
        raise Exception('Dictionary doesnt match')


def get_last_date(dbm, ticker):
    dict_val = dbm.select_df(f"select ticker, last_date AT TIME ZONE \'America/New_York\' as last_date from {constants.DAY_BAR_DICTIONARY_TABLE} where ticker = \'{ticker.upper()}\'")
    return dict_val.iloc[0].last_date


def create_record_in_dict(dbm: DBManager, ticker: str):
    df = {'ticker': [ticker.upper()], 'table_name': [constants.DAY_BAR_NAMING + ticker.lower()],
          'years': [[]], 'bars_count': [0], 'memory': [0]}
    dbm.insert_df_simple(pd.DataFrame(df), constants.DAY_BAR_DICTIONARY_TABLE)


def update_record_in_dict(dbm: DBManager, ticker: str, new_bars_count: int, last_date: datetime, rus_year: int):
    """
    Must be called after ticker table update
    :param dbm:
    :param ticker:
    :param new_bars_count:
    :return:
    """
    df = dbm.select_df(f"""select pg_relation_size('"'||table_schema||'"."'||table_name||'"')
                           from information_schema.tables
                           where table_schema = 'public' and table_name = \'{constants.DAY_BAR_NAMING + ticker.lower()}\'""")
    assert df.shape == (1, 1), 'df shape must be (1, 1)'
    size = 9.3132e-10 * df.iloc[0, 0]
    size = round(size, 6)

    df = dbm.select_df(f"""select years from {constants.DAY_BAR_DICTIONARY_TABLE}
                               where ticker = \'{ticker}\'""")
    years = set(df.iloc[0].years)
    years.add(rus_year)
    years = list(years)
    years.sort()
    dbm.update_table_row(constants.DAY_BAR_DICTIONARY_TABLE,
                         {'last_date': f"\'{last_date}\'",
                          'bars_count': f'bars_count + {new_bars_count}',
                          'years': "\'{" + ', '.join([str(y) for y in years]) + "}\'",
                          'memory': size},
                         {'ticker': f"=\'{ticker.upper()}\'"})


def update_presence(dbm: DBManager, ticker: str, year: int, which: str):
    assert which in ('present', 'absent'), 'which argument must be one of (present, absent)'
    presence = dbm.select_df(f'select * from {constants.DAY_BAR_RUSSEL_PRESENCE_TABLE} where yr={year}')
    assert presence.shape[0] <= 1, 'More than 2 rows of same year'
    if presence.shape[0] == 0:
        row = pd.DataFrame([{'yr': year,
                             which: [ticker],
                             'absent' if which == 'present' else 'present': []}])
        dbm.insert_df_simple(row, constants.DAY_BAR_RUSSEL_PRESENCE_TABLE)
    else:
        if which == 'present':
            # assert ticker not in presence.iloc[0].absent, f'{ticker} ticker was already absent'

            if ticker not in presence.iloc[0].present:
                dbm.update_table_row(constants.DAY_BAR_RUSSEL_PRESENCE_TABLE,
                                     {which: f"array_append({which}, \'{ticker}\')"}, {"yr=": year})
        else:
            # print(presence)
            # print(presence.iloc[0])
            # assert ticker not in presence.iloc[0].present, f'{ticker} ticker was already present'
            if ticker not in presence.iloc[0].absent and ticker not in presence.iloc[0].present:
                dbm.update_table_row(constants.DAY_BAR_RUSSEL_PRESENCE_TABLE,
                                     {which: f"array_append({which}, \'{ticker}\')"}, {"yr=": year})


def align_splits(dbm, ticker, old_last_date, parser):
    check_window = 25
    # print('-------------------------------------------')
    # print(ticker)
    _, split_adjusted_df = parser.get_chunk(ticker, (old_last_date - timedelta(check_window)), old_last_date)
    old_df = dbm.select_df(f'select * from bars_1_day_{ticker.lower()} '
                           f"where dt between \'{(old_last_date - timedelta(check_window)).strftime('%Y-%m-%d')} 00:00:00+00:00\'"
                           f"and \'{old_last_date.strftime('%Y-%m-%d')} 23:59:59+00:00\'")
    split_adjusted_df['date'] = pd.to_datetime(split_adjusted_df.date)
    old_df['dt'] = pd.to_datetime(old_df['dt']).dt.tz_localize(None)
    old_df['dt'] = old_df['dt'].dt.date
    old_df['dt'] = pd.to_datetime(old_df['dt'])

    # print(split_adjusted_df.dtypes)
    # print(old_df.dtypes)
    df = split_adjusted_df.merge(old_df, how='left', left_on='date', right_on='dt', suffixes=('_left', '_right'))
    # with pd.option_context('display.max_rows', None, 'display.max_columns',
    #                        None):  # more options can be specified also
    #     # print(split_adjusted_df, '\n', old_df)
    #     print(df)
    ratios = pd.concat([(df.open_left / df.open_right),
                       (df.high_left / df.high_right),
                       (df.low_left / df.low_right),
                       (df.close_left / df.close_right)],
                       axis=0)
    ratio = round(ratios.value_counts().index[0], 5)

    if ratio != 1.0:
        ratio_frac = Fraction(ratio).limit_denominator(10)
        print(f'{ticker} split: {ratio_frac}')
        logger.info(f'{ticker} split: {ratio_frac}')
        dbm.update_table_row(f'bars_1_day_{ticker.lower()}',
                             {
                                 'open': f'ROUND((open * {ratio_frac.numerator} / {ratio_frac.denominator})::numeric, 2)',
                                 'high': f'ROUND((high * {ratio_frac.numerator} / {ratio_frac.denominator})::numeric, 2)',
                                 'low': f'ROUND((low * {ratio_frac.numerator} / {ratio_frac.denominator})::numeric, 2)',
                                 'close': f'ROUND((close * {ratio_frac.numerator} / {ratio_frac.denominator})::numeric, 2)'
                             },
                             {'dt': f'<= \'{old_last_date} 23:59:59-04\''})


def main(dbm, tgm, rus_year):
    while True:
        try:
            for port in (IB_GW_PORT, IB_TWS_PORT):
                try:
                    ib = IB()
                    ib.connect(SERVER_IP, port, clientId=1, readonly=True, timeout=90)
                    break
                except:
                    if port == IB_GW_PORT:
                        continue
                    raise Exception()
            logger.info('Connected to IB gateway successfully')
            tgm.send(f'Connected to IB\n{SCRIPT_NAME} in progress', 'logs_group')
            break
        except:
            tgm.send('Could not launch DataEngine\nWaiting for IB gateway re-login...', 'logs_group')
            tgm.tws_alert_block()

    today = datetime.now(tz=pytz.timezone('America/New_York'))
    rth_now = time_obj(9, 30, 0) <= today.time() <= time_obj(16, 0, 0)
    today = today.date()
    parser = OneDayRangeParser(dbm, ib, logger)
    tickers = get_curr_year_tickers(dbm, rus_year)
    # tickers = tickers[tickers.index('THG'):]
    # tickers = ['THG']
    for ticker in tickers:
        ticker_in_db = ticker_exists_in_db(dbm, ticker)
        if ticker_in_db:
            old_last_date = get_last_date(dbm, ticker).date()  # ny time
        else:
            old_last_date = datetime(1970, 1, 1).date()
        start_from = max(old_last_date + timedelta(1), date(rus_year, 7, 1))
        end_at = min(today - timedelta(1) if rth_now else today, date(rus_year + 1, 6, 30))
        success, last_date, bars_amnt = parser.parse_range(ticker,
                                                           start_from,
                                                           end_at)
        if ticker_in_db:
            if success:
                align_splits(dbm, ticker, old_last_date, parser)

                update_record_in_dict(dbm, ticker, bars_amnt, last_date, rus_year)
                update_presence(dbm, ticker, rus_year, 'present')
                dbm.commit()
            else:
                if bars_amnt is None:
                    logger.info(f'{ticker} already exists for {rus_year}')
                    dbm.rollback()
                else:
                    # logger.error(f'{ticker} already up to date')
                    dbm.rollback()
                    update_presence(dbm, ticker, rus_year, 'absent')
                    dbm.commit()
        else:
            if success:
                create_record_in_dict(dbm, ticker)
                update_record_in_dict(dbm, ticker, bars_amnt, last_date, rus_year)
                update_presence(dbm, ticker, rus_year, 'present')
                dbm.commit()
            else:
                if bars_amnt is None:  # Already in database
                    dbm.rollback()
                    raise Exception('Impossible')
                else:  # Failed to fetch from IB
                    # logger.error(f'{ticker} not found in IB for {rus_year}')
                    dbm.rollback()
                    update_presence(dbm, ticker, rus_year, 'absent')
                    dbm.commit()
    ib.disconnect()
    # sys.exit()


if __name__ == '__main__':
    logger.info('Reserved PID: %s', os.getpid())
    dbm = DBManager(DB_USERNAME, DB_PASS, 'BARS_DAY')
    logger.info('Connected to DB successfully')
    tgm = TGManager(QC_TOKEN, TWS_USR_ID, ML_GROUP_ID, LOGS_GROUP_ID)
    logger.info('Connected to TG bot successfully')
    try:
        ny_today = datetime.now(pytz.timezone('America/New_York'))
        main(dbm, tgm, get_russel_year(ny_today))
        # for i in range(2014, 2022):
        #     print(i)
        #     main(dbm, tgm, i)
    except Exception as e:
        dbm.rollback()
        tgm.send(SCRIPT_NAME + ': ' + repr(e), 'logs_group')
        logger.error('Error at %s', 'division', exc_info=e)
    logger.info('end of script___________________________________________________')
