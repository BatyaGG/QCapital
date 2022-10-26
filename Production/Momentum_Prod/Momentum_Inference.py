#!/usr/bin/env python3

import sys, os
import random
import math
import pickle as pk
# import threading
import logging

from collections import deque
from collections import defaultdict
from datetime import date, timedelta

import telegram
import numpy as np
import pandas as pd

from scipy import stats
from ib_insync import Stock, IB, util
from matplotlib import pyplot as plt

pd.set_option('display.max_columns', None)
sys.path.append('../..')
import config
from QuantumCapital import constants
from QuantumCapital.DBManager import DBManager
from QuantumCapital.TGManager import TGManager
from QuantumCapital.PlotManager import *

dots_n = 125
step = 1
rebalance_window = 5
vola_window = 20
assert dots_n >= vola_window + 1

cap_quantile = 0.25
momentum_top_n = 30
momentum_threshold = 40
com_reserve_perc = 1  # Commission reserve in percent
ma_window = 200
change_thresh = 0.2

# CONSTANTSt
SCRIPT_NAME = 'Momentum Inference'

LOG_FILE_PATH = 'Logs/Momentum_Inference.log'
DB_USERNAME = config.DB_USERNAME
DB_PASS = config.DB_PASS

DAY_BAR_RUSSEL_PRESENCE_TABLE = constants.DAY_BAR_RUSSEL_PRESENCE_TABLE
DAY_BAR_DICTIONARY_TABLE = constants.DAY_BAR_DICTIONARY_TABLE
DAY_BAR_NAMING = constants.DAY_BAR_NAMING

SERVER_IP = config.SERVER_IP
IB_PORT = config.IB_GW_PORT

QC_TOKEN = constants.QC_TOKEN
TWS_USR_ID = constants.TWS_USER_ID
ML_GROUP_ID = constants.QC_ML_GROUP_ID
LOGS_GROUP_ID = constants.QC_BOT_LOGS_GROUP_ID


# Configuring logging rules -------------------------------------------------------------- /
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


logging.basicConfig(
    handlers=[rfh],
    #filename=LOG_FILE_PATH,
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


def momentum_score(closes):
    log_closes = np.log(closes)
    slope, intercept, r_value, p_value, std_err = stats.linregress(np.arange(len(closes)), log_closes)
    annualized_slope = (np.power(np.exp(slope), 252) - 1) * 100
    score = annualized_slope * (r_value ** 2)
    return score, slope, intercept, r_value, annualized_slope


def cap_score(closes, volumes):
    # print(len(closes), len(volumes), rebalance_window)
    assert len(closes) == len(volumes) == rebalance_window
    avg_close = sum(closes) / rebalance_window
    avg_vol = sum(volumes) / rebalance_window
    return avg_close * avg_vol


def volatility_score(prices):
    prices = pd.Series(prices)
    return float(prices.pct_change().rolling(vola_window).std().iloc[-1])


def get_avail_tickers(dbm: DBManager):
    df = dbm.select_df(f'select * from {DAY_BAR_RUSSEL_PRESENCE_TABLE} order by yr')
    return df.iloc[-1].present


def get_data_db(dbm: DBManager, ticker: str, dots_n):
    df = dbm.select_df(f'select * from {DAY_BAR_NAMING}{ticker.upper()} order by dt desc limit {dots_n}')
    # print(ticker)
    # print(df)
    # print('-------------------------------------')
    # print()
    return df.sort_values(by=['dt']).reset_index(inplace=False, drop=True)


def get_data_ib(ib, ticker: str, dots_n):
    contract = Stock(ticker.upper().replace('_', ' '), 'SMART/ISLAND', 'USD')
    bars = ib.reqHistoricalData(
        contract,
        endDateTime='',
        durationStr=f'{dots_n} D',
        barSizeSetting='1 day',
        whatToShow='TRADES',
        useRTH=True,
        formatDate=2,
        timeout=60)
    idf = util.df(bars)
    return idf


def get_last_30min_bar(ib: IB, ticker: str):
    contract = Stock(ticker.upper().replace('_', ' '), 'SMART/ISLAND', 'USD')
    bars = ib.reqHistoricalData(
        contract,
        endDateTime='',
        durationStr='1 D',
        barSizeSetting='30 mins',
        whatToShow='TRADES',
        useRTH=True,
        formatDate=2,
        timeout=60)
    idf = util.df(bars)
    if idf is not None and idf.shape[0] > 0:
        return True, idf.iloc[-1]
    else:
        return False, None


def main(dbm: DBManager, tgm: TGManager):
    while True:
        try:
            logger.info('Attempt to connect to TWS')
            ib = IB()
            ib.connect(SERVER_IP, IB_PORT, clientId=4, readonly=True, timeout=90)
            logger.info('Connected to IB gateway successfully')
            tgm.send(f'Connected to IB\n{SCRIPT_NAME} in progress', 'logs_group')
            break
        except:
            logger.info(f'Could not launch {SCRIPT_NAME}\nWaiting for IB gateway re-login...')
            tgm.send(f'Could not launch {SCRIPT_NAME}\nWaiting for IB gateway re-login...', 'logs_group')
            tgm.tws_alert_block()

    tickers = get_avail_tickers(dbm)
    buy_tickers = {'ticker': [], 'price': [], 'momentum': [], 'cap': [], 'volatility': []}
    for ticker in tickers:
        df = get_data_db(dbm, ticker, dots_n)
        if df.shape[0] < dots_n:
            df = get_data_ib(ib, ticker, dots_n)
        if df is None or df.shape[0] < dots_n:
            logger.error(f'Not enough data for {ticker}')
            continue
        closes = df.close
        volumes = df.volume
        c_score = cap_score(closes.iloc[-rebalance_window:], volumes.iloc[-rebalance_window:])

        success, last_bar = get_last_30min_bar(ib, ticker)
        if not success:
            logger.info(f'{ticker} not found in IB')
            continue
        last_close = last_bar.close
        closes.iloc[-1] = last_close
        v_score = volatility_score(closes.iloc[-vola_window-1:])
        m_score = momentum_score(closes)[0]

        buy_tickers['ticker'].append(ticker)
        buy_tickers['price'].append(closes.iloc[-1])
        buy_tickers['momentum'].append(m_score)
        buy_tickers['cap'].append(c_score)
        buy_tickers['volatility'].append(v_score)
    buy_tickers = pd.DataFrame(buy_tickers)
    cap_thresh = buy_tickers.cap.quantile(cap_quantile)
    buy_tickers = buy_tickers[buy_tickers.cap > cap_thresh]
    buy_tickers.sort_values(by='momentum', inplace=True, ascending=False)
    buy_tickers = buy_tickers.iloc[:momentum_top_n]
    buy_tickers['inv_vola'] = 1 / buy_tickers.volatility
    inv_vola_sum = buy_tickers.inv_vola.sum()
    buy_tickers['inv_vola_norm'] = buy_tickers['inv_vola'] / inv_vola_sum
    buy_tickers = buy_tickers[buy_tickers.momentum > momentum_threshold]
    buy_tickers.reset_index(drop=True, inplace=True)
    print(buy_tickers[['ticker', 'price', 'momentum', 'cap', 'inv_vola_norm']])
    logger.info(buy_tickers[['ticker', 'price', 'momentum', 'cap', 'inv_vola_norm']])
    buy_tickers[['ticker', 'price', 'momentum', 'cap', 'inv_vola_norm']].to_csv(
        'momentum_tickers.csv', index=False)
    with open('momentum_tickers.csv', 'rb') as file:
        tgm.send_file(file, 'logs_group')
        logger.info(f'Sent results file to TG group')
    ib.disconnect()
    sys.exit()


if __name__ == '__main__':
    dbm = DBManager(DB_USERNAME, DB_PASS, 'BARS_DAY')
    logger.info('Connected to DB successfully')
    tgm = TGManager(QC_TOKEN, TWS_USR_ID, ML_GROUP_ID, LOGS_GROUP_ID)
    logger.info('Connected to TG bot successfully')
    try:
        main(dbm, tgm)
    except Exception as e:
        dbm.rollback()
        tgm.send(SCRIPT_NAME + ': ' + repr(e), 'logs_group')
        logger.error('Error at %s', 'division', exc_info=e)
    logger.info('end of script___________________________________________________')
