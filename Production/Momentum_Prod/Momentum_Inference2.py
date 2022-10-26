#!/usr/bin/env python3

import sys, os
import logging
import argparse

from datetime import date, timedelta

import numpy as np
import pandas as pd
import torch


pd.set_option('display.max_columns', None)
sys.path.append('../..')
import config
from DataParsers.DataEngine import DataEngine
from QuantumCapital import constants
from QuantumCapital.DBManager import DBManager
from QuantumCapital.TGManager import TGManager
from QuantumCapital.PlotManager import *

dots_n = 105
step = 5
vola_window = 160

cap_quantile = 0.25
momentum_top_n = 20
top_n_to_show = 50
momentum_threshold = 40
com_reserve_perc = 0  # Commission reserve in percent
ma_window = 200
change_thresh = 0.0
close_all_at_low_ma = False
use_trailing_stop = True
trailing_stop_window = 180
trailing_stop_coeff = 4

# CONSTANTS
SCRIPT_NAME = 'Momentum Inference V2'

LOG_FILE_PATH = 'Logs/Momentum_Inference_V2.log'
DB_USERNAME = config.DB_USERNAME
DB_PASS = config.DB_PASS

DAY_BAR_RUSSEL_PRESENCE_TABLE = constants.DAY_BAR_RUSSEL_PRESENCE_TABLE
DAY_BAR_DICTIONARY_TABLE = constants.DAY_BAR_DICTIONARY_TABLE
DAY_BAR_NAMING = constants.DAY_BAR_NAMING

QC_TOKEN = constants.QC_TOKEN
TWS_USR_ID = constants.TWS_USER_ID
ML_GROUP_ID = constants.QC_ML_GROUP_ID
LOGS_GROUP_ID = constants.QC_BOT_LOGS_GROUP_ID

if not torch.cuda.is_available():
    raise Exception('No GPU')

gpu = torch.device("cuda:0")


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


def momentum_score_calc(price_df):
    price_df = torch.from_numpy(price_df.to_numpy()).to(gpu)
    x = torch.arange(dots_n).view(dots_n, 1).repeat(1, price_df.shape[1]).to(gpu)
    price_df_log = torch.log(price_df[-dots_n:, :])
    x_mean = (x * 1.0).mean(axis=0)
    x_minus_xmean = x - x_mean
    price_df_log_mean = price_df_log.mean(axis=0)
    price_df_log_minus_mean = price_df_log - price_df_log_mean
    num = (x_minus_xmean * price_df_log_minus_mean).sum(axis=0)
    den = (x_minus_xmean ** 2).sum(axis=0)
    b = num / den
    a = price_df_log_mean - b * x_mean

    yhat = b * x + a

    annualized_slope = (torch.pow(torch.exp(b), 252) - 1) * 100

    num = ((yhat - price_df_log_mean) ** 2).sum(axis=0)
    den = ((price_df_log - price_df_log_mean) ** 2).sum(axis=0)
    r_value = num / den

    score = annualized_slope * r_value
    return score.cpu().numpy()


def cap_score_calc(price_df, vol_df):
    price_df = torch.from_numpy(price_df.to_numpy()).to(gpu)
    vol_df = torch.from_numpy(vol_df.to_numpy()).to(gpu)
    return (price_df[-step:, :] * vol_df[-step:, :]).sum(axis=0).cpu().numpy()


def vola_score_calc(price_df, window):
    price_df = torch.from_numpy(price_df.to_numpy()).to(gpu)
    pct_change = (price_df[-window:, :] - price_df[-window - 1:-1, :]) / price_df[-window - 1:-1, :]
    std = torch.std(pct_change, 0)
    return std.cpu().numpy()


def get_avail_tickers(dbm: DBManager):
    df = dbm.select_df(f'select * from {DAY_BAR_RUSSEL_PRESENCE_TABLE} order by yr')
    return df.iloc[-1].present


def main(dbm, tgm, parse_method):
    data_parser = DataEngine(5)
    tickers = get_avail_tickers(dbm)
    fetched_tickers = []
    prices = []
    volumes = []

    dots_number = max(dots_n, vola_window + 1, trailing_stop_window + 1)

    spy = data_parser.get_bars('SPY', datetime.today() - timedelta(dots_number * 2), datetime.today(), False)
    dates = spy[['dt']]

    for i, ticker in enumerate(tickers):
        print(round(100 * (i + 1) / len(tickers), 2), '% done')
        data = data_parser.get_bars(ticker, datetime.today() - timedelta(dots_number * 2), datetime.today(), False,
                                    which=parse_method)
        if data is None:
            continue
        data = data[['dt', 'close', 'volume']]
        data = dates.merge(data, how='left', on='dt')

        data.fillna(method='ffill', inplace=True)
        data.set_index('dt', drop=True, inplace=True)

        fetched_tickers.append(ticker)
        prices.append(data['close'])
        volumes.append(data['volume'])

    price_df = pd.concat(prices, axis=1).iloc[-dots_number:]
    vol_df = pd.concat(volumes, axis=1).iloc[-dots_number:]

    momentum_scores = momentum_score_calc(price_df)
    cap_scores = cap_score_calc(price_df, vol_df)
    vola_scores_alloc = vola_score_calc(price_df, vola_window)
    vola_scores_trlstp = vola_score_calc(price_df, trailing_stop_window)

    buy_tickers = {'ticker': fetched_tickers,
                   'price': price_df.iloc[-1, :],
                   'momentum': momentum_scores,
                   'cap': np.round(cap_scores, 3),
                   'volatility': vola_scores_alloc,
                   'volatility_trlstp': vola_scores_trlstp}
    buy_tickers = pd.DataFrame(buy_tickers)
    buy_tickers.sort_values(by='momentum', inplace=True, ascending=False)
    buy_tickers['momentum'] = round(buy_tickers.momentum, 3)
    buy_tickers.index = list(range(buy_tickers.shape[0]))
    buy_tickers = buy_tickers.iloc[:top_n_to_show]

    buy_tickers['inv_vola'] = 1 / buy_tickers.volatility
    inv_vola_sum = buy_tickers.inv_vola.sum()
    buy_tickers['inv_vola_norm'] = round(buy_tickers['inv_vola'] / inv_vola_sum, 3)
    inv_vola_sum_top20 = buy_tickers.iloc[:momentum_top_n].inv_vola.sum()
    buy_tickers[f'inv_vola_norm_top{momentum_top_n}'] = round(buy_tickers['inv_vola'] / inv_vola_sum_top20, 3)
    buy_tickers.loc[list(range(momentum_top_n, top_n_to_show)), f'inv_vola_norm_top{momentum_top_n}'] = None

    buy_tickers['stop_at'] = round(buy_tickers.price - buy_tickers.price * trailing_stop_coeff * buy_tickers.volatility_trlstp, 3)
    buy_tickers[
        ['ticker', 'momentum', 'price', 'stop_at', 'cap', 'inv_vola_norm', f'inv_vola_norm_top{momentum_top_n}']
            ].to_csv(f'momentum_tickers_V{version}.csv', index=False)
    with open(f'momentum_tickers_V{version}.csv', 'rb') as file:
        tgm.send_file(file, 'logs_group')
        logger.info(f'Sent results file to TG group')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("parse_method")
    parser.add_argument("dots_n")
    parser.add_argument("vola_win")
    parser.add_argument("v")
    args = parser.parse_args()
    dots_n = int(args.dots_n)
    vola_win = int(args.vola_win)
    version = args.v

    logger.info('Reserved PID: %s', os.getpid())
    dbm = DBManager(DB_USERNAME, DB_PASS, 'BARS_DAY')
    logger.info('Connected to DB successfully')
    tgm = TGManager(QC_TOKEN, TWS_USR_ID, ML_GROUP_ID, LOGS_GROUP_ID)
    logger.info('Connected to TG bot successfully')
    try:
        main(dbm, tgm, args.parse_method)
    except Exception as e:
        dbm.rollback()
        tgm.send(SCRIPT_NAME + ': ' + repr(e), 'logs_group')
        logger.error('Error at %s', 'division', exc_info=e)
    logger.info('end of script___________________________________________________')
