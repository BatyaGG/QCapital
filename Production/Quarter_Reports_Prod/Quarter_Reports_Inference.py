#!/usr/bin/env python3

import sys, os
import math
import pickle
import warnings
import logging
import logging.config
warnings.filterwarnings("ignore")

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import telegram
from sklearn.linear_model import LinearRegression

from tabulate import tabulate

from ib_insync import Stock, IB, util

sys.path.append('../..')
import config
from QuantumCapital import constants
from QuantumCapital.DBManager import DBManager
from QuantumCapital.TGManager import TGManager


# CONSTANTS
SCRIPT_NAME = 'Quarter Reports Inference'
LOG_FILE_PATH = 'Logs/Quarter_Reports_Inference.log'

QC_TOKEN = constants.QC_TOKEN
TWS_USR_ID = constants.TWS_USER_ID
ML_GROUP_ID = constants.QC_ML_GROUP_ID
LOGS_GROUP_ID = constants.QC_BOT_LOGS_GROUP_ID

MODELS_PATH = '../../Models_&_Files/Quarter_Reports_Models/'

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

rfh = logging.handlers.RotatingFileHandler(filename=LOG_FILE_PATH,
                                           mode='a',
                                           maxBytes=5*1024*1024,
                                           backupCount=2,
                                           encoding=None,
                                           delay=0)

logging.basicConfig(handlers=[rfh],
    # filename=[rfh],
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


def get_stock_data(ib, tick, time_span='1 D', barSize='1 day'):
    contract = Stock(tick, 'SMART/ISLAND', 'USD')
    bars = ib.reqHistoricalData(
        contract,
        endDateTime='',
        durationStr=time_span,
        barSizeSetting=barSize,
        whatToShow='TRADES',
        useRTH=True,
        formatDate=2,
        timeout=60)

    idf = util.df(bars).set_index('date')
    return idf[['close', 'volume']]


def compress(series, sides=0.001):
    series = pd.Series(np.where(series > series.quantile(q=(1 - sides)), series.quantile(q=(1 - sides)), series),
                       index=series.index)
    series = np.where(series < series.quantile(q=sides), series.quantile(q=sides), series)
    return series


def compress_right_tail(series, sides=0.01):
    series = pd.Series(np.where(series > series.quantile(q=(1 - sides)), series.quantile(q=(1 - sides)), series),
                       index=series.index)
    return series


def create_features(dtf, tick, lead=5,
                    ret_lag=[1, 2, 5, 10, 22, 44, 51, 66, 118, 132, 246, 261, 375, 480, 520],
                    tr=0.02,
                    vol_lag=[1, 2, 5, 10, 22, 44, 51, 66, 118, 132],
                    vol_window=100, sides=0.01):
    feat_cols = []
    avg_vol_name = 'avg_vol_' + str(vol_window)
    dtf[avg_vol_name] = dtf['volume'].rolling(window=vol_window).mean()
    dtf[avg_vol_name] = np.log(dtf[avg_vol_name])

    for lag in vol_lag:
        name_simple = 'vol' + str(lag)
        dtf[name_simple] = dtf['volume'].rolling(window=lag).mean()
        dtf[name_simple] = np.log(dtf[name_simple])
        dtf[name_simple] = dtf[name_simple] / dtf[avg_vol_name]
        dtf[name_simple] = compress(dtf[name_simple], sides=sides)
        dtf[name_simple] = (dtf[name_simple] - dtf[name_simple].cummin()) / (
                    dtf[name_simple].cummax() - dtf[name_simple].cummin())
        feat_cols.append(name_simple)
    for lag in ret_lag:
        name = 'ret' + str(lag)
        dtf[name] = dtf['close'] / dtf['close'].shift(lag) - 1
        dtf[name] = compress(dtf[name], sides=sides)
        dtf[name] = (dtf[name] - dtf[name].cummin()) / (dtf[name].cummax() - dtf[name].cummin())
        feat_cols.append(name)

    dtf['tick'] = tick
    dtf['fwd'] = dtf['close'].shift(-lead - 1) / dtf['close'] - 1
    dtf['ycol'] = np.where(dtf['fwd'] >= tr, 1, 0)
    # feat_cols.append('ycol')
    return dtf, feat_cols


def ticker_features(combined_dtfs, ticker, growth_tr=0.02, back_window=28, b4rep_window=10,
                    ret_lag=(1, 2, 5, 10, 20, 44, 261), vol_lag=(1, 2, 5, 10, 20, 44, 261), vol_window=100, sides=0.01):
    idf = combined_dtfs[ticker]
    if combined_dtfs[ticker].shape[0] == 0:
        return
    else:
        idf['idate'] = idf.index
        idf['idate'] = idf['idate'].shift(1)
        idf['b4rep_window'] = idf.index
        idf['b4rep_window'] = idf['b4rep_window'].shift(b4rep_window + 1)
        idf['back_window'] = pd.to_datetime(np.where(idf.rep_date == 1, idf['idate'] - \
                                                     pd.Timedelta(days=back_window), pd.to_datetime(np.nan)))
        idf['back_window'] = idf['back_window'].fillna(method='backfill')

        idf['b4rep_window'] = pd.to_datetime(np.where(idf.rep_date == 1, idf['b4rep_window'], pd.to_datetime(np.nan)))
        idf['b4rep_window'] = idf['b4rep_window'].fillna(method='backfill')

        idf['b4rep'] = np.where(idf.idate >= idf.back_window, 1, 0)
        idf['next_rep'] = pd.to_datetime(np.where(idf.rep_date == 1, idf.idate, pd.to_datetime(np.nan)))
        idf['next_rep'] = idf['next_rep'].fillna(method='backfill')
        idf['price_atrep'] = idf['next_rep'].map(idf[ticker])
        idf['price_b4rep'] = idf['b4rep_window'].map(idf[ticker])
        idf['b4rep_chng'] = idf['price_atrep'] / idf['price_b4rep'] - 1
        idf, feat_cols = create_features(idf, ticker, lead=b4rep_window, ret_lag=ret_lag, vol_lag=vol_lag, tr=growth_tr,
                                         vol_window=vol_window, sides=sides)
        feat_cols.append('fwd')
        feat_cols.append('tick')
        idf = idf[idf.idate == idf.b4rep_window]

        return idf[feat_cols].dropna()


def nasdaq_features(combined, params=None):
    if params == None:
        params = {
            'b4rep_window': 10,
            'growth_tr': 0.02,
            'back_window': 28,
            'ret_lag': [1, 2, 5, 10, 22, 44, 51, 66, 118, 132, 246, 261, 375, 480, 520],
            'vol_lag': [1, 2, 5, 10, 22, 44, 51, 66, 118, 132],

            'vol_window': 100,
            'sides': 0.01
        }

    df = pd.DataFrame()
    for i in combined:
        df = df.append(ticker_features(combined, i, **params))
    df = df.sort_index()

    return df


def exp_function(intercept, slope, X):
    res = []
    for x in X:
        try:
            val = math.exp(intercept + slope * x)
        except OverflowError:
            val = None
        res.append(val)
    max_val = max([r if r is not None else float('-inf') for r in res])
    res = np.array([r if r is not None else max_val for r in res])
    res_mean = np.mean(res)
    res_std = np.std(res)

    res2 = []
    inliers = abs(res - res_mean) < 2 * res_std
    for r, inlier in zip(res, inliers):
        if inlier:
            res2.append(r)
        else:
            if r >= res_mean:
                res2.append(res_mean + 2 * res_std)
            else:
                res2.append(res_mean - 2 * res_std)
    return res2


def fill_df(row):
    if np.isnan(row.close_x):
        row['close_x'] = row.close_y
        row['volume_x'] = row.volume_y
    return None


def preprocess_tickers(tickers, combined, models, ib):
    features_df = pd.DataFrame()
    logger.info('Starting preprocessing TWS data for %s tickers', len(tickers))
    for tick in tickers:
        # print(tick)
        # try:
        if tick in combined and tick in models:
            df = combined[tick]
            model = models[tick]
        else:
            logger.error(f'No data in COMBINED or no model in RUS_MODELS_30MINS for {tick} -> skipping')
            continue

        df.columns = ['close', 'volume']
        df.index = df.index.date

        days = (pd.to_datetime('today').date() - df.index[-1]).days
        period = str(days + 10) + ' D'
        try:
            idf = get_stock_data(ib, tick, period, barSize='30 mins')
        except:
            logger.error(f'No data in IB request for {tick} 30 mins -> skipping')
            continue

        idf = idf.iloc[:-1]
        idf.index = pd.to_datetime(idf.index)

        idf['day'] = pd.Series(idf.index, index=idf.index).apply(lambda x: x.date())

        idf['next_period'] = idf['day'].shift(-1)

        idf = idf[~(idf.next_period > idf.day)].set_index(keys='day', drop=True)[['close', 'volume']]

        # with pd.option_context('display.max_rows', None, 'display.max_columns',
        #                        None):  # more options can be specified also
        #     print(idf)
        close = idf.groupby(by=['day']).tail(1).close.copy()
        volume = idf.groupby(by=['day']).sum().volume.copy()
        idf = idf.groupby(by=['day']).tail(1)
        idf['close'] = close
        idf['volume'] = volume
        idf['volume'] = exp_function(model[1], model[0], idf.volume.to_numpy().reshape(-1, 1))
        # idf['volume'] = np.exp(model.predict(idf.volume.to_numpy().reshape(-1, 1))) * 1000
        # print(df)

        df = pd.concat([df, idf])
        df = df[~df.index.duplicated(keep='first')]

        na_idxs = np.where(df.close.isna())[0]
        if na_idxs.shape[0] > 0:
            first_fetch_date = df.index[na_idxs[0]]
            days = (pd.to_datetime('today').date() - first_fetch_date).days
            years = math.ceil(days / 365)
            period = str(years) + ' Y'
            try:
                idf = get_stock_data(ib, tick, period)
            except:
                logger.error(f'No data in IB request for {tick} 1 daily -> skipping')
                continue
            if idf.index[0] > first_fetch_date:
                logger.error(f'Not enough data for inference for {tick}: {idf.index[0]} but {first_fetch_date} needed')
                continue
            idf = idf[(idf.index >= df.index[0]) & (idf.index <= df.index[-1])]
            idf['volume'] = exp_function(model[1], model[0], idf.volume.to_numpy().reshape(-1, 1))
            df = df.merge(idf, left_index=True, right_index=True, how='left')
            df.apply(fill_df, axis=1)
            df = df[['close_x', 'volume_x']]
            df.columns = ['close', 'volume']

        df['volume'] = (df['volume'] - df['volume'].cummin()) / (df['volume'].cummax() - df['volume'].cummin())

        feats, feat_names = create_features(df, tick)
        feats = feats.iloc[[-1], :][feat_names]
        feats['ticker'] = tick
        feats.set_index('ticker', inplace=True)

        features_df = pd.concat([features_df, feats])

    logger.info('Preprocessing finished')
    return features_df


def inference(estimatorCopy, is_new_model, features_df, is_large_cap, send_to_logs):
    cols = ['vol2', 'vol5', 'vol10', 'vol22', 'vol44', 'vol51', 'vol66',
            'vol118', 'vol132', 'ret1', 'ret2', 'ret5', 'ret10', 'ret22', 'ret44',
            'ret51', 'ret66', 'ret118', 'ret132', 'ret246', 'ret261', 'ret375',
            'ret480', 'ret520']
    if is_new_model:
        cols.remove('ret1')
    print(features_df[cols])
    prediction = estimatorCopy.predict_proba(np.nan_to_num(features_df[cols]))

    logger.info('Starting model inference process for %s tickers', features_df.shape[0])
    ticker_proba = {'ticker': [], 'long': [], 'hold': [], 'short': []}

    for i, ticker in enumerate(features_df.index):
        ticker_proba['ticker'].append(ticker)
        if is_new_model:
            ticker_proba['hold'].append(0)
            ticker_proba['long'].append(prediction[i, 1])
        else:
            ticker_proba['hold'].append(prediction[i, 1])
            ticker_proba['long'].append(prediction[i, 2])
        ticker_proba['short'].append(prediction[i, 0])
    logger.info('Inference done')

    ticker_proba = pd.DataFrame(ticker_proba)
    ticker_proba = ticker_proba[['ticker', 'long', 'short']]
    ticker_proba.set_index('ticker', inplace=True)

    long_tickers = ticker_proba.sort_values(by=['long'], ascending=False).round(2)
    short_tickers = ticker_proba.sort_values(by=['short'], ascending=False).round(2)

    long_str = f"Tickers for LONG:\n" + f"{'Large cap' if is_large_cap else 'Small cap'}\n" + \
               tabulate(long_tickers.head(10),
                        [
                         'Ticker',
                         'Long',
                         # 'Hold',
                         'Short'
                        ],
                        tablefmt='simple')

    short_str = f"Tickers for SHORT:\n" + f"{'Large cap' if is_large_cap else 'Small cap'}\n" + \
                tabulate(short_tickers.head(10),
                         [
                          'Ticker',
                          'Long',
                          # 'Hold',
                          'Short'
                          ],
                         tablefmt='simple')

    if send_to_logs:
        chat = 'logs_group'
    else:
        chat = 'ml_group'
    tgm.send(long_str, chat)
    print(long_str)
    tgm.send(short_str, chat)
    logger.info('Sent message to telegram group')


# logger.info('Reserved PID: %s', os.getpid())
# dbm = DBManager(DB_USERNAME, DB_PASS, 'ODS')
# logger.info('Connected to DB successfully')
# tgm = TGManager(QC_TOKEN, TWS_USR_ID, ML_GROUP_ID, LOGS_GROUP_ID)
# logger.info('Connected to TG bot successfully')

def main(dbm: DBManager, tgm: TGManager):
    while True:
        try:
            for port in (IB_GW_PORT, IB_TWS_PORT):
                logger.info('Attempt to connect to TWS')
                try:
                    ib = IB()
                    ib.connect(SERVER_IP, port, clientId=3, readonly=True, timeout=90)
                    break
                except:
                    if port == IB_GW_PORT:
                        continue
                    raise Exception()
            logger.info('Connected to IB gateway successfully')
            tgm.send(f'Connected to IB\n{SCRIPT_NAME} in progress', 'logs_group')
            break
        except:
            logger.info(f'Could not launch {SCRIPT_NAME}\nWaiting for IB gateway re-login...')
            tgm.send(f'Could not launch {SCRIPT_NAME}\nWaiting for IB gateway re-login...', 'logs_group')
            tgm.tws_alert_block()

    with open(MODELS_PATH + '../Bloom_Files/rus_combined_140721_280721.dictionary', 'rb') as config_dictionary_file:
        combined = pickle.load(config_dictionary_file)
    logger.info('Loaded rus_combined_140721_280721.dictionary')
    first_date = datetime.now().date() - timedelta(1000)
    for i, ticker in enumerate(combined):
        df = combined[ticker]
        df = df[df.index > datetime(first_date.year, first_date.month, first_date.day)]
        combined[ticker] = df.drop(f'{ticker}_rep_date', axis=1)

    with open(MODELS_PATH + 'rus_models_30mins_160721.dictionary', 'rb') as config_dictionary_file:
        models = pickle.load(config_dictionary_file)
    logger.info('Loaded rus_models_30mins_160721.dictionary')
    # print(models)
    large_tickers = list(dbm.select_df('select * from reporting_large_tickers').iloc[:, 0].values)
    logger.info('Loaded reporting large tickers')

    large_prp_data = preprocess_tickers(large_tickers, combined, models, ib)
    old_model = pickle.load(open(MODELS_PATH + 'q2_inference_before_rus.sav', 'rb'))
    logger.info(f'Loaded inference model q2_inference_before_rus.sav')
    inference(old_model, False, large_prp_data, True, False)

    small_tickers = list(dbm.select_df('select * from reporting_small_tickers').iloc[:, 0].values)
    logger.info('Loaded reporting small tickers')
    small_prp_data = preprocess_tickers(small_tickers, combined, models, ib)
    inference(old_model, False, small_prp_data, False, False)

    # new_model = pickle.load(open(MODELS_PATH + 'binary_rus.sav', 'rb'))
    # logger.info(f'Loaded inference model test_inference_before_rus.sav')
    # large_prp_data = preprocess_tickers2(large_tickers, combined, models, ib)
    # inference(old_model, False, large_prp_data, True, True)
    #
    # small_prp_data = preprocess_tickers2(small_tickers, combined, models, ib)
    # inference(old_model, False, small_prp_data, False, True)
    ib.disconnect()
    sys.exit()


if __name__ == '__main__':

    logger.info('Reserved PID: %s', os.getpid())
    dbm = DBManager(DB_USERNAME, DB_PASS, 'DWH')
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
