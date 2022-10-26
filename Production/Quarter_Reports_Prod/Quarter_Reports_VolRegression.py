import os
import sys
import logging
import pickle as pk

from scipy import stats
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from ib_insync import Stock, IB, util
from matplotlib import pyplot as plt

sys.path.append('../..')
import config
from QuantumCapital import constants
from QuantumCapital.DBManager import DBManager
from QuantumCapital.TGManager import TGManager


# CONSTANTS
SCRIPT_NAME = 'Quarter Reports Volume Regression'
LOG_FILE_PATH = 'Logs/Quarter_Reports_VolRegression.log'

MODELS_PATH = '../../Models_&_Files/Quarter_Reports_Models/'

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
    maxBytes=5*1024*1024,
    backupCount=2,
    encoding=None,
    delay=0
)

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


def main():
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

    tickers = dbm.select_df(f'select * from {constants.DAY_BAR_RUSSEL_PRESENCE_TABLE} '
                            f'order by yr desc limit 1').iloc[0].present
    with open(MODELS_PATH + '../Bloom_Files/rus_combined_140721.dictionary', 'rb') as config_dictionary_file:
        combined = pk.load(config_dictionary_file)

    models = {}
    for i, ticker in enumerate(tickers):
        if ticker in combined:
            df_bloom = combined[ticker]
        else:
            continue
        df_bloom = df_bloom[['volume']].iloc[-150:]
        df_bloom = df_bloom[~df_bloom.volume.isna()]

        days = (pd.to_datetime('today') - df_bloom.index[0]).days
        try:
            df_ib = get_stock_data(ib, ticker, f'{days} D', barSize='30 mins')
        except:
            continue
        print(i, ticker)

        df_ib.index = pd.to_datetime(df_ib.index)
        df_ib['day'] = pd.Series(df_ib.index, index=df_ib.index).apply(lambda x: x.date())
        df_ib['day'] = pd.to_datetime(df_ib['day'])
        df_ib = df_ib[df_ib.day >= df_bloom.index[0]]
        df_ib['next_period'] = df_ib['day'].shift(-1)
        df_ib['last'] = np.where(df_ib.next_period > df_ib.day, 1, 0)
        df_ib = df_ib[df_ib['last'] == 0].iloc[:-1]
        # with pd.option_context('display.max_rows', None, 'display.max_columns',
        #                        None):  # more options can be specified also
        df_fit = df_bloom.merge(df_ib[['day', 'volume']],
                                how='left',
                                left_index=True,
                                right_on='day',
                                suffixes=('_y', '_x'))[['volume_x', 'volume_y']]
        df_fit.dropna(inplace=True)
        # print(df_bloom)
        # print(df_ib)
        # print(df_fit)
        # print('-----------------------------------')
        # plt.plot(df_fit.volume_x, df_fit.volume_y, '.')
        # plt.show()

        # if i == 80:
        #     with pd.option_context('display.max_rows', None, 'display.max_columns',
        #                            None):  # more options can be specified also
        #         print(df_fit)

        x = df_fit.volume_x.to_numpy()
        y = np.log(df_fit.volume_y.to_numpy())
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

        # model = LinearRegression().fit(x.reshape(-1, 1), y)
        models[ticker] = (slope, intercept)
    with open(MODELS_PATH + f"rus_models_30mins_{pd.to_datetime('today').date().strftime('%d%m%y')}.dictionary", 'wb') as file:
        pk.dump(models, file, protocol=pk.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    logger.info('Reserved PID: %s', os.getpid())
    dbm = DBManager(DB_USERNAME, DB_PASS, 'BARS_DAY')
    logger.info('Connected to DB successfully')
    tgm = TGManager(QC_TOKEN, TWS_USR_ID, ML_GROUP_ID, LOGS_GROUP_ID)
    logger.info('Connected to TG bot successfully')

    try:
        main()
    except Exception as e:
        dbm.rollback()
        tgm.send(SCRIPT_NAME + ': ' + repr(e), 'logs_group')
        logger.error('Error at %s', 'division', exc_info=e)
    logger.info('end of script___________________________________________________')
