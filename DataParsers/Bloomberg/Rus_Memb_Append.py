import sys

import pandas as pd

sys.path.append('../../..')

import config
from QuantumCapital import constants
from QuantumCapital.DBManager import DBManager
from QuantumCapital.PlotManager import *


DB_USERNAME = config.DB_USERNAME
DB_PASS = config.DB_PASS

dbm = DBManager(DB_USERNAME, DB_PASS, 'BARS_DAY')

df = pd.read_csv('Data/RAY as of Jul 07 20211.csv')
yr = 2021
tickers = []
for i in range(df.shape[0]):
    ticker = df.iloc[i].Ticker.split()[0]
    tickers.append(ticker)

row = pd.DataFrame({'yr': [yr], 'tickers': [tickers]})
dbm.insert_df_simple(row, constants.RUSSEL_MEMBERSHIP_HISTORICAL)
dbm.commit()
