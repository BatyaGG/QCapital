import sys

import pandas as pd

sys.path.append('../..')

import config
from QuantumCapital import constants
from QuantumCapital.DBManager import DBManager
from QuantumCapital.PlotManager import *


DB_USERNAME = config.DB_USERNAME
DB_PASS = config.DB_PASS


data = pd.read_csv('Data/ray280621.csv')
new_tickers = data.Ticker
new_tickers = new_tickers.apply(lambda x: x.split()[0])
new_tickers = new_tickers.to_list()

dbm = DBManager(DB_USERNAME, DB_PASS, 'BARS_DAY')

old_tickers = dbm.select_df('select * from rus_tickers_historical order by yr desc limit 1').iloc[0].tickers
old_tickers = [ticker.split()[0] for ticker in old_tickers]

print(old_tickers)
print(new_tickers)


print(f'New tickers came: {len(set(new_tickers) - set(old_tickers))}')
print(f'Old tickers lost: {len(set(old_tickers) - set(new_tickers))}')