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

for i in range(2002, 2021):
    old_tickers = dbm.select_df(f'select * from {constants.RUSSEL_MEMBERSHIP_HISTORICAL} where yr = {i}').tickers.iloc[0]
    new_tickers = set()
    for old_ticker in old_tickers:
        new_tickers.add(old_ticker.split()[0])
    dbm.update_table_row(constants.RUSSEL_MEMBERSHIP_HISTORICAL,
                         {'tickers': "%s"}, {'yr': f'={i}'},
                         add_params=(list(new_tickers),))
    dbm.commit()
