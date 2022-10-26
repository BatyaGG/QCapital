import sys, os
sys.path.append('../..')
sys.path.append('../../Backtester')

import pandas as pd

from StrategyClass import Strategy

pd.set_option('display.max_columns', None)
# path = os.path.dirname(__file__)
# # path = os.path.join(os.path.dirname(__file__), os.pardir)
# print(path)
# sys.path.append(path)
import config
from QuantumCapital import constants
from QuantumCapital.DBManager import DBManager
from QuantumCapital.TGManager import TGManager
from QuantumCapital.PlotManager import *

# pd.read_parquet('../../Models_&_Files/Bloom_Files/rus_aligned_df_140721_280721.parquet')


# LOG_FILE_PATH = 'Logs/Momentum_Inference.log'
DB_USERNAME = config.DB_USERNAME
DB_PASS = config.DB_PASS

dbm = DBManager(DB_USERNAME, DB_PASS, 'DWH')

# param1 = ('trailing_stop_window', (170, 200 + 1, 10))
# param2 = ('trailing_stop_coeff', (1, 20 + 1, 1))

param1 = ('dots_n', (10, 200 + 1, 20))
param2 = ('vola_window', (10, 200 + 1, 20))

records_table = f'momentum_tuning_new_dots_vola_seventy'

# data = pd.read_parquet('../../Models_&_Files/Bloom_Files/rus_aligned_df_140721_280721.parquet')

cnt = 0
for i in range(param1[1][0], param1[1][1], param1[1][2]):
    for j in range(param2[1][0], param2[1][1], param2[1][2]):
        print(f'#{cnt}: {i} | {j}')
        strategy = Strategy(use_trailing_stop=False, precise_stops=False)
        strategy.params[param1[0]] = i
        strategy.params[param2[0]] = j
        strategy.init_broker()
        res = strategy.run(plot_mv=False)
        dbm.insert_df_simple(res, records_table)
        dbm.commit()
        cnt += 1
