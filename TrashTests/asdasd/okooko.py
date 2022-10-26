import os
import pandas as pd

from ib_insync import Stock, IB, util, Client
from ib_insync.order import LimitOrder
# print(os.listdir("asdasd"))

print(os.getcwd())

spx = pd.read_csv('../../DataParsers/Bloomberg/Data/SPXT_290721.csv')
# spx = pd.read_csv('spx index.csv')
spx.columns = ['date', 'price']
print(spx.dtypes)
# spx['date'] = spx['date'].astype(str)
spx['date'] = pd.to_datetime(spx['date'], format='%d-%b-%y')
print(spx)
