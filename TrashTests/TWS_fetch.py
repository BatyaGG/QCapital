import pickle as pk

import sys
import pytz

import pandas as pd
from ib_insync import Stock, IB, util

import pickle
from datetime import datetime

sys.path[0] = '../..'
import config
from QuantumCapital import constants
from QuantumCapital.DBManager import DBManager

# with open('russel_presence_2009.pickle', 'rb') as file:
#     pr = pk.load(file)
#
# a = pr.present.str.len()
# print(a)
#
# print(pr)

ib = IB()
ib.connect('192.168.31.7', 4001, clientId=2, readonly=True, timeout=90)

contract = Stock('AAPL', 'SMART/ISLAND', 'USD')
bars = ib.reqHistoricalData(
    contract,
    endDateTime='',
    durationStr='3 Y',
    barSizeSetting='1 day',
    whatToShow='TRADES',
    useRTH=True,
    formatDate=2,
    timeout=60)
idf = util.df(bars)
print(idf)
# dates = list(idf.date)
# dates = sorted(list(set([dt.date() for dt in dates])))
# print(dates)
