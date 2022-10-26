import sys, pytz

import pandas as pd
from ib_insync import Stock, IB, util

from datetime import datetime

ib = IB()
ib.connect('192.168.31.12', 4001, clientId=10, readonly=True, timeout=90)
contract = Stock('UKYE', 'SMART/ISLAND', 'USD')

# a = ib.reqFundamentalData(contract, 'CalendarReport')
# print(a)
end = datetime(2012, 6, 30)
# end = datetime.now(pytz.timezone('America/New_York'))
#
bars = ib.reqHistoricalData(
    contract,
    endDateTime=end,
    durationStr='365 D',
    barSizeSetting='1 day',
    whatToShow='TRADES',
    useRTH=True,
    formatDate=2,
    timeout=30)
idf = util.df(bars)
print(end)
print(idf)
