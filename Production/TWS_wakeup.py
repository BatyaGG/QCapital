from datetime import datetime

from ib_insync import IB


ib = IB()
ib.connect('127.0.0.1', 4001, clientId=0, readonly=True, timeout=90)

while True:
    print(datetime.now(), ib.managedAccounts())
    ib.sleep(600)
