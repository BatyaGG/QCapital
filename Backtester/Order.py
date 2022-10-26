import os
# cwd = os.getcwd()
# print(cwd)


class Order:
    def __init__(self, ticker: str, kind: str, dir: str, amount: float, what: str, info=None):
        assert kind in ('market', 'limit', 'stop', 'market_TS')
        assert dir in ('buy', 'sell', 'short', 'cover')
        assert type(amount) in (int, float)
        # assert amount > 0
        assert what in ('shares', '$', '%')

        self.ticker = ticker
        self.kind = kind
        self.dir = dir
        self._amount = amount
        self.what = what
        self.info = info
        # self.ts = ts  # trailing stop

    @property
    def amount(self):
        # assert self._amount > 0
        return self._amount

    @amount.setter
    def amount(self, val):
        # assert self._amount > 0
        # assert val > 0
        self._amount = val

    def __str__(self):
        # self.ticker = ticker
        # self.kind = kind
        # self.dir = dir
        # self._amount = amount
        # self.what = what
        return f"""
            Ticker: {self.ticker}
            Kind: {self.kind}
            Dir: {self.dir}
            Amount: {self._amount}
            """
