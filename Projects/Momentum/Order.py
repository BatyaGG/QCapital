

class Order:
    def __init__(self, ticker: str, kind: str, dir: str, amount: float, what: str):
        assert kind in ('market', 'limit', 'stop')
        assert dir in ('buy', 'sell')
        assert type(amount) in (int, float)
        assert amount > 0
        assert what in ('shares', '$', '%')

        self.ticker = ticker
        self.kind = kind
        self.dir = dir
        self._amount = amount
        self.what = what

    @property
    def amount(self):
        assert self._amount > 0
        return self._amount

    @amount.setter
    def amount(self, val):
        assert self._amount > 0
        assert val > 0
        self._amount = val

    # def __str__(self):
    #     return f"""
    #         Ticker: {self.ticker}
    #         Amount: {self._amount}
    #         PnL: {self.pnl}
    #         """