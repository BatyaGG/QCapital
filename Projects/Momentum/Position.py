from datetime import date


class Position:
    def __init__(self, entry_dt: date, entry_price: float, ticker: str, amount: float):
        self.entry_dt = entry_dt
        self.exit_dt = None
        self.entry_price = entry_price
        self.exit_price = None
        self.ticker = ticker
        self._amount = amount
        self.pnl = None

    # @property
    # def entry_dt(self):
    #     assert self._entry_dt is not None
    #     return self._entry_dt
    #
    # @entry_dt.setter
    # def entry_dt(self, val):
    #     assert type(val) == date
    #     self._entry_dt = val

    @property
    def amount(self):
        assert self._amount > 0
        return self._amount

    @amount.setter
    def amount(self, val):
        assert self._amount > 0
        assert val > 0
        self._amount = val

    def __str__(self):
        return f"""
            Entry dt: {self.entry_dt}
            Exit dt: {self.exit_dt}
            Entry price: {self.entry_price}
            Exit price: {self.exit_price}
            Ticker: {self.ticker}
            Amount: {self.amount}
            PnL: {self.pnl}
            """
