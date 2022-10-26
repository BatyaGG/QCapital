from datetime import date


class Position:
    def __init__(self,
                 entry_dt: date,
                 entry_price: float,
                 ticker: str,
                 amount: float,
                 stop: float = None,
                 take: float = None,
                 info=None):
        self._entry_dt = entry_dt
        self._exit_dt = None
        self._entry_price = entry_price if type(entry_price) is float else float(entry_price.cpu().numpy())
        self._exit_price = None
        self.ticker = ticker
        self._amount = amount if type(amount) is int else int(amount.cpu().numpy())
        self._pnl = None
        self._stop = stop if type(amount) is float else None if stop is None else float(stop.cpu().numpy())
        self.stop_hist = [[self.entry_dt, self._stop]] if stop is not None else []
        self.stop_triggered = False

        self._take = take if type(amount) is float else None if take is None else float(take.cpu().numpy())
        self.take_hist = [[self.entry_dt, self._take]] if take is not None else []
        self.take_triggered = False

        self.precise_take_stop = False
        self.info = info

    @property
    def entry_dt(self):
        return self._entry_dt

    @entry_dt.setter
    def entry_dt(self, val):
        assert type(val) == date
        self._entry_dt = val

    @property
    def exit_dt(self):
        return self._exit_dt

    @exit_dt.setter
    def exit_dt(self, val):
        assert type(val) == date
        self._exit_dt = val

    @property
    def entry_price(self):
        assert self._entry_price > 0
        return self._entry_price

    @entry_price.setter
    def entry_price(self, val):
        assert val > 0
        self._entry_price = val if type(val) is float else float(val.cpu().numpy())

    @property
    def exit_price(self):
        assert self._exit_price > 0
        return self._exit_price

    @exit_price.setter
    def exit_price(self, val):
        self._exit_price = val if type(val) is float else float(val.cpu().numpy())
        self._pnl = self._amount * (self._exit_price - self._entry_price)

    @property
    def amount(self):
        # assert self._amount > 0
        return self._amount

    @amount.setter
    def amount(self, val):
        # assert val > 0
        self._amount = val if type(val) is int else int(val.cpu().numpy())

    @property
    def pnl(self):
        return self._pnl

    @property
    def stop(self):
        # assert self._ts > 0
        return self._stop

    @stop.setter
    def stop(self, val):
        # assert val > 0
        assert self.amount != 0
        if self.amount > 0:
            self._stop = max(self._stop if self._stop else 0, val[1]) if type(val[1]) is float \
                else max(self._stop if self._stop else 0, float(val[1].cpu().numpy()))
            self.stop_hist.append([val[0], self._stop])
        else:
            self._stop = min(self._stop if self._stop else float('inf'), val[1]) if type(val[1]) is float \
                else min(self._stop if self._stop else float('inf'), float(val[1].cpu().numpy()))
            self.stop_hist.append([val[0], self._stop])

    @property
    def take(self):
        # assert self._ts > 0
        return self._take

    @take.setter
    def take(self, val):
        # assert val > 0
        assert self.amount != 0
        self._take = max(0, val[1]) if type(val[1]) is float else max(0, float(val[1].cpu().numpy()))
        self.take_hist.append([val[0], self._take])

    def __str__(self):
        return f"""
            entry_dt: {self._entry_dt}
            exit_dt: {self._exit_dt}
            entry_price: {self._entry_price}
            exit_price: {self._exit_price}
            ticker: {self.ticker}
            amount: {self._amount}
            pnl: {self._pnl}
            ts: {self._stop}
            ts_hist: {self.stop_hist}
            info: \n{self.info}
            """
