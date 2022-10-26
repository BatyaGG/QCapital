import sys

sys.path[0] = '../..'
import config
from QuantumCapital import constants
from QuantumCapital.DBManager import DBManager


class YearRemover:
    def __init__(self):
        self.dbm = DBManager(config.DB_USERNAME, config.DB_PASS, 'BARS_DAY')
        self.dict_tbl = constants.DAY_BAR_DICTIONARY_TABLE
        self.presence_tbl = constants.DAY_BAR_RUSSEL_PRESENCE_TABLE

    def remove_year(self, year):
        tickers = self.get_year_tickers(year)
        for ticker in tickers:
            ticker_info = self.dbm.select_df(f'select table_name, years, bars_count from {self.dict_tbl} where ticker={ticker}')
            print(ticker_info)

    def get_year_tickers(self, year):
        df = self.dbm.select_df(f'select * from {self.presence_tbl} where yr={year}')
        assert df.shape[0] == 1
        return df.iloc[0].present


if __name__ == '__main__':
    yr = YearRemover()
    yr.remove_year(2020)
