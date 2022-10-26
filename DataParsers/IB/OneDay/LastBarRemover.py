import sys

sys.path[0] = '../..'
import config
from QuantumCapital import constants
from QuantumCapital.DBManager import DBManager


class LastBarRemover:
    def __init__(self):
        self.dbm = DBManager(config.DB_USERNAME, config.DB_PASS, 'BARS_DAY')
        self.dict_tbl = constants.DAY_BAR_DICTIONARY_TABLE
        self.presence_tbl = constants.DAY_BAR_RUSSEL_PRESENCE_TABLE

    def remove_last_bars(self, year):
        tickers = self.get_year_tickers(year)
        for ticker in tickers:
            ticker_info = self.dbm.select_df(f'select table_name, last_date, bars_count \
                                              from {self.dict_tbl} where ticker=\'{ticker}\'')
            table_name = ticker_info.iloc[0].table_name
            last_date = ticker_info.iloc[0].last_date
            bars_count = ticker_info.iloc[0].bars_count

            self.dbm.delete_table_row(table_name, {'dt': "\'" + str(last_date) + "\'"})

            last_date_new = self.dbm.select_df(f'select dt from {table_name} order by dt desc')
            last_date_new = last_date_new.iloc[0]['dt']
            print(last_date, last_date_new)

            # (constants.DAY_BAR_DICTIONARY_TABLE,
            #  {'last_date': f"\'{last_date}\'",
            #   'bars_count': f'bars_count + {new_bars_count}',
            #   'years': "\'{" + ', '.join([str(y) for y in years]) + "}\'",
            #   'memory': size},
            #  {'ticker': f"\'{ticker.upper()}\'"})
            self.dbm.update_table_row(self.dict_tbl, {'last_date': f"\'{last_date_new}\'", 'bars_count': f'bars_count - 1'},
                                      {'ticker': f"\'{ticker.upper()}\'"})

            print('----------------------')
            self.dbm.commit()

    def get_year_tickers(self, year):
        df = self.dbm.select_df(f'select * from {self.presence_tbl} where yr={year}')
        assert df.shape[0] == 1
        return df.iloc[0].present


if __name__ == '__main__':
    remover = LastBarRemover()
    remover.remove_last_bars(2020)
