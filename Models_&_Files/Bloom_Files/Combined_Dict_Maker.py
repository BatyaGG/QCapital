import pickle as pk
from datetime import date, datetime

import pandas as pd
import numpy as np

import config
from QuantumCapital import constants
from QuantumCapital import HelperFunctions as hf
from QuantumCapital.DBManager import DBManager
# pd.set_option("display.max_rows", None)
# pd.set_option("display.max_columns", None)

DB_USERNAME = config.DB_USERNAME
DB_PASS = config.DB_PASS

price_parq = 'rus_price_201120.parquet'
volume_parq = 'rus_volume_201120.parquet'
raw_dt_parq = 'rus_rawdates_201120.parquet'
parq_updated_date = price_parq.split('_')[2].split('.')[0]

combined_name = f"rus_combined_{parq_updated_date}_{date.today().strftime('%d%m%y')}.dictionary"
aligned_df_name = f"rus_aligned_df_{parq_updated_date}_{date.today().strftime('%d%m%y')}.parquet"

price = pd.read_parquet(price_parq)
volume = pd.read_parquet(volume_parq)
raw_dt = pd.read_parquet(raw_dt_parq)

dates = set()

for i in range(0, price.shape[1], 2):
    price_df = price.iloc[:, i:i + 2]
    ticker_p = price.columns[i + 1]
    dts = price_df.iloc[:, 0]
    try:
        dates.update(dts[~np.isnan(dts)])
    except:
        print(price_df)
dates = pd.DataFrame({'dt': sorted(list(dates))[:-1]})

combined = {}
combined_df = dates.copy().set_index('dt', inplace=False, drop=True)

dbm = DBManager(DB_USERNAME, DB_PASS, 'BARS_DAY')
rus_index = dbm.select_df(f'select * from {constants.RUSSEL_MEMBERSHIP_HISTORICAL}')
rus_index_yr_min = rus_index.yr.min()
rus_index_yr_max = rus_index.yr.max()

combined_dfs = []


def get_dt_range_rmv(year: int):
    return datetime(year, 7, 1, 0, 0, 0), datetime(year, 12, 31, 23, 59, 59)


def get_dt_range_fill(year: int):
    return datetime(year, 1, 1, 0, 0, 0), datetime(year + 1, 6, 30, 23, 59, 59)

# with open(combined_name, 'rb') as file:
#     combined = pk.load(file)


for i in range(0, price.shape[1], 2):
    ticker_p = price.columns[i + 1]
    ticker_v = volume.columns[i + 1]
    ticker_d = price.columns[i + 1]

    assert ticker_p == ticker_v == ticker_d

    absent_years = [rus_index.iloc[j, :].yr for j in range(rus_index.shape[0])
                    if ticker_p.upper() not in rus_index.iloc[j, :].tickers]
    present_years = sorted(list(set(np.arange(rus_index_yr_min, rus_index_yr_max + 1)) - set(absent_years)))

    absent_years = [get_dt_range_rmv(yr) for yr in sorted(absent_years)]
    present_years = [get_dt_range_fill(yr) for yr in sorted(present_years)]

    price_df = price.iloc[:, i:i + 2]
    volume_df = volume.iloc[:, i:i + 2]
    dates_df = raw_dt.iloc[:, i:i + 2]

    try:
        if i == 0:
            price_df = dates.merge(price_df, how='left', left_on='dt', right_on='date')
            volume_df = dates.merge(volume_df, how='left', left_on='dt', right_on='date')
        else:
            price_df = dates.merge(price_df, how='left', left_on='dt', right_on=f'date.{i//2}')
            volume_df = dates.merge(volume_df, how='left', left_on='dt', right_on=f'date.{i//2}')
    except Exception as e:
        continue

    try:
        if i == 0:
            dates_df = dates.merge(dates_df, how='left', left_on='dt', right_on='date')
        else:
            dates_df = dates.merge(dates_df, how='left', left_on='dt', right_on=f'date.{i // 2}')
    except Exception as e:
        dates_df = dates.copy()
        dates_df['date'] = pd.NaT
        dates_df['rep_date'] = pd.NA

    ticker = ticker_p.upper().replace('/', '_')
    df = pd.concat([dates, price_df.iloc[:, 2], dates_df.iloc[:, 2], volume_df.iloc[:, 2]], axis=1)
    df.columns = ['dt', ticker + '_price', ticker + '_rep_date', ticker + '_volume']
    df.set_index('dt', inplace=True, drop=True)
    df[ticker + '_rep_date'] = df[ticker + '_rep_date'].notnull()
    df = df.iloc[:-1]

    # print(df)
    # exit()
    # tk = ['PG']
    tk = []

    # if ticker in tk:
    # with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    #     print([y[0].year for y in present_years])
    #     print([y[0].year for y in absent_years])
    #     print(df)
    # with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    #     print(df)

    for j, prs_rng in enumerate(present_years):
        if prs_rng == get_dt_range_fill(rus_index_yr_min):
            if ticker in tk:
                print('do:', prs_rng)
                print(df[df.index.isin(pd.date_range(date(1972, 1, 1), prs_rng[1]))][ticker + '_price'])
            df.loc[df.index.isin(pd.date_range(date(1972, 1, 1), prs_rng[1])), ticker + '_volume'] = \
                df.loc[df.index.isin(pd.date_range(date(1972, 1, 1), prs_rng[1])), ticker + '_volume'].fillna(0)
            df.loc[df.index.isin(pd.date_range(date(1972, 1, 1), prs_rng[1])), ticker + '_price'] = \
                df.loc[df.index.isin(pd.date_range(date(1972, 1, 1), prs_rng[1])), ticker + '_price'].fillna(method='ffill')
            if ticker in tk:
                print(df[df.index.isin(pd.date_range(date(1972, 1, 1), prs_rng[1]))][ticker + '_price'])
                print('-------------------------------------------------------------------')
        elif prs_rng == get_dt_range_fill(rus_index_yr_max):
            if ticker in tk:
                print('do:', prs_rng)
                print(df[df.index.isin(pd.date_range(prs_rng[0], date.today()))][ticker + '_price'])
            df.loc[df.index.isin(pd.date_range(prs_rng[0], date.today())), ticker + '_volume'] = \
                df.loc[df.index.isin(pd.date_range(prs_rng[0], date.today())), ticker + '_volume'].fillna(0)
            if j > 0 and present_years[j - 1] == get_dt_range_fill(rus_index_yr_max - 1):
                prs_rng = datetime(rus_index_yr_max - 1, 12, 1), prs_rng[1]
            df.loc[df.index.isin(pd.date_range(prs_rng[0], date.today())), ticker + '_price'] = \
                df.loc[df.index.isin(pd.date_range(prs_rng[0], date.today())), ticker + '_price'].fillna(method='ffill')
            if ticker in tk:
                print(df[df.index.isin(pd.date_range(prs_rng[0], date.today()))][ticker + '_price'])
                print('-------------------------------------------------------------------')
        else:
            if ticker in tk:
                print('do:', prs_rng)
                print(df[df.index.isin(pd.date_range(prs_rng[0], prs_rng[1]))][ticker + '_price'])
            df.loc[df.index.isin(pd.date_range(prs_rng[0], prs_rng[1])), ticker + '_volume'] = \
                df.loc[df.index.isin(pd.date_range(prs_rng[0], prs_rng[1])), ticker + '_volume'].fillna(0)
            if j > 0 and present_years[j - 1] == get_dt_range_fill(prs_rng[0].year - 1):
                prs_rng = datetime(prs_rng[0].year - 1, 12, 1), prs_rng[1]
            df.loc[df.index.isin(pd.date_range(prs_rng[0], prs_rng[1])), ticker + '_price'] = \
                df.loc[df.index.isin(pd.date_range(prs_rng[0], prs_rng[1])), ticker + '_price'].fillna(method='ffill')
            if ticker in tk:
                print(df[df.index.isin(pd.date_range(prs_rng[0], prs_rng[1]))][ticker + '_price'])
                print('-------------------------------------------------------------------')

    for j, abs_rng in enumerate(absent_years):
        if abs_rng == get_dt_range_rmv(rus_index_yr_min):
            if j < len(absent_years) - 1 and absent_years[j + 1] == get_dt_range_rmv(rus_index_yr_min + 1):
                abs_rng = hf.get_russel_dt_range(date(rus_index_yr_min, 12, 1))
            df.loc[df.index.isin(pd.date_range(date(1972, 1, 1), abs_rng[1])), :] = [np.nan, pd.NA, np.nan]
        elif abs_rng == get_dt_range_rmv(rus_index_yr_max):
            df.loc[df.index.isin(pd.date_range(abs_rng[0], date.today())), :] = [np.nan, pd.NA, np.nan]
        else:
            if j < len(absent_years) - 1 and absent_years[j + 1] == get_dt_range_rmv(abs_rng[0].year + 1):
                abs_rng = hf.get_russel_dt_range(date(abs_rng[0].year, 12, 1))
            df.loc[df.index.isin(pd.date_range(abs_rng[0], abs_rng[1])), :] = [np.nan, pd.NA, np.nan]

    # with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    #     price_df[price_df.iloc[:, 0].isin(pd.date_range(abs_rng[0], abs_rng[1]))] = [None, None]
    #     volume_df[volume_df.iloc[:, 0].isin(pd.date_range(abs_rng[0], abs_rng[1]))] = [None, None]
    #     dates_df[dates_df.iloc[:, 0].isin(pd.date_range(abs_rng[0], abs_rng[1]))] = [None, None]

    # if ticker in tk:
    # with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    #     print([y[0].year for y in present_years])
    #     print([y[0].year for y in absent_years])
    #     print(df)
    #     print('----------------------------------------------------------------\n\n')
        # exit()
    combined[ticker] = df
    combined_dfs.append(df)
    # combined_df = pd.concat([combined_df, df], axis=1)
    print()
    print(f'{round(100 * (2 * i) / int(price.shape[1]), 1)} % -------------------------')

combined_df = pd.concat(combined_dfs, axis=1)
print(combined_df)

with open(combined_name, 'wb') as file:
    pk.dump(combined, file, protocol=pk.HIGHEST_PROTOCOL)

combined_df.to_parquet(aligned_df_name)
