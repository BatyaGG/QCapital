import pandas as pd

raw_dt_parq = 'rus_rawdates_140721.parquet'
raw_dt = pd.read_parquet(raw_dt_parq)
raw_dt_drop = raw_dt.dropna(axis=1, how='all')
print((raw_dt.shape[1] - raw_dt_drop.shape[1]) / 2)
