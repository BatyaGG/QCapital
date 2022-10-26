import pandas as pd
import sys
sys.path.append('../..')
import config
# from QuantumCapital import constants
# from QuantumCapital.DBManager import DBManager
# from QuantumCapital.TGManager import TGManager
# from QuantumCapital.PlotManager import *
#
#
# DB_USERNAME = config.DB_USERNAME
# DB_PASS = config.DB_PASS
#
#
# tickers = pd.read_csv('reporting_small_tickers (50).csv')
#
# print(tickers)
#
# dbm = DBManager(DB_USERNAME, DB_PASS, 'ODS')
# dbm.truncate_table('reporting_small_tickers')
# dbm.insert_df_simple(tickers, 'reporting_small_tickers')
# dbm.commit()

# import requests
#
# # replace the "demo" apikey below with your own key from https://www.alphavantage.co/support/#api-key
# url = 'https://www.alphavantage.co/query?function=EARNINGS&symbol=IBM&apikey=HIPEKPCQ0Z5WWNXF'
# r = requests.get(url)
# data = r.json()
#
# print(data)

# import csv
# import requests
# import pandas as pd
#
# CSV_URL = 'https://www.alphavantage.co/query?function=EARNINGS_CALENDAR&horizon=3month&apikey=HIPEKPCQ0Z5WWNXF'
#
# with requests.Session() as s:
#     download = s.get(CSV_URL)
#     decoded_content = download.content.decode('utf-8')
#     cr = csv.reader(decoded_content.splitlines(), delimiter=',')
#     df = pd.DataFrame(cr)
#     df.columns = df.iloc[0]
#     df = df.iloc[1:]
#
# print(df)

import pickle

MODELS_PATH = '../Models_&_Files/Quarter_Reports_Models/'


with open(MODELS_PATH + 'rus_combined.dictionary', 'rb') as config_dictionary_file:
    combined = pickle.load(config_dictionary_file)

print(combined['tsla'])


