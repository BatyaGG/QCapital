#!/usr/bin/env python3

import pytz
import requests
import time
import sys, os
import logging.config

import enchant
import telegram
import pandas as pd
pd.set_option('display.max_columns', None)

from datetime import datetime
from string import punctuation

sys.path[0] = '../..'
import config
from QuantumCapital import constants
from QuantumCapital.DBManager import DBManager
from QuantumCapital.TGManager import TGManager
from QuantumCapital.HelperFunctions import get_year_rus_tickers

import demoji
demoji.download_codes()

# CONSTANTS
SCRIPT_NAME = 'Discussion Thread Parser'

QC_TOKEN = constants.QC_TOKEN
TWS_USR_ID = constants.TWS_USER_ID
ML_GROUP_ID = constants.QC_ML_GROUP_ID
LOGS_GROUP_ID = constants.QC_BOT_LOGS_GROUP_ID

DB_USERNAME = config.DB_USERNAME
DB_PASS = config.DB_PASS

LOG_FILE_PATH = 'Logs/Discussion_Thread_Comments.log'


# Configuring logging rules -------------------------------------------------------------- /
a = logging.Logger.manager.loggerDict  # Disabling other loggers
for k in a:
    a[k].disabled = True

for handler in logging.root.handlers[:]:  # Disabling root logger handlers
    logging.root.removeHandler(handler)

rfh = logging.handlers.RotatingFileHandler(
    filename=LOG_FILE_PATH,
    mode='a',
    maxBytes=5*1024*1024,
    backupCount=2,
    encoding=None,
    delay=0
)

logging.basicConfig(handlers=[rfh],
    # filename=LOG_FILE_PATH,
                    format='%(asctime)s %(levelname)s [%(funcName)s]: %(message)s',
                    datefmt='%Y-%m-%d | %H:%M:%S',
                    level=logging.ERROR)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

requests_logger = logging.getLogger('requests')
requests_logger.setLevel(logging.ERROR)

handler = logging.StreamHandler()
handler.setLevel(logging.ERROR)
logger.addHandler(handler)
requests_logger.addHandler(handler)

# logging.getLogger().setLevel(logging.DEBUG)  # If you want logger write in console
logger.info(
    '\n\n_______________________________________________________________________________________________________')
# --------------------------------------------------------------------------------------- /


def clean_ticker(tick):
    r = ''.join([t for t in list(tick) if t not in punctuation])
    return demoji.replace(r, '')


def is_ticker(word: str):
    clear_word = []
    for letter in word:
        if letter.isalpha():
            clear_word.append(letter)
    clear_word = ''.join(clear_word).upper()
    if clear_word in all_tickers and not dictionary.check(clear_word):
        return True, clear_word
    return False, None


def find_tickers(text):
    tickers = set()
    for word in text.split(' '):
        ticker_found, ticker_like = is_ticker(word)
        if ticker_found:
            tickers.add(ticker_like)
    tickers = ','.join(tickers)
    return tickers


def get_posts():
    fresh = ''
    while fresh == '':
        r = requests.get(
            'https://www.reddit.com/r/wallstreetbets/.json',
            headers={'user-agent': 'Mozilla/5.0'})
        fresh = 'https://www.reddit.com' + r.json()['data']['children'][0]['data']['permalink'] + '.json'
    r = requests.get(fresh, headers={'user-agent': 'Mozilla/5.0'})
    posts = r.json()[1]['data']['children']
    posts_new = []
    for post in posts:
        if 'body' not in post['data']:
            continue
        dt = datetime.fromtimestamp(post['data']['created_utc'], pytz.UTC)
        posts_new.append({'dt': dt, 'id': post['data']['id'], 'ticker_like': find_tickers(post['data']['body']), 'text': post['data']['body']})
    return posts_new


def get_last_record_id(dbm: DBManager):
    cursor = dbm.select_cursor(f'select id from {constants.WSB_DISCUSSION_THREAD_TABLE} order by dt desc limit 1')
    try:
        id_num = cursor.fetchone()[0]
    except TypeError as error:
        id_num = None
    finally:
        cursor.close()
    return id_num


def cut_posts(last_id, posts):
    i = 0
    post = posts[i]
    while i < len(posts) and post['id'] != last_id:
        i += 1
        if i < len(posts):
            post = posts[i]
    return posts[:i]


def send(msg, chat_id, token):
    msg = '```python\n' + msg + '```'
    bot = telegram.Bot(token=token)
    bot.sendMessage(chat_id=chat_id, text=msg, parse_mode='MarkdownV2')


def main(dbm):
    last_id = get_last_record_id(dbm)
    while True:
        posts = get_posts()
        if posts:
            posts = cut_posts(last_id, posts)
        if posts:
            dbm.insert_df_simple(pd.DataFrame(posts[::-1]), constants.WSB_DISCUSSION_THREAD_TABLE)
            dbm.commit()
            logger.info(f'Fetched {len(posts)} new comments and inserted to DB')
            last_id = posts[0]['id']
        time.sleep(10)


if __name__ == '__main__':
    logger.info('Reserved PID: %s', os.getpid())
    dbm = DBManager(DB_USERNAME, DB_PASS, 'ODS')
    logger.info('Connected to DB successfully')
    tgm = TGManager(QC_TOKEN, TWS_USR_ID, ML_GROUP_ID, LOGS_GROUP_ID)
    logger.info('Connected to TG bot successfully')

    all_tickers = get_year_rus_tickers(dbm)
    dictionary = enchant.Dict('en_US')
    try:
        tgm.send(SCRIPT_NAME + ' launched', 'logs_group')
        main(dbm)
    except Exception as e:
        dbm.rollback()
        tgm.send(SCRIPT_NAME + ': ' + repr(e), 'logs_group')
        logger.error('Error at %s', 'division', exc_info=e)
    logger.info('end of script___________________________________________________')
