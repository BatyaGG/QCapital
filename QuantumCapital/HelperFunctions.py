from datetime import datetime


# def get_offset_to_date(ib, ticker: str, today_dt: datetime, goto_dt: datetime = None) -> int:
#     """
#     Returns number of exchange working days backwards to reach given goto_dt date
#     If goto_dt not given, then it returns the offset for last July 1st date
#     :param today_dt: today date datetime
#     :param goto_dt: go backward datetime
#     :return: number of offset days
#     """
#     if goto_dt is None:
#         if today_dt.month < 7:
#             goto_dt = datetime(today_dt.year - 1, 7, 1, tzinfo=pytz.UTC)
#         else:
#             goto_dt = datetime(today_dt.year, 7, 1, tzinfo=pytz.UTC)
#     nyse = mcal.get_calendar('NYSE')
#     r_days = nyse.valid_days(start_date=goto_dt.strftime('%Y-%m-%d'),
#                              end_date=today_dt.strftime('%Y-%m-%d'))
#     dur_str = len(r_days) + 3
#     bars = ib.reqHistoricalData(Stock(ticker, 'SMART/ISLAND', 'USD'),
#                                 endDateTime=today_dt,
#                                 durationStr=f'{dur_str} D',
#                                 barSizeSetting='1 min',
#                                 whatToShow='TRADES',
#                                 useRTH=True,
#                                 formatDate=2,
#                                 timeout=60)
#     if not bars:
#         return None
#     fd = bars[0].date.date()
#
#     while fd < goto_dt.date():
#         dur_str -= 1
#         bars = ib.reqHistoricalData(Stock(ticker, 'SMART/ISLAND', 'USD'),
#                                     endDateTime=today_dt,
#                                     durationStr=f'{dur_str} D',
#                                     barSizeSetting='1 min',
#                                     whatToShow='TRADES',
#                                     useRTH=True,
#                                     formatDate=2,
#                                     timeout=60)
#         fd = bars[0].date.date()
#         logger.info(str((fd, dur_str)))
#     return dur_str


def get_russel_dt_range(dt):
    if 1 <= dt.month <= 6:
        start_dt = datetime(dt.year - 1, 7, 1, 0, 0, 0)
        end_dt = datetime(dt.year, 6, 30, 23, 59, 59)
    else:
        start_dt = datetime(dt.year, 7, 1, 0, 0, 0)
        end_dt = datetime(dt.year + 1, 6, 30, 23, 59, 59)
    return start_dt, end_dt



def get_year_rus_tickers(dbm, year=None):
    if year is not None:
        tickers = dbm.select_df('select tickers '
                                'from rus_tickers_historical '
                                f'where yr = {year}')
    else:
        tickers = dbm.select_df('select tickers '
                                'from rus_tickers_historical '
                                'order by yr desc limit 1')
    tickers = tickers.iloc[0].tickers
    tickers = [ticker.split()[0].upper() for ticker in tickers]
    tickers = [ticker.replace('/', '_') for ticker in tickers]
    return tickers
