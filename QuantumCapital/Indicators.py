import numpy as np


def macd(df, field='close', slow_period=26, fast_period=12, signal_period=9):
    start = 2 * slow_period
    emaslow = df.iloc[-start:][field].ewm(span=slow_period, adjust=False).mean()
    emafast = df.iloc[-start:][field].ewm(span=fast_period, adjust=False).mean()
    macd_line = emafast - emaslow
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    macd_hist = macd_line - signal_line
    df.loc[df.iloc[-1].name, 'macd_line'] = macd_line[-1]
    df.loc[df.iloc[-1].name, 'signal_line'] = signal_line[-1]
    df.loc[df.iloc[-1].name, 'macd_hist'] = macd_hist[-1]


def atr(df, num_bars=14):
    if df.shape[0] == 1:
        df.loc[df.iloc[-1].name, 'ATR'] = df.loc[df.iloc[-1].name, 'high'] - df.loc[df.iloc[-1].name, 'low']
        return
    bar = df.iloc[-1]
    last_bar = df.iloc[-2]
    tr = max(bar['high'] - bar['low'], abs(bar['high'] - last_bar['close']),
             abs(bar['low'] - last_bar['close']))
    if df.shape[0] < num_bars:
        df.loc[df.iloc[-1].name, 'ATR'] = (last_bar['ATR'] + tr) / 2
        return
    df.loc[df.iloc[-1].name, 'ATR'] = ((num_bars - 1) * last_bar['ATR'] + tr) / num_bars


def super_trend(df, multiplier=2):
    if 'ATR' not in list(df):
        atr(df)
    elif np.isnan(df.iloc[-1]['ATR']):
        atr(df)

    if df.shape[0] < 2:
        df.loc[df.iloc[-1].name, 'final_ub'] = 0.0
        df.loc[df.iloc[-1].name, 'final_lb'] = 0.0
        df.loc[df.iloc[-1].name, 'super_trend'] = 0.0
        df.loc[df.iloc[-1].name, 'trend_direction'] = -1
        return

    last_bar = df.iloc[-2]
    basic_ub = (df.loc[df.iloc[-1].name, 'high'] + df.loc[df.iloc[-1].name, 'low']) / 2 + multiplier * df.loc[df.iloc[-1].name, 'ATR']
    basic_lb = (df.loc[df.iloc[-1].name, 'high'] + df.loc[df.iloc[-1].name, 'low']) / 2 - multiplier * df.loc[df.iloc[-1].name, 'ATR']
    df.loc[df.iloc[-1].name, 'final_ub'] = basic_ub if basic_ub < last_bar['final_ub'] or last_bar['close'] > last_bar['final_ub'] else \
        last_bar['final_ub']
    df.loc[df.iloc[-1].name, 'final_lb'] = basic_lb if basic_lb > last_bar['final_lb'] or last_bar['close'] < last_bar['final_lb'] else \
        last_bar['final_lb']

    df.loc[df.iloc[-1].name, 'super_trend'] = df.iloc[-1]['final_ub'] if last_bar['super_trend'] == last_bar['final_ub']\
                                                                         and df.iloc[-1]['close'] <=  df.iloc[-1]['final_ub'] \
        else df.iloc[-1]['final_lb'] if last_bar['super_trend'] == last_bar['final_ub'] and df.iloc[-1]['close'] > df.iloc[-1]['final_ub'] else df.iloc[-1]['final_lb'] if \
            last_bar['super_trend'] == last_bar['final_lb'] and df.iloc[-1]['close'] >= df.iloc[-1]['final_lb'] else df.iloc[-1]['final_ub'] if \
            last_bar['super_trend'] == last_bar['final_lb'] and df.iloc[-1]['close'] < df.iloc[-1]['final_lb'] else 0.00
    df.loc[df.iloc[-1].name, 'trend_direction'] = 1 if df.iloc[-1]['close'] > df.iloc[-1]['super_trend'] else 0

