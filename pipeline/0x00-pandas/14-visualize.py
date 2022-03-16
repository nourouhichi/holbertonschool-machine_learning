#!/usr/bin/env python3

from datetime import date
import matplotlib.pyplot as plt
import pandas as pd
from_file = __import__('2-from_file').from_file

df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')
df = df.loc[df["Timestamp"] >= 1483228800]

# YOUR CODE HERE
df = df.drop(columns=['Weighted_Price'])
df = df.rename(columns={"Timestamp": "Date"})
df["Date"] = pd.to_datetime(df["Date"], unit='s')
df = df.set_index('Date')
df["Volume_(BTC)"] = df["Volume_(BTC)"].fillna(0)
df["Volume_(Currency)"] = df["Volume_(Currency)"].fillna(0)
df["Close"] = df["Close"].fillna(method='ffill')
df["High"] = df["High"].fillna(df['Close'].shift(1))
df["Low"] = df["Low"].fillna(df['Close'].shift(1))
df["Open"] = df["Open"].fillna(df['Close'].shift(1))


def op(arr):
    return arr[0]


def cl(arr):
    return arr[-1]


df1 = pd.DataFrame()
df1['High'] = df['High'].resample('D').max()
df1['Low'] = df['Low'].resample('D').min()
df1['Volume_(BTC)'] = df['Volume_(BTC)'].resample('D').sum()
df1['Volume_(Currency)'] = df['Volume_(Currency)'].resample('D').sum()
df1["Open"] = df.Open.resample('D').apply(op)
df1["close"] = df.Close.resample('D').apply(cl)

df1.plot()
plt.show()
