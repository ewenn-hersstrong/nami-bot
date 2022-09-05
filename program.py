from binance import Client
import time
import config
import pandas as pd
import pandas_ta as ta
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

client = Client(config.API_KEY, config.API_SECRET)

symbol = 'ETHUSDT'
interval = '5m'
start_date = '24 hours ago UTC'
short = 12
long = 26
signal = 9

def getKlines(symbol: str, interval: str, start_data: str) -> pd.DataFrame:
    data = client.get_historical_klines(symbol, interval, start_date)
    header = [
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "vol",
        "close_time",
        "qa_vol",
        "num_of_trades",
        "taker_buy_ba_vol",
        "taker_buy_qa_vol",
        "delete"
    ]

    for i, values in enumerate(data):
        for j, val in enumerate(values):
            if (j == 0 or j == 6):
                data[i][j] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(float(data[i][j] / 1000)))
            elif (j == 8):
                data[i][j] = int(data[i][j])
            else:
                data[i][j] = float(data[i][j])

    df = pd.DataFrame(data, columns = header)
    return df

df = getKlines(symbol, interval, start_date)
print(df.head())

def calcStochRSI(close: pd.DataFrame, period: pd.DataFrame, length: int = 14, upper_b: int = 80, lower_b: int = 20) -> pd.DataFrame:
    rsi = ta.stochrsi(close)
    df = pd.DataFrame()
    df['close_time'] = period
    df[f'StochRSIk_{length}'] = rsi[f'STOCHRSIk_{length}_14_3_3']
    df[f'StochRSId_{length}'] = rsi[f'STOCHRSId_{length}_14_3_3']
    df.loc[(df[f'StochRSIk_{length}'] < lower_b) & (df[f'StochRSId_{length}'] < lower_b), 'signal'] = 'buy'
    df.loc[(df[f'StochRSIk_{length}'] > upper_b) & (df[f'StochRSId_{length}'] > upper_b), 'signal'] = 'sell'
    return df
        
length = 14
df2 = calcStochRSI(close = df['close'], period = df['close_time'], length = length)
print(df2)
df_temp = df2.dropna()
del df_temp[f'StochRSIk_{length}'], df_temp[f'StochRSId_{length}']
print(df_temp)
df2['StochRSIk_14'].plot(label="StockRSIk", c="b")
df2['StochRSId_14'].plot(label="StockRSId", c="r")
for i, v in enumerate(df_temp['close_time']):
    for j, v2 in enumerate(df2['close_time']):
        if v == v2:
            if df_temp['signal'].iloc[i] == 'sell':
                plt.scatter(j, df2['StochRSIk_14'].iloc[j], marker="v", c="#FF0000")
            else:
                plt.scatter(j, df2['StochRSIk_14'].iloc[j], marker="^", c="#49FF00")
plt.legend()
x = [0, len(df)]
y1 = [20, 20]
y2 = [80, 80]
fig2 = plt.plot(x, y1, c="k", linestyle="--")
fig2 = plt.plot(x, y2, c="k", linestyle="--")
plt.title('StochRSI')
plt.show()

