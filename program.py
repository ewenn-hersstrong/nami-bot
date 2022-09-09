from binance import Client
import time
import pandas as pd
import pandas_ta as ta
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import config


#Setting initial parameters
client = Client(config.API_KEY, config.API_SECRET)
symbol = 'ETHUSDT'
interval = '15m'
start_date = '72 hours ago UTC'
short = 12
long = 26
signal = 9
rsi_length = 14

def getKlines(symbol: str, interval: str, start_date: str, save: bool = False) -> pd.DataFrame:
    """This function downloads data for the chosen coin pair and returns it as pandas dataframe

    Args:
        symbol (str): Verbal coinpair name in CAPS
        interval (str): kline interval (1m, 1h, 1d etc.)
        start_data (str): starting date of the downloadable period
        save (bool): if True - save to csv file

    Returns:
        pd.DataFrame: dataframe with 11 colums: open_time, open, high, low, close, vol, close_time, qa_vol, num_of_trades, taker_buy_ba_vol, taker_buy_qa_vol
    """    ''''''
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
    del df['delete']
    
    if save:
        df.to_csv(f"{symbol}_{interval}_kline_df.csv")
        
    return df

def calcStochRSI(init_df: pd.DataFrame, length: int = 14, upper_b: int = 80, lower_b: int = 20, save: bool = False) -> pd.DataFrame:
    """_summary_

    Args:
        close (pd.DataFrame): _description_
        period (pd.DataFrame): _description_
        length (int, optional): _description_. Defaults to 14.
        upper_b (int, optional): _description_. Defaults to 80.
        lower_b (int, optional): _description_. Defaults to 20.

    Returns:
        pd.DataFrame: _description_
    """    ''''''
    rsi = ta.stochrsi(init_df['close'])
    df = pd.DataFrame()
    df['close_time'] = init_df['close_time']
    df[f'StochRSIk_{length}'] = rsi[f'STOCHRSIk_{length}_14_3_3']
    df[f'StochRSId_{length}'] = rsi[f'STOCHRSId_{length}_14_3_3']
    df.loc[(df[f'StochRSIk_{length}'] < lower_b) & (df[f'StochRSId_{length}'] < lower_b), 'signal'] = 'buy'
    df.loc[(df[f'StochRSIk_{length}'] > upper_b) & (df[f'StochRSId_{length}'] > upper_b), 'signal'] = 'sell'
    
    if save:
        df.to_csv(f"{symbol}_{interval}_StochRSI_df.csv")
        
    return df

def calcMACD(df: pd.DataFrame, ema_short: int, ema_long: int, ema_signal: int, save: bool = False) -> pd.DataFrame:
    """_summary_

    Args:
        df (pd.DataFrame): _description_
        ema_short (int): _description_
        ema_long (int): _description_
        ema_signal (int): _description_

    Returns:
        pd.DataFrame: _description_
    """    
    macd_df = ta.macd(df['close'], fast = ema_short, slow = ema_long, signal = ema_signal)
    macd_df['close_time'] = df['close_time']
    macd_df = macd_df.iloc[:, [3,0,1,2]]
    macd_df.rename(columns = {f'MACD_{ema_short}_{ema_long}_{ema_signal}':'value',
                              f'MACDh_{ema_short}_{ema_long}_{ema_signal}':'histogram',
                              f'MACDs_{ema_short}_{ema_long}_{ema_signal}':'signal'}, inplace = True)
    macd_df['deal']=''
    
    #Checking when does histogram switch its sign (negative to positive - bullish trend, buy)
    for i, v in enumerate(macd_df['histogram']):
        if i>=1 and macd_df['histogram'].iloc[i] < 0 and macd_df['histogram'].iloc[i-1] > 0:
            macd_df['deal'].iloc[i] = 'sell'
        if i>=2 and macd_df['histogram'].iloc[i] > 0 and macd_df['histogram'].iloc[i-1] < 0:
            macd_df['deal'].iloc[i] = 'buy'
            
    if save:
        df.to_csv(f"{symbol}_{interval}_MACD_df.csv")
        
    return macd_df

def drawStichRSIGraph(rsi_df: pd.DataFrame, symbol: str, interval: str, save: bool = False):
    """_summary_

    Args:
        rsi_df (pd.DataFrame): _description_
        symbol (str): _description_
        interval (str): _description_
    """  
    current_time = time.strftime("%Y-%d-%m %H:%M:%S", time.localtime()) 
    df_temp = rsi_df.dropna()
    del df_temp[f'StochRSIk_{rsi_length}'], df_temp[f'StochRSId_{rsi_length}']
    #print('dataframe of signals with timestamps: \n', df_temp)
    rsi_df['StochRSIk_14'].plot(label="StockRSIk", c="b")
    rsi_df['StochRSId_14'].plot(label="StockRSId", c="r")

    #drawing signal markers on the plot
    for i, v in enumerate(df_temp['close_time']):
        for j, v2 in enumerate(rsi_df['close_time']):
            if v == v2:
                if df_temp['signal'].iloc[i] == 'sell':
                    plt.scatter(j, rsi_df['StochRSIk_14'].iloc[j], marker="v", c="#FF0000")
                else:
                    plt.scatter(j, rsi_df['StochRSIk_14'].iloc[j], marker="^", c="#49FF00")

    plt.subplots_adjust(left=0.036, bottom=0.06, right=0.98, top=0.97)
    plt.legend()
    plt.grid("both", axis="both")
    plt.xlabel("K-Line closing time")
    plt.ylabel("Stochastic RSI")
    plt.title(f"{symbol} {interval} RSI at {current_time}")
    x = [0, len(rsi_df)]
    y1 = [20, 20]
    y2 = [80, 80]
    fig2 = plt.plot(x, y1, c="k", linestyle="--")
    fig2 = plt.plot(x, y2, c="k", linestyle="--")
    fig = plt.gcf()
    mng = plt.get_current_fig_manager()
    mng.window.state("zoomed")
    
    if save:
        fig.savefig(f"{symbol}_StochRSI_graph.png")
    
    plt.show()
    
def drawTickerKlines(df: pd.DataFrame, symbol: str, interval: str, save: bool = False):
    x = df['close_time']
    y = df['close']
    
    # coloring the price graph (red-green)
    for x1, x2, y1, y2 in zip(x, x[1:], y, y[1:]):
        if y1 > y2:
            plt.plot([x1, x2], [y1, y2], c="b")
        else:
            plt.plot([x1, x2], [y1, y2], c="r")
    
    current_time = time.strftime("%Y-%d-%m %H:%M:%S", time.localtime())
    plt.subplots_adjust(left=0.036, bottom=0.06, right=0.98, top=0.97)
    plt.legend()
    plt.grid("both", axis="y")
    plt.xlabel("K-Line closing time")
    plt.ylabel("Closing price")
    plt.title(f"{symbol} {interval} at {current_time}")
    fig = plt.gcf()
    mng = plt.get_current_fig_manager()
    mng.window.state("zoomed")
    
    if save:
        fig.savefig(f"{symbol}_graph.png")
        
    plt.show()

def drawMACDGraph(macd_df: pd.DataFrame, symbol: str, interval: str, save: bool = False):
    """_summary_

    Args:
        rsi_df (pd.DataFrame): _description_
        symbol (str): _description_
        interval (str): _description_
    """  
    current_time = time.strftime("%Y-%d-%m %H:%M:%S", time.localtime()) 
    macd_df['value'].plot(label="MACD", c="b")
    macd_df['signal'].plot(label="Signal", c="r")
    x_values = list(range(len(macd_df)))
    
    plt.fill_between(x = x_values, y1 = macd_df['histogram'], interpolate=True, where=macd_df['histogram']>=0, color = 'c')
    plt.fill_between(x = x_values, y1 = macd_df['histogram'], interpolate=True, where=macd_df['histogram']<=0, color = 'm')

    #drawing signal markers on the plot
    for i, v in enumerate(macd_df['deal']):
        if macd_df['deal'].iloc[i] == 'sell':
            plt.scatter(i, macd_df['value'].iloc[i], marker="v", c="#FF0000")
        elif macd_df['deal'].iloc[i] == 'buy':
            plt.scatter(i, macd_df['value'].iloc[i], marker="^", c="#49FF00")

    plt.subplots_adjust(left=0.036, bottom=0.2, right=0.98, top=0.97)
    plt.legend()
    plt.grid("both", axis="y")
    plt.xlabel("K-Line closing time")
    plt.ylabel("MACD")
    plt.title(f"{symbol} {interval} MACD at {current_time}")
    fig = plt.gcf()
    mng = plt.get_current_fig_manager()
    mng.window.state("zoomed")
    
    if save:
        fig.savefig(f"{symbol}_MACD_graph.png")
        
    plt.show()



df = getKlines(symbol, interval, start_date, save = True)
print('initial dataframe: \n',df)

df2 = calcStochRSI(df, length = rsi_length, save = True)
print('StochRSI dataframe: \n', df2)

df_macd = calcMACD(df, short, long, signal, save = True)
print('MACD dataframe: \n', df_macd)

drawTickerKlines(df, symbol, interval, save = True)
drawStichRSIGraph(df2, symbol, interval, save = True)
drawMACDGraph(df_macd, symbol, interval, save = True)
