FILE_ADDRESS = "Data/Raw/world_industrials_index/DJworld_indust_15mins_2009-2018.csv"
FILE_SAVE_ADDRESS = "Data/TA/DJworld_indust_15mins_2009-2018_TA.csv"

import pandas as pd
from talib.abstract import *
import talib

df = pd.read_csv(FILE_ADDRESS)

talib.get_functions()[0:10]

output_sma = SMA(df, timeperiod=25, price = '<CLOSE>')
df['<SMA>'] = output_sma


output_macd = MACD(df, price = '<CLOSE>')
df['<MACD>'] = output_macd['macd']
df['<MACDSIGNAL>'] = output_macd['macdsignal']
df['<MACDHIST>'] = output_macd['macdhist']

output_cci = CCI(df, prices = ['<HIGH>', '<LOW>','<CLOSE>'])
df['<CCI>'] = output_cci

atr = ATR(df, prices = ['<HIGH>', '<LOW>','<CLOSE>'])
df['<ATR>'] = atr

boll = BBANDS(df, price = '<LOW>')
df['<UPPERBAND>'] = boll['upperband']
df['<MIDDLEBAND>'] = boll['middleband']
df['<LOWERBAND>'] = boll['lowerband']

ema20 = EMA(df, price = '<CLOSE>', timeperiod=20)
df['<EMA20>'] = ema20

ma5 = MA(df, timeperiod=5, price='<CLOSE>')
ma10 = MA(df, timeperiod=10, price='<CLOSE>')

df['<MA5>'] = ma5
df['<MA10>'] = ma10

roc = ROC(df, price='<CLOSE>')
df['<ROC>'] = roc

willr = WILLR(df, prices = ['<HIGH>', '<LOW>','<CLOSE>'])
df['<WILLR>'] = willr

mom = MOM(df, price='<CLOSE>')
df['<MOM>'] = mom

rsi = RSI(df, price='<CLOSE>')
df['<RSI>'] = rsi

df = df.dropna()

df.to_csv(FILE_SAVE_ADDRESS, index = False)
