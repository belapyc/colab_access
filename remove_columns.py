
FILE_SAVE_ADDRESS = "Data/TA/RL_15mins-2009-2018_TA.csv"

import pandas as pd


df = pd.read_csv(FILE_SAVE_ADDRESS)


del df['<TICKER>']
del df['<PER>']

df.to_csv(FILE_SAVE_ADDRESS, index = False)
