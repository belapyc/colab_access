# Location of the file, for which SAE should be performed
FILE_ADDRESS = "Data/TA/BAC_15-mins_9years_TA.csv"
# Period length for splitting the original dataset
SPLIT_PERIOD = 1000
# Hidden layers in LSTM (dimensionality of state vector)
HIDDEN_LSTM_UNITS = 5
# Hidden layers in Autoencoder
HIDDEN_LAYERS_AUTOENCODER = 10
# Depth of Stacked Autoencoder
DEPTH_SAE = 1
# How to split each dataset to train/test?
TEST_TRAIN_SPLIT_COEFFICENT = 0.5
# For which year to perform LSTM predicting
CURRENT_YEAR = 2012
# Amount of training epochs
EPOCHS = 100
# Batch size for LSTM (None for default)
BATCH_SIZE = None
# Batch size for autoencoder
BATCH_SIZE_AUTOENCODER = 256
# Amount of input features
INPUT_SHAPE = 15
# Please set 1 - to print all epochs, 0 - to ignore printing epochs
SHOW_PROGRESS = 0
# Zipping all parameters to one list for easier passing
PARAMETERS = [SPLIT_PERIOD, HIDDEN_LSTM_UNITS, TEST_TRAIN_SPLIT_COEFFICENT, CURRENT_YEAR, EPOCHS, BATCH_SIZE, INPUT_SHAPE, SHOW_PROGRESS]

import pandas as pd
import time
from deepstock import *
from sklearn.ensemble import RandomForestClassifier, RandomTreesEmbedding


start = time.time()

df = pd.read_csv(FILE_ADDRESS)
df = df.loc[:, '<OPEN>':]

deleted_correlated_features(df)
df = normalize(df)


'''
Forest AutoEncoders
'''
eforest = RandomTreesEmbedding(n_estimators=20, max_depth=None, n_jobs=-1, random_state=0)
print("Training forests autoencoder...")
eforest.fit(df)
print("Encoding data...")
encoding = eforest.encode(df)
print("Decoding data...")
# decoded = eforest.decode(encoding)
print("Data decoded")

# print("Training RMSE: ", find_mean_rmse(df.as_matrix(),decoded))

features = encoding
'''
End of forests
'''



# '''
# Stacked Autoencoders
# '''
# print("Training SAE...")
# encoders, decoders = SAE_train(df, HIDDEN_LAYERS_AUTOENCODER, EPOCHS, BATCH_SIZE_AUTOENCODER, DEPTH_SAE, SHOW_PROGRESS)
# print("SAE finished training.")
# print("Predicting features...")
# features = SAE_predict(encoders, decoders, df)
# print("Features predicted")
features = pd.DataFrame(features)
features = features.loc[:, (features != 0).any(axis=0)]
INPUT_SHAPE += features.shape[1]
print('New input shape: ', INPUT_SHAPE)
PARAMETERS = [SPLIT_PERIOD, HIDDEN_LSTM_UNITS, TEST_TRAIN_SPLIT_COEFFICENT, CURRENT_YEAR, EPOCHS, BATCH_SIZE, INPUT_SHAPE, SHOW_PROGRESS]


df_initial = pd.read_csv(FILE_ADDRESS)

df = pd.concat([features, df_initial], axis=1)

data_preparing(df)
df.head(1)

print('Overeall shape of all data: ',df.shape)

df = df.loc[df['<YEAR>'] == CURRENT_YEAR]
# df = df[:1000]
del df['<YEAR>']

print('Dimensinality of data for ', CURRENT_YEAR, ' year is: ', df.shape)


'''
Perform LSTM predicting with a sliding window approach
'''
periods = len(df)
profits = []
mapes = []
sliding_interval = SPLIT_PERIOD - int(SPLIT_PERIOD*TEST_TRAIN_SPLIT_COEFFICENT)
# model = create_LSTM();
for i in range(0, periods, sliding_interval):
    profit = 0.0
    mape = 0.0
    start_period = i
    if i + SPLIT_PERIOD > periods:
        break
    else:
        end_period = i + SPLIT_PERIOD

#     MAIN ACTION

    print('Current period: ', start_period, ' to ', end_period)
    current_df = df[start_period:end_period]
#     Deviding
    x_train, y_train, x_test, y_test = train_test_splitting(current_df, PARAMETERS)
#     Scaling
    x_train, y_train, x_test, y_test, scaler_x, scaler_y = scale_all_separetly(x_train, y_train, x_test, y_test, PARAMETERS)
#     Performing LSTM
    profit, mape = perform_LSTM(x_train, y_train, x_test, y_test, scaler_x, scaler_y, PARAMETERS)

    profits.append(profit)
    mapes.append(mape)
#     END ACTION

end = time.time()
print("Time spent: ", (end-start), " seconds.")

print(profits)

print("Overall yearly profitability for ", CURRENT_YEAR, " year: ")
print(sum(profits))
