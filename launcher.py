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
CURRENT_YEAR = 2013
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

import argparse, logging, sys, _helper_env, data_prep
import logging.handlers
import os
import pandas as pd
import for_finance
from data_prep import DataPrep
from auto_encoder import SAE_train,SAE_predict
import LSTM

'''

'''

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", dest="data_path", default=FILE_ADDRESS,
                    help='Path to the data.')
    parser.add_argument("--curr_year", dest="curr_year", type=int, default=CURRENT_YEAR,
                    help='For which year to perform predicting.')
    parser.add_argument("--verbose", dest="verbose", type=int, default=30,
                    help='Set a level of verbosity')
    args = parser.parse_args()
    return args

def run_algorithm(df):
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
        x_train, y_train, x_test, y_test = DataPrep.train_test_splitting(current_df, PARAMETERS)
        x_train, y_train, x_test, y_test, scaler_x, scaler_y = DataPrep.scale_all_separetly(x_train, y_train, x_test, y_test, PARAMETERS)
    #     Scaling
    #     Performing LSTM
        profit, mape = LSTM.perform_LSTM(x_train, y_train, x_test, y_test, scaler_x, scaler_y, PARAMETERS)

        profits.append(profit)
        mapes.append(mape)
    #     END ACTION
    print("Overall yearly profitability for ", CURRENT_YEAR, " year: ")
    print(sum(profits))


if __name__ == "__main__":
    args = parse_args()
    # Creating logger object for this module
    logger = logging.getLogger('MainLogger')
    _helper_env.set_logger(args.verbose, logger)
    logger.info("______________________________________________")
    logger.info("Logger level set.")
    logger.info("Arguments parsed.")
    logger.info("Setting random seeds...")
    _helper_env.setup_seed()
    logger.info("Random seeds set.")
    data_preparer = DataPrep(logger)
    data_preparer.read_file(FILE_ADDRESS)
    data_preparer.initial_prep()

    encoders, decoders = SAE_train(data_preparer.data, HIDDEN_LAYERS_AUTOENCODER, EPOCHS, BATCH_SIZE_AUTOENCODER, DEPTH_SAE, SHOW_PROGRESS)
    features = SAE_predict(encoders, decoders, data_preparer.data)
    print("Features predicted")
    features = pd.DataFrame(features)
    features = features.loc[:, (features != 0).any(axis=0)]
    INPUT_SHAPE += features.shape[1]
    print('New input shape: ', INPUT_SHAPE)
    PARAMETERS = [SPLIT_PERIOD, HIDDEN_LSTM_UNITS, TEST_TRAIN_SPLIT_COEFFICENT, CURRENT_YEAR, EPOCHS, BATCH_SIZE, INPUT_SHAPE, SHOW_PROGRESS]

    data_preparer.data_preparing(features)
    
    data_preparer.choose_year(CURRENT_YEAR)

    run_algorithm(data_preparer.data)
