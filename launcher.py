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

import argparse, logging, sys#, _helper_env
import pandas as pd
from deepstock import *

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

def set_logger(level):
    '''
    Setting the levels for logger output.
    Creates File and Stream handlers for outputting messages to a file and
    console respectively.
    File output level is set to DEBUG, ie all of the messages will be output
    to the file regardless of the level
    Console output level can be set by user through passing --verbose
    argument in command line. The default level for console output is WARNING
    Logging levels:
    10 - debug, 20 - info, 30 - warning, 40 - error, 50 - critical

    :level: integer, representing level of output to console
    '''

    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Creating console handler for the logger
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    ch.setLevel(level)

    logPath = "logs"
    fileName = "logLauncher"
    # Creating file handler for logger
    fh = logging.FileHandler("{0}/{1}.log".format(logPath, fileName))
    fh.setFormatter(formatter)
    fh.setLevel(logging.DEBUG)

    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.info("Logger is set.")

# https://docs.python.org/3/howto/logging-cookbook.html
# Using logging in multiple modules (for further reference)


if __name__ == "__main__":
    args = parse_args()
    # Creating logger object for this module
    logger = logging.getLogger('MainLogger')
    set_logger(args.verbose)
    logger.info("______________________________________________")
    logger.info("Logger level set.")
    logger.info("Arguments parsed.")
    logger.info("Setting random seeds...")
    _helper_env.setup_seed()
    logger.info("Random seeds set.")

    # df = pd.read_csv(FILE_ADDRESS)
    # df = df.loc[:, '<OPEN>':]
    # print(df.columns)
    # deleted_correlated_features(df)
    # print(df.columns)
    # df = normalize(df)
    #
    # '''
    # Stacked Autoencoders
    # '''
    # print("Training SAE...")
    # encoders, decoders = SAE_train(df, HIDDEN_LAYERS_AUTOENCODER, EPOCHS, BATCH_SIZE_AUTOENCODER, DEPTH_SAE, SHOW_PROGRESS)
    # print("SAE finished training.")
    # print("Predicting features...")
    # features = SAE_predict(encoders, decoders, df)
    # print("Features predicted")
    # features = pd.DataFrame(features)
    # print(features.head(5))
