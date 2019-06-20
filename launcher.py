
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

    LSTM.run_algorithm(data_preparer, CURRENT_YEAR, SPLIT_PERIOD, TEST_TRAIN_SPLIT_COEFFICENT, PARAMETERS)
