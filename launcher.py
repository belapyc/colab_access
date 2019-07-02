
import argparse, logging, sys, _helper_env, data_prep
import logging.handlers
import os
import pandas as pd
import for_finance
from data_prep import DataPrep
from auto_encoder import SAE_train,SAE_predict
import LSTM
from __init__ import PARAMETERS, FILE_ADDRESS
from sklearn.ensemble import RandomForestClassifier, RandomTreesEmbedding



'''

'''
# https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", dest="data_path", default=FILE_ADDRESS,
                    help='Path to the data.')
    parser.add_argument("--curr_year", dest="curr_year", type=int, default=PARAMETERS['CURRENT_YEAR'],
                    help='For which year to perform predicting.')
    parser.add_argument("--verbose", dest="verbose", type=int, default=30,
                    help='Set a level of verbosity')
    parser.add_argument("--wavelet", type=str2bool, nargs='?',
                        const=True, default=False,
                        help="Activate wavelet transform.")
    parser.add_argument("--forest", type=str2bool, nargs='?',
                        const=True, default=False,
                        help="Activate encoding forest.")                    
    parser.add_argument("--all_years", type=str2bool, nargs='?',
                        const=True, default=False,
                        help="Perform algo on all years.")
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
    if(not args.all_years):
        print("performing on all years")
    _helper_env.setup_seed()
    logger.info("Random seeds set.")
    data_preparer = DataPrep(logger)
    data_preparer.read_file(FILE_ADDRESS)
    data_preparer.initial_prep()

    forest = True
    if forest:
        eforest = RandomTreesEmbedding(n_estimators=15, max_depth=None, n_jobs=-1, random_state=0)
        print("Training forests autoencoder...")
        eforest.fit(data_preparer.data)
        print("Encoding data...")
        encoding = eforest.encode(data_preparer.data)
        features = encoding
        features = pd.DataFrame(features)
        features = features.loc[:, (features != 0).any(axis=0)]
    else:
        encoders, decoders = SAE_train(data_preparer.data, PARAMETERS['HIDDEN_LAYERS_AUTOENCODER'], PARAMETERS['EPOCHS'], PARAMETERS['BATCH_SIZE_AUTOENCODER'], PARAMETERS['DEPTH_SAE'], PARAMETERS['SHOW_PROGRESS'])
        features = SAE_predict(encoders, decoders, data_preparer.data)
        print("Features predicted")
        features = pd.DataFrame(features)
        features = features.loc[:, (features != 0).any(axis=0)]

    # Updating Input Shape as we added features from SAE
    print(features.shape[1])
    PARAMETERS['INPUT_SHAPE'] += features.shape[1]
    print(PARAMETERS['INPUT_SHAPE'])




    # Removing unneccessary features and adding NEXT and YEAR variables
    data_preparer.data_preparing(features)

    profits_per_year = {}
    for year in PARAMETERS['ALL_YEARS']:
        profits = LSTM.run_algorithm(data_preparer, year, PARAMETERS['SPLIT_PERIOD'], PARAMETERS['TEST_TRAIN_SPLIT_COEFFICENT'], PARAMETERS, args.wavelet)
        profits_per_year[year] = profits
    print(profits_per_year)
    total_profits = 0
    for key,value in profits_per_year:
        total_profits = total_profits + value
    print(total_profits/len(PARAMETERS['ALL_YEARS']))
