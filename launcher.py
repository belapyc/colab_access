
import argparse, logging, sys, _helper_env, data_prep
import logging.handlers
import os
import time
import pandas as pd
from data_prep import DataPrep
from auto_encoder import SAE_train,SAE_predict
import LSTM
from __init__ import PARAMETERS, FILE_ADDRESS
from sklearn.ensemble import RandomTreesEmbedding



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
    parser.add_argument("--forest_no", dest='forest_no', type=int, default=10)
    parser.add_argument("--random_seed", dest = 'random_seed', type=int, default=0)
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
    _helper_env.setup_seed(args.random_seed)
    logger.info("Random seeds set.")
    data_preparer = DataPrep(logger)
    data_preparer.read_file(args.data_path)

    print("INPUT SHAPE", PARAMETERS['INPUT_SHAPE'])
    data_preparer.initial_prep()
    init_shape = len(data_preparer.data.columns)
    PARAMETERS['INPUT_SHAPE'] = init_shape
    print(init_shape)


    start = time.time()
    forest = args.forest
    encoding_start = time.time()
    if forest:
        rand = RandomTreesEmbedding(n_estimators=args.forest_no, max_depth = None, random_state = args.random_seed)
        rand.fit(data_preparer.data)
        encoded = rand.apply(data_preparer.data)
        features = pd.DataFrame(encoded)
    else:
        encoders, decoders = SAE_train(data_preparer.data, PARAMETERS['HIDDEN_LAYERS_AUTOENCODER'], PARAMETERS['EPOCHS'], PARAMETERS['BATCH_SIZE_AUTOENCODER'], PARAMETERS['DEPTH_SAE'], PARAMETERS['SHOW_PROGRESS'])
        features = SAE_predict(encoders, decoders, data_preparer.data)
        print("Features predicted")
        features = pd.DataFrame(features)
        features = features.loc[:, (features != 0).any(axis=0)]
    encoding_time = time.time() - encoding_start

    # Updating Input Shape as we added features from SAE
    print(PARAMETERS['INPUT_SHAPE'])
    #print(features.shape[1])
    PARAMETERS['INPUT_SHAPE'] += features.shape[1]
    print(PARAMETERS['INPUT_SHAPE'])




    # Removing unneccessary features and adding NEXT and YEAR variables
    data_preparer.data_preparing(features)
    PARAMETERS['INPUT_SHAPE'] = len(data_preparer.data.columns)
    print(PARAMETERS['INPUT_SHAPE'])

    PARAMETERS['INPUT_SHAPE'] -= 2 # Removing beforehand since we remove year and next from train dataset
    total_profits = 0.0
    profits_per_year = {}
    for year in PARAMETERS['ALL_YEARS']:
        profit = LSTM.run_algorithm(data_preparer, year, PARAMETERS['SPLIT_PERIOD'], PARAMETERS['TEST_TRAIN_SPLIT_COEFFICENT'], PARAMETERS, args)
        total_profits = profit + total_profits
        profits_per_year[year] = profit
    end = time.time()
    elapsed_time = end-start
    f= open("results_forest_"+str(args.forest)+"wavelet_"+str(args.wavelet)+".txt","a")

    print("Writing to a file...")
    f.write("\n============================================================\n")
    f.write("forest: "+ str(args.forest) + "\n")
    f.write("Features predicted: "+ str(features.shape[1]) + "\n")
    f.write("Predicted based on "+ str(PARAMETERS['INPUT_SHAPE'])+ " features" + "\n")
    f.write("Random seed: "+ str(args.random_seed) + "\n")
    f.write("wavelet: "+ str(args.wavelet) + "\n")
    f.write("total time: " + str(elapsed_time) + "\n")
    f.write("encoding time: " + str(encoding_time) + "\n")
    f.write(str(PARAMETERS) + "\n")
    f.write("Total profitability: "+ str(total_profits)+"\n")
    f.write("Average profitability: " + str(total_profits/len(PARAMETERS['ALL_YEARS']))+"\n")
    f.write("File path: " + str(args.data_path) + "\n")

    f.write(str(profits_per_year))

    f.close()
