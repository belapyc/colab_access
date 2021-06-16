import keras
from keras.layers import Input, Dense, Dropout
from keras.layers.core import Dense
from keras.layers.recurrent import LSTM
from keras.models import Model, Sequential
from sklearn.metrics import mean_squared_error
import numpy as np
from metrics import *
from data_prep import DataPrep
import matplotlib.pyplot as plt
import pandas as pd

def create_LSTM():
    """
    Create an LSTM model of pre-defined topology

    :Example:
    >>> model = create_LSTM()

    Author: Nikita Vasilenko
    """
    model = Sequential ()
    model.add (LSTM (PARAM_UNITS, activation = 'tanh', inner_activation = 'hard_sigmoid' , input_shape =(PARAM_INPUT_SHAPE, 1) ))
    model.add (Dense (output_dim = 1, activation = 'linear'))
    model.compile (loss ="mean_squared_error" , optimizer = "adam")
    return model

def perform_LSTM(x_train, y_train, x_test, y_test, scaler_x, scaler_y, parameters):
    """
    Create, Train and Test LSTM model on financial time series. Return profitability and MAPE error.

    :x_train: training features
    :y_train: training labels
    :x_test: test features
    :y_test: test labels
    :scaler_x: scaler, contains scale factor for feature sets
    :scaler_y: scaler, contains scale factor for labels set
    :show_progress: 1 - to show training epochs, 0 - otherwise

    :Example:
    >>> profit, mape = perform_LSTM(x_train, y_train, x_test, y_test, scaler_x, scaler_y, 1)

    Author: Nikita Vasilenko
    """
    x_train = x_train.reshape (x_train. shape + (1,))
    x_test = x_test.reshape (x_test. shape + (1,))

    model = Sequential ()
    model.add (LSTM (parameters['HIDDEN_LSTM_UNITS'], activation = 'tanh', recurrent_activation="hard_sigmoid", input_shape =(parameters['INPUT_SHAPE'], 1) ))
    model.add (Dense (units=1, activation = 'linear'))
    model.compile (loss ="mean_squared_error" , optimizer = "adam")
    model.fit (x_train, y_train, batch_size = parameters['BATCH_SIZE'], epochs = parameters['EPOCHS'], shuffle = False, verbose = parameters['SHOW_PROGRESS'])

    prediction = model.predict (x_test)
    prediction = scaler_y.inverse_transform (np. array (prediction). reshape ((len( prediction), 1)))
    y_test = scaler_y.inverse_transform (np. array (y_test). reshape ((len( y_test), 1)))

    profit = profitability_test(prediction, y_test)
    mape = mean_absolute_percentage_error(prediction, y_test)
    print('\t\tProfitability: {:0.2f} %'.format(profit))
    print('\t\tMAPE: {:0.2f} %'.format(mape))

    return profit, mape, prediction, [row[0] for row in y_test]

def run_algorithm(data_preparer, year, SPLIT_PERIOD, TEST_TRAIN_SPLIT_COEFFICENT, PARAMETERS, args):
    '''
    Perform LSTM predicting with a sliding window approach
    '''
    print(len(data_preparer.data))
    df = data_preparer.choose_year(year)

    print("LENGTH OF COLUMNS", len(df.columns))
    periods = len(df)
    print(periods)
    profits = []
    mapes = []
    sliding_interval = SPLIT_PERIOD - int(SPLIT_PERIOD*TEST_TRAIN_SPLIT_COEFFICENT)
    predictions_total = np.zeros(shape=(0,1))
    actual_total = np.zeros(shape=(0,1))
    # model = create_LSTM();
    for i in range(0, periods, sliding_interval):
        profit = 0.0
        mape = 0.0
        start_period = i
        if i + SPLIT_PERIOD > periods:
            break
        else:
            end_period = i + SPLIT_PERIOD

    #     MAIN ACTIONa


        print('Current period: ', start_period, ' to ', end_period)
        current_df = df[start_period:end_period]

    #     Deviding
        if args.wavelet:
            x_train, y_train, x_test, y_test = DataPrep.train_test_splitting_wavelet(current_df, PARAMETERS)
        else:
            x_train, y_train, x_test, y_test = DataPrep.train_test_splitting(current_df, PARAMETERS)

        x_train, y_train, x_test, y_test, scaler_x, scaler_y = DataPrep.scale_all_separetly(x_train, y_train, x_test, y_test, PARAMETERS)
    #     Scaling
    #     Performing LSTM
        profit, mape, predictions, actual = perform_LSTM(x_train, y_train, x_test, y_test, scaler_x, scaler_y, PARAMETERS)
        print(predictions.shape)
        print(predictions.dtype)

        predictions_total = np.append(predictions_total,predictions)
        actual_total = np.append(actual_total,actual)

        print(len(actual_total))
        print(len(predictions_total))
        profits.append(profit)
        mapes.append(mape)
    #     END ACTION

    print("Overall yearly profitability for ", year, " year: ")
    print(sum(profits))
    print(args.data_path[8:12]+"  forests"+str(args.forest)+"  random"+ str(args.random_seed))
    plt.figure()
    plt.plot(predictions_total, label="predictions")
    # y_test = scaler_y.inverse_transform (np. array (y_test). reshape ((len( y_test), 1)))
    plt.xlabel("Tick")
    plt.ylabel("Price")
    plt.plot( actual_total, label="actual")
    plt.title("Performance of year "+ str(year))
    plt.legend()
    preds = pd.DataFrame(predictions_total)
    act = pd.DataFrame(actual_total)
    if args.forest:
        save_csv_address_predictions = "predictions/forest/"+args.data_path[8:12]+"_"+"forest_"+str(args.forest)+"_wavelet_"+str(args.wavelet)\
        +"_randomSeed_"+str(args.random_seed)+"_forest_no_"+str(args.forest_no)+"_YEAR_"+str(year)+"predictions"
    else:
        save_csv_address_predictions = "predictions/SAE/"+args.data_path[8:12]+"_"+"forest_"+str(args.forest)+"_wavelet_"+str(args.wavelet)\
        +"_randomSeed_"+str(args.random_seed)+"_forest_no_"+str(args.forest_no)+"_YEAR_"+str(year)+"predictions"

    save_csv_address_actual = "predictions/"+args.data_path[8:12]+"_"+"forest_"+str(args.forest)+"_wavelet_"+str(args.wavelet)\
    +"_randomSeed_"+str(args.random_seed)+"_forest_no_"+str(args.forest_no)+"_YEAR_"+str(year)+"actual"
    preds.to_csv(save_csv_address_predictions)
    act.to_csv(save_csv_address_actual)
    save_plot_address = "Plots/"+args.data_path[8:12]+"_"+"forest_"+str(args.forest)+"_wavelet_"+str(args.wavelet)\
    +"_randomSeed_"+str(args.random_seed)+"_forest_no_"+str(args.forest_no)+"_YEAR_"+str(year)+".png"
    plt.savefig(save_plot_address)
    # plt.show()
    return (sum(profits))
