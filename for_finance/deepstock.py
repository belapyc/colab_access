# Importing relevant functions from open-source libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pywt
import tensorflow as tf
import keras
from keras.layers import Input, Dense, Dropout
from keras.layers.core import Dense
from keras.layers.recurrent import LSTM
from keras.models import Model, Sequential
from sklearn.metrics import mean_squared_error
from scipy import stats
from sklearn import preprocessing
from math import sqrt


def apply_wavelet(df, levels):
    """
    This functions performa a 1-layer wavelet transform for all rows and all columns of a given Pandas Dataframe

    :df: Pandas dataframe, for which to perform wavelet transform
    :levels: The amount of consiquent wavelet transforms to perform

	:Example:
	# Read dataframe from file:
    >>> df = pd.read_csv(FILE_ADDRESS)
    # Apply 1-level wavelet transform and save to a new dataframe
    >>> waved_df = apply_wavelet(df, 1)
    """
    waved_df = pd.DataFrame()
    for items in df.items():
        current_name = items[0]
        current_series = items[1]
#         N- layer Wavelet Transform with Haar Function
        cA = pywt.wavedec(current_series, 'haar', level=levels)
        waved_df[current_name] = cA[0]
    return waved_df

def normalize(df):
    """
    This functions performa a normalization for all rows and all columns of a given Pandas Dataframe

    Type of normalization: MinMax scaling, changing ranges of all values to a range [0..1]
    :df: Pandas dataframe, for which to perform normalisation

	:Example:
	# Read dataframe from file:
    >>> df = pd.read_csv(FILE_ADDRESS)
    # Apply MinMax normalization
    >>> df = normalize(df, 1)
    """
    result = df.copy()
    for feature_name in df.columns:
        max_value = df[feature_name].max()
        min_value = df[feature_name].min()
        result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
    return result

def deleted_correlated_features(df):
    """
    Deleting features, which are pre-defined as correlated.

    :df: Pandas dataframe, where to remove correlated features

    :Example:
	# Read dataframe from file:
    >>> df = pd.read_csv(FILE_ADDRESS)
    # Remove pre-defined correlated features
    >>> df = deleted_correlated_features(df)
    """
    del df['<UPPERBAND>']
    del df['<MIDDLEBAND>']
    del df['<LOWERBAND>']
    del df['<EMA20>']
    del df['<MA5>']
    del df['<MA10>']

def rmse(input_layer, reconstruction_layer):
    """
    Function returning a root mean square error of two input vectors.

    :input_layer: first vector
    :reconstruction_layer: second vector

    :Example:
    >>> rmse =  rmse(input_layer, reconstruction_layer)
    """
    return sqrt(mean_squared_error(input_layer, reconstruction_layer))

def find_mean_rmse(test_x,reconstruction_layers):
    """
    Function returning a root mean square error of two input matrices


    :test_x: first matrix
    :reconstruction_layers: second matrix

    :Example:
    >>> mean_rmse = find_mean_rmse(test_x,reconstruction_layers)
    """
    sum_rmse = 0
    for i in range(0, len(test_x)):
        input_array = test_x[i]
        reconstructed_array = reconstruction_layers[i]
        current_rmse = rmse(input_array, reconstructed_array)
        sum_rmse += current_rmse
    return(sum_rmse/len(test_x))

def AE_train(dataframe, hidden_layer_size, epochs_amount, batch_amount, show_progress):
    """
    Creating and training single-layer autoencoder.
    Using Keras API for tensorflow environment.

    :dataframe: which dataframe to use for training a single-layer autoencoder
    :hidden_layer_size: amount of units in the hidden layer
    :epochs_amount: amount of epochs
    :batch_amount: batch size (16, 32, 64, 128, 256)
    :show_progress: 1 - to show epochs, 0 - to ignore epochs

    :Example:
    >>> encoder, decoder = AE_train(df, 10, 100, 256, 1)
    """
    input_layer_size = dataframe.shape[1]
    # Creating input layer of the neural network
    input_sample = Input(shape=(input_layer_size,))
    # Second (hidden) layer of the neural network with ReLU activation function
    encoded = Dense(hidden_layer_size, activation='relu')(input_sample)
    # Third (output) layer of the neural network with Logistic Sigmoid activation function
    decoded = Dense(input_layer_size, activation='sigmoid')(encoded)
    # Initialising a model, mapping from input layer to the output layer (input -> reconstruction)
    autoencoder = Model(input_sample, decoded)
    # Splitting the model into two parts. Encoder - first two layers of the neural network
    encoder = Model(input_sample, encoded)
    # Creating an additional tensor (layer), effectively representing the encoded input (middle layer)
    encoded_input = Input(shape=(hidden_layer_size,))
    # Reconstructing decoder layer
    decoder_layer = autoencoder.layers[-1](encoded_input)
    # Create the decoder model
    decoder = Model(encoded_input, decoder_layer)
    # Compiling autoencoder model
    autoencoder.compile(optimizer='Adadelta', loss='binary_crossentropy')
    print('... [training in process] ...')
    # Training autoencoder
    autoencoder.fit(dataframe, dataframe,
                epochs=epochs_amount,
                batch_size=batch_amount,
                shuffle=True,
                verbose = show_progress)
    # Computing the training error (RMSE)
    hidden_layer = encoder.predict(dataframe)
    reconstruction_layer = decoder.predict(hidden_layer)
    print("Training RMSE: ", find_mean_rmse(dataframe.as_matrix(),reconstruction_layer))
    return encoder, decoder

def AE_predict(encoder, decoder, df):
    """
    Given a trained mode, fit the data to the model and get output from the hidden layer

    :encoder: trained Encoder model
    :decoder: trained Decoder model
    :df: data to fit to the model

    :Example:
    >>> features = AE_predict(encoder, decoder, df)
    """
    hidden_layer = encoder.predict(df)
#     The reconstruction layer is cast out (we dont need it anymore)
    reconstruction_layer = decoder.predict(hidden_layer)
    return hidden_layer

def SAE_train(dataframe, hidden_layer_size, epochs_amount, batch_amount, depth, show_progress):
    """
    Train a series (stack) of single-layer autoencoders

    :dataframe: which dataframe to use for training a single-layer autoencoder
    :hidden_layer_size: amount of units in the hidden layer
    :epochs_amount: amount of epochs
    :batch_amount: batch size (16, 32, 64, 128, 256)
    :show_progress: 1 - to show epochs, 0 - to ignore epochs

    :Example:
    >>> encoder, decoder = SAE_train(df, 10, 100, 256, 4, 1)
    """
    encoders = []
    decoders = []
    print('Training AutoEncoder #1')
    encoder, decoder = AE_train(dataframe, hidden_layer_size, epochs_amount, batch_amount, show_progress)
    hidden_layer = AE_predict(encoder, decoder, dataframe)
    encoders.append(encoder)
    decoders.append(decoder)

    for i in range(0, depth - 1):
        print('Training AutoEncoder #', (i + 2))
        encoder, decoder = AE_train(pd.DataFrame(hidden_layer), hidden_layer_size, epochs_amount, batch_amount, show_progress)
        hidden_layer = AE_predict(encoder, decoder, hidden_layer)
        encoders.append(encoder)
        decoders.append(decoder)
    return encoders, decoders

def SAE_predict(encoders, decoders, dataframe):
    """
    Fit data to a trained stacked autoencoder

    :encoders: a LIST of trained encoders
    :decoders: a LIST of trained decoders
    :dataframe: data to fit to the model

    :Example:
    >>> features = SAE_predict(encoders, decoders, df)
    """
    hidden_layer = AE_predict(encoders[0], decoders[0], dataframe)

    for i in range(1, len(encoders)):
        hidden_layer = AE_predict(encoders[i], decoders[i], hidden_layer)

    return hidden_layer

def create_LSTM():
    """
    Create an LSTM model of pre-defined topology

    :Example:
    >>> model = create_LSTM()
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
    """
    x_train = x_train.reshape (x_train. shape + (1,))
    x_test = x_test.reshape (x_test. shape + (1,))

    model = Sequential ()
    model.add (LSTM (parameters[1], activation = 'tanh', inner_activation = 'hard_sigmoid' , input_shape =(parameters[6], 1) ))
    model.add (Dense (output_dim = 1, activation = 'linear'))
    model.compile (loss ="mean_squared_error" , optimizer = "adam")
    model.fit (x_train, y_train, batch_size = parameters[5], epochs = parameters[4], shuffle = False, verbose = parameters[7])

    prediction = model.predict (x_test)
    prediction = scaler_y.inverse_transform (np. array (prediction). reshape ((len( prediction), 1)))
    y_test = scaler_y.inverse_transform (np. array (y_test). reshape ((len( y_test), 1)))

    profit = profitability_test(prediction, y_test)
    mape = mean_absolute_percentage_error(prediction, y_test)
    print('\t\tProfitability: {:0.2f} %'.format(profit))
    print('\t\tMAPE: {:0.2f} %'.format(mape))

    # # Plot
    # plt.plot(prediction, label="predictions")
    # # y_test = scaler_y.inverse_transform (np. array (y_test). reshape ((len( y_test), 1)))
    # plt.plot( [row[0] for row in y_test], label="actual")
    # plt.show() 
    return profit, mape


def data_preparing(df):
    """
    Given a data for several dates removing unneeded features, and prepare for yearly splitting.
    Adding additional feature: NEXT, which is essentially close price of the next time slot.
    Adding additional feature: YEAR, which is just a year.

    :df: dataframe to process.


    :Example:
    >>>data_preparing(df)
    """
    years = []
    for index, row in df.iterrows():
        years.append(int(str(row['<DATE>'])[:4]))
    df['<YEAR>'] = years

    # Deleting collerated and unneeded features
    del df['<DATE>']
    del df['<TIME>']
    del df['<UPPERBAND>']
    del df['<MIDDLEBAND>']
    del df['<LOWERBAND>']
    del df['<EMA20>']
    del df['<MA5>']
    del df['<MA10>']
#     Creating one step ahead price
    df['<NEXT>'] = (df['<CLOSE>']).shift(-1)
#     Removing last row!
    df = df.dropna()

def train_test_splitting(df, parameters):
    """
    Split dataframe to four subsets.

    :df: dataframe to split.
    :parameters: list of user - defined parameters

    :Example:
    >>> x_train, y_train, x_test, y_test = train_test_splitting(df, PARAMETERS)

    """
    x = (df.loc[:, : '<RSI>']).as_matrix()
    y = (df.loc[:, '<NEXT>':]).as_matrix()

    train_end = int(parameters[0] * parameters[2])
    x_train = x [0: train_end,]
    x_test = x[ train_end +1:len(x),]
    y_train = y [0: train_end]
    y_test = y[ train_end +1:len(y)]
    return x_train, y_train, x_test, y_test

def mean_absolute_percentage_error(y_true, y_pred):
    """
    Return MAPE metrics - Mean absolute percentage error between two vectors

    :y_true: true values vector
    :y_pred: predicted values vector

    :Example:
    >>> mape = mean_absolute_percentage_error(y_true, y_pred)
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true))*100

def profitability_test(predicted, actual):
    """Profitability Test
    given two vectors of prices: predicted and actual, compute percentage profitability.
    The logic is as follows: if the predicted NEXT value is bigger than actual current - simulate BUY operation, otherwise - simulate SELL operation
    Returns profitability in percents.

    :predicted: vector of predicted prices
    :actual: vector of actual prices

    :Example:
    >>> profit = profitability_test(predicted, actual,0,0)
    """
    R = 0
#     Not used because only buy/sell strategy
    trades = 0
    for i in range(len(actual) - 1):
        yt = actual[i][0]
        yt_next = actual[i+1][0]
        if (predicted[i+1][0] > predicted[i][0]):
            R += (yt_next - yt) / yt
        if (predicted[i+1][0] < predicted[i][0]):
            R += (yt - yt_next) / yt
    return(R * 100)

def scale_all_separetly(x_train, y_train, x_test, y_test, parameters):
    """
    Scaling sepretly train and test datasets (to avoid data leakage).

    :x_train: training features
    :y_train: training labels
    :x_test: test features
    :y_test: test labels
    :parameters: list of user-defined parameters

    :Example:
    >>> x_train, y_train, x_test, y_test = scale_all_separetly(x_train, y_train, x_test, y_test, parameters)

    """
    scaler_x = preprocessing.MinMaxScaler ( feature_range =( -1, 1))
    x_train = np. array (x_train).reshape ((len(x_train) , parameters[6]))
    x_train = scaler_x.fit_transform(x_train)
    x_test = np. array (x_test).reshape ((len(x_test) , parameters[6]))
    x_test = scaler_x.fit_transform(x_test)

    scaler_y = preprocessing. MinMaxScaler ( feature_range =( -1, 1))
    y_train = np.array (y_train).reshape ((len(y_train), 1))
    y_train = scaler_y.fit_transform (y_train)
    y_test = np.array (y_test).reshape ((len(y_test), 1))
    y_test = scaler_y.fit_transform (y_test)

    return x_train, y_train, x_test, y_test, scaler_x, scaler_y


def train_test_splitting_wavelet(df, parameters):
    """
    Split the dataframe to train and test subsets, and apply wavelet to TRAIN feature set.
    No data shuffeling. Returns four subsets.

    :df: - dataframe to split
    :parameters: - list of user-defined parameters

    :Example:
    >>> x_train, y_train, x_test, y_test = train_test_splitting_wavelet(df, PARAMETERS)

    """
    train_end = int(parameters[0] * parameters[2])

    df_train = df[:train_end]
    df_test = df[train_end:]

    df_train = apply_wavelet(df_train, 1)

    x_train = (df_train.loc[:, : '<RSI>']).as_matrix()
    y_train = (df_train.loc[:, '<NEXT>':]).as_matrix()


    x_test =  (df_test.loc[:, : '<RSI>']).as_matrix()
    y_test =  (df_test.loc[:, '<NEXT>':]).as_matrix()

    return x_train, y_train, x_test, y_test
