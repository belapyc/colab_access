import keras
from keras.layers import Input, Dense, Dropout
from keras.layers.core import Dense
from keras.layers.recurrent import LSTM
from keras.models import Model, Sequential
from sklearn.metrics import mean_squared_error

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
