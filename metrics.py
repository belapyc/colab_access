import numpy as np

def rmse(input_layer, reconstruction_layer):
    """
    Function returning a root mean square error of two input vectors.

    :input_layer: first vector
    :reconstruction_layer: second vector

    :Example:
    >>> rmse =  rmse(input_layer, reconstruction_layer)

    Author: Nikita Vasilenko
    """
    return sqrt(mean_squared_error(input_layer, reconstruction_layer))

def find_mean_rmse(test_x,reconstruction_layers):
    """
    Function returning a root mean square error of two input matrices


    :test_x: first matrix
    :reconstruction_layers: second matrix

    :Example:
    >>> mean_rmse = find_mean_rmse(test_x,reconstruction_layers)

    Author: Nikita Vasilenko
    """
    sum_rmse = 0
    for i in range(0, len(test_x)):
        input_array = test_x[i]
        reconstructed_array = reconstruction_layers[i]
        current_rmse = rmse(input_array, reconstructed_array)
        sum_rmse += current_rmse
    return(sum_rmse/len(test_x))

def mean_absolute_percentage_error(y_true, y_pred):
    """
    Return MAPE metrics - Mean absolute percentage error between two vectors

    :y_true: true values vector
    :y_pred: predicted values vector

    :Example:
    >>> mape = mean_absolute_percentage_error(y_true, y_pred)

    Author: Nikita Vasilenko
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

    Author: Nikita Vasilenko
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
