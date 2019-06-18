import pandas as pd
from for_finance import deepstock

def read_file(path):
    return pd.read_csv(path)

def deleted_correlated_features(df):
    """
    Deleting features, which are pre-defined as correlated.

    :df: Pandas dataframe, where to remove correlated features

    :Example:
	# Read dataframe from file:
    >>> df = pd.read_csv(FILE_ADDRESS)
    # Remove pre-defined correlated features
    >>> df = deleted_correlated_features(df)

    Author: Nikita Vasilenko
    """
    del df['<UPPERBAND>']
    del df['<MIDDLEBAND>']
    del df['<LOWERBAND>']
    del df['<EMA20>']
    del df['<MA5>']
    del df['<MA10>']

def data_preparing(df):
    """
    Given a data for several dates removing unneeded features, and prepare for yearly splitting.
    Adding additional feature: NEXT, which is essentially close price of the next time slot.
    Adding additional feature: YEAR, which is just a year.

    :df: dataframe to process.


    :Example:
    >>>data_preparing(df)

    Author: Nikita Vasilenko
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

    Author: Nikita Vasilenko
    """
    x = (df.loc[:, : '<RSI>']).as_matrix()
    y = (df.loc[:, '<NEXT>':]).as_matrix()

    train_end = int(parameters[0] * parameters[2])
    x_train = x [0: train_end,]
    x_test = x[ train_end +1:len(x),]
    y_train = y [0: train_end]
    y_test = y[ train_end +1:len(y)]
    return x_train, y_train, x_test, y_test


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

    Author: Nikita Vasilenko
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

    Author: Nikita Vasilenko
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

    Author: Nikita Vasilenko
    """
    result = df.copy()
    for feature_name in df.columns:
        max_value = df[feature_name].max()
        min_value = df[feature_name].min()
        result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
    return result

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

    Author: Nikita Vasilenko
    """
    waved_df = pd.DataFrame()
    for items in df.items():
        current_name = items[0]
        current_series = items[1]
#         N- layer Wavelet Transform with Haar Function
        cA = pywt.wavedec(current_series, 'haar', level=levels)
        waved_df[current_name] = cA[0]
    return waved_df
