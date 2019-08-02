import pandas as pd
from sklearn.metrics import mean_squared_error
from scipy import stats
from sklearn import preprocessing
from math import sqrt
import numpy as np
from pywt import wavedec

class DataPrep:

    def __init__(self, logger):
        self.logger = logger

    def read_file(self, path):
        self.data = pd.read_csv(path)
        self.initial_data = pd.read_csv(path)
        #return pd.read_csv(path)

    def initial_prep(self):
        self.deleted_correlated_features()
        self.normalize()

    def deleted_correlated_features(self):
        """
        Deleting features, which are pre-defined as correlated.

        :self.data: Pandas dataframe, where to remove correlated features

        :Example:
    	# Read dataframe from file:
        >>> self.data = pd.read_csv(FILE_ADDRESS)
        # Remove pre-defined correlated features
        >>> self.data = deleted_correlated_features(self.data)

        Author: Nikita Vasilenko
        """
        del self.data['<UPPERBAND>']
        del self.data['<MIDDLEBAND>']
        del self.data['<LOWERBAND>']
        del self.data['<EMA20>']
        del self.data['<MA5>']
        del self.data['<MA10>']

    def data_preparing(self, features):
        """
        Given a data for several dates removing unneeded features, and prepare for yearly splitting.
        Adding additional feature: NEXT, which is essentially close price of the next time slot.
        Adding additional feature: YEAR, which is just a year.

        :self.data: dataframe to process.


        :Example:
        >>>data_preparing(self.data)

        Author: Nikita Vasilenko
        """
        self.data = pd.concat([features, self.initial_data], axis=1)
        years = []
        for index, row in self.data.iterrows():
            years.append(int(str(row['<DATE>'])[:4]))
        self.data['<YEAR>'] = years

        # Deleting collerated and unneeded features
        del self.data['<DATE>']
        del self.data['<TIME>']
        del self.data['<UPPERBAND>']
        del self.data['<MIDDLEBAND>']
        del self.data['<LOWERBAND>']
        del self.data['<EMA20>']
        del self.data['<MA5>']
        del self.data['<MA10>']
    #     Creating one step ahead price
        self.data['<NEXT>'] = (self.data['<CLOSE>']).shift(-1)
    #     Removing last row!
        self.data = self.data.dropna()

    def choose_year(self, year):
        df = self.data.loc[self.data['<YEAR>'] == year]
        del df['<YEAR>']
        return df

    @staticmethod
    def train_test_splitting( df, parameters):
        """
        Split dataframe to four subsets.

        :self.data: dataframe to split.
        :parameters: list of user - defined parameters

        :Example:
        >>> x_train, y_train, x_test, y_test = train_test_splitting(self.data, PARAMETERS)

        Author: Nikita Vasilenko
        """
        x = (df.loc[:, : '<RSI>']).values
        y = (df.loc[:, '<NEXT>':]).values

        train_end = int(parameters['SPLIT_PERIOD'] * parameters['TEST_TRAIN_SPLIT_COEFFICENT'])
        x_train = x [0: train_end,]
        print("TRAINING END AT ", len(x_train))
        x_test = x[ train_end +1:len(x),]
        y_train = y [0: train_end]
        y_test = y[ train_end +1:len(y)]
        return x_train, y_train, x_test, y_test

    @staticmethod
    def scale_all_separetly( x_train, y_train, x_test, y_test, parameters):
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

        x_train = np.array(x_train).reshape((len(x_train) , parameters['INPUT_SHAPE']))

        x_train = scaler_x.fit_transform(x_train)
        x_test = np. array (x_test).reshape ((len(x_test) , parameters['INPUT_SHAPE']))
        x_test = scaler_x.fit_transform(x_test)

        scaler_y = preprocessing. MinMaxScaler ( feature_range =( -1, 1))
        y_train = np.array (y_train).reshape ((len(y_train), 1))
        y_train = scaler_y.fit_transform (y_train)
        y_test = np.array (y_test).reshape ((len(y_test), 1))
        y_test = scaler_y.fit_transform (y_test)

        return x_train, y_train, x_test, y_test, scaler_x, scaler_y

    @staticmethod
    def train_test_splitting_wavelet(df, parameters):
        """
        Split the dataframe to train and test subsets, and apply wavelet to TRAIN feature set.
        No data shuffeling. Returns four subsets.

        :self.data: - dataframe to split
        :parameters: - list of user-defined parameters

        :Example:
        >>> x_train, y_train, x_test, y_test = train_test_splitting_wavelet(self.data, PARAMETERS)

        Author: Nikita Vasilenko
        """
        x = (df.loc[:, : '<RSI>']).values
        y = (df.loc[:, '<NEXT>':]).values

        train_end = int(parameters['SPLIT_PERIOD'] * parameters['TEST_TRAIN_SPLIT_COEFFICENT'])


        data_train = df[0:train_end]
        data_test = df[train_end:]

        data_train = DataPrep.apply_wavelet(data_train, 1)

        x_train = (data_train.loc[:, : '<RSI>']).values
        y_train = (data_train.loc[:, '<NEXT>':]).values

        x_test =  (data_test.loc[:, : '<RSI>']).values
        y_test =  (data_test.loc[:, '<NEXT>':]).values

        return x_train, y_train, x_test, y_test

    def normalize(self):
        """
        This functions performa a normalization for all rows and all columns of a given Pandas Dataframe

        Type of normalization: MinMax scaling, changing ranges of all values to a range [0..1]
        :self.data: Pandas dataframe, for which to perform normalisation

    	:Example:
    	# Read dataframe from file:
        >>> self.data = pd.read_csv(FILE_ADDRESS)
        # Apply MinMax normalization
        >>> self.data = normalize(self.data, 1)

        Author: Nikita Vasilenko
        """
        result = self.data.copy()
        for feature_name in self.data.columns:
            max_value = self.data[feature_name].max()
            min_value = self.data[feature_name].min()
            result[feature_name] = (self.data[feature_name] - min_value) / (max_value - min_value)
        self.data = result

    @staticmethod
    def apply_wavelet(df, levels):
        """
        This functions performa a 1-layer wavelet transform for all rows and all columns of a given Pandas Dataframe

        :self.data: Pandas dataframe, for which to perform wavelet transform
        :levels: The amount of consiquent wavelet transforms to perform

    	:Example:
    	# Read dataframe from file:
        >>> self.data = pd.read_csv(FILE_ADDRESS)
        # Apply 1-level wavelet transform and save to a new dataframe
        >>> waved_self.data = apply_wavelet(self.data, 1)

        Author: Nikita Vasilenko
        """
        waved_data = pd.DataFrame()
        for items in df.items():
            current_name = items[0]
            current_series = items[1]
    #         N- layer Wavelet Transform with Haar Function
            cA = wavedec(current_series, 'haar', level=levels)
            waved_data[current_name] = cA[0]
        return waved_data
