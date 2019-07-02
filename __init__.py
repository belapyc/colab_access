FILE_ADDRESS = "Data/TA/BAC_15-mins_9years_TA.csv"
# Period length for splitting the original dataset
SPLIT_PERIOD_DEFAULT = 1000
# Hidden layers in LSTM (dimensionality of state vector)
HIDDEN_LSTM_UNITS_DEFAULT = 5
# Hidden layers in Autoencoder
HIDDEN_LAYERS_AUTOENCODER_DEFAULT = 10
# Depth of Stacked Autoencoder
DEPTH_SAE_DEFAULT = 1
# How to split each dataset to train/test?
TEST_TRAIN_SPLIT_COEFFICENT_DEFAULT = 0.5
# For which year to perform LSTM predicting
CURRENT_YEAR_DEFAULT = 2013
# Amount of training epochs
EPOCHS_DEFAULT = 100
# Batch size for LSTM (None for default)
BATCH_SIZE_DEFAULT  = None
# Batch size for autoencoder
BATCH_SIZE_AUTOENCODER_DEFAULT  = 256
# Amount of input features
INPUT_SHAPE_DEFAULT  = 15
# Please set 1 - to print all epochs, 0 - to ignore printing epochs
SHOW_PROGRESS_DEFAULT  = 0
# Zipping all parameters to one list for easier passing
ALL_YEARS = [2009,2010,2011,2012,2013,2014,2015,2016,2017]
PARAMETERS = {
    "SPLIT_PERIOD":SPLIT_PERIOD_DEFAULT ,
    "HIDDEN_LSTM_UNITS":HIDDEN_LSTM_UNITS_DEFAULT ,
    "TEST_TRAIN_SPLIT_COEFFICENT": TEST_TRAIN_SPLIT_COEFFICENT_DEFAULT ,
    "CURRENT_YEAR":CURRENT_YEAR_DEFAULT ,
    "EPOCHS":EPOCHS_DEFAULT ,
    "BATCH_SIZE":BATCH_SIZE_DEFAULT ,
    "INPUT_SHAPE":INPUT_SHAPE_DEFAULT ,
    "SHOW_PROGRESS":SHOW_PROGRESS_DEFAULT,
    "HIDDEN_LAYERS_AUTOENCODER":HIDDEN_LAYERS_AUTOENCODER_DEFAULT,
    "DEPTH_SAE":DEPTH_SAE_DEFAULT,
    "BATCH_SIZE_AUTOENCODER" : BATCH_SIZE_AUTOENCODER_DEFAULT,
    "ALL_YEARS" : ALL_YEARS
}
