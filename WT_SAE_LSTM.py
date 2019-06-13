
# coding: utf-8

# # Wavelet Transform + Stacked Autoencoder + Long-Short Term Memory (LSTM) Prediction

# # User parameters:

# In[26]:


# Location of the file, for which SAE should be performed
FILE_ADDRESS = "Data/TA/BAC_15-mins_9years_TA.csv"
# Period length for splitting the original dataset
SPLIT_PERIOD = 1000
# Hidden layers in LSTM (dimensionality of state vector)
HIDDEN_LSTM_UNITS = 5
# Hidden layers in Autoencoder
HIDDEN_LAYERS_AUTOENCODER = 10
# Depth of Stacked Autoencoder
DEPTH_SAE = 1
# How to split each dataset to train/test?
TEST_TRAIN_SPLIT_COEFFICENT = 0.5
# For which year to perform LSTM predicting
CURRENT_YEAR = 2012
# Amount of training epochs
EPOCHS = 100
# Batch size for LSTM (None for default)
BATCH_SIZE = None
# Batch size for autoencoder
BATCH_SIZE_AUTOENCODER = 256
# Amount of input features
INPUT_SHAPE = 15
# Please set 1 - to print all epochs, 0 - to ignore printing epochs
SHOW_PROGRESS = 0
# Zipping all parameters to one list for easier passing
PARAMETERS = [SPLIT_PERIOD, HIDDEN_LSTM_UNITS, TEST_TRAIN_SPLIT_COEFFICENT, CURRENT_YEAR, EPOCHS, BATCH_SIZE, INPUT_SHAPE, SHOW_PROGRESS]


# ## Loading dependencies:

# In[27]:


import pandas as pd
import time
from deepstock import *


# Start time measuring

# In[28]:


start = time.time()


# ## Features extracting
# * Loading initial dataset

# In[29]:


df = pd.read_csv(FILE_ADDRESS)
df = df.loc[:, '<OPEN>':]


# * Delete pre-defined correlated features

# In[30]:


deleted_correlated_features(df)


# * Normalize dataset (MinMax normalisation, range 0..1)

# In[31]:


df = normalize(df)


# * Train a Stacked Autoencoder, according to the user parameters above

# In[32]:


encoders, decoders = SAE_train(df, HIDDEN_LAYERS_AUTOENCODER, EPOCHS, BATCH_SIZE_AUTOENCODER, DEPTH_SAE, SHOW_PROGRESS)


# * Extract deep features (LAST hidden layer of SAE)

# In[33]:


features = SAE_predict(encoders, decoders, df)


# * Remove columns with all zero values

# In[34]:


features = pd.DataFrame(features)
features = features.loc[:, (features != 0).any(axis=0)]


# * Update input shape of LSTM network

# In[35]:


INPUT_SHAPE += features.shape[1]
print('New input shape: ', INPUT_SHAPE)


# * Update parameters list

# In[36]:


PARAMETERS = [SPLIT_PERIOD, HIDDEN_LSTM_UNITS, TEST_TRAIN_SPLIT_COEFFICENT, CURRENT_YEAR, EPOCHS, BATCH_SIZE, INPUT_SHAPE, SHOW_PROGRESS]


# ## Merge initial features and extracted features to a single dataset

# In[37]:


df_initial = pd.read_csv(FILE_ADDRESS)


# In[38]:


df = pd.concat([features, df_initial], axis=1)


# * Remove correlated features, create a YEAR and NEXT additional columns

# In[39]:


data_preparing(df)
df.head(1)


# In[40]:


print('Overeall shape of all data: ',df.shape)


# # Taking one year

# In[41]:


df = df.loc[df['<YEAR>'] == CURRENT_YEAR]
# df = df[:1000]
del df['<YEAR>']


# In[42]:


print('Dimensinality of data for ', CURRENT_YEAR, ' year is: ', df.shape)


# ## Perform LSTM predicting with a sliding window approach

# In[43]:


periods = len(df)
profits = []
mapes = []
sliding_interval = SPLIT_PERIOD - int(SPLIT_PERIOD*TEST_TRAIN_SPLIT_COEFFICENT)
# model = create_LSTM();
for i in range(0, periods, sliding_interval):
    profit = 0.0
    mape = 0.0
    start_period = i
    if i + SPLIT_PERIOD > periods:
        break
    else:
        end_period = i + SPLIT_PERIOD

#     MAIN ACTION

    print('Current period: ', start_period, ' to ', end_period)
    current_df = df[start_period:end_period]
#     Deviding
    x_train, y_train, x_test, y_test = train_test_splitting_wavelet(current_df, PARAMETERS)
#     Scaling
    x_train, y_train, x_test, y_test, scaler_x, scaler_y = scale_all_separetly(x_train, y_train, x_test, y_test, PARAMETERS)
#     Performing LSTM
    profit, mape = perform_LSTM(x_train, y_train, x_test, y_test, scaler_x, scaler_y, PARAMETERS)

    profits.append(profit)
    mapes.append(mape)
#     END ACTION


# In[44]:


end = time.time()
print("Time spent: ", (end-start), " seconds.")


# ## Profitability results:

# * Profits distribution for all periods within a year:

# In[45]:


profits


# In[46]:


print("Overall yearly profitability for ", CURRENT_YEAR, " year: ")
print(sum(profits))


# In[47]:


print("Minimum profitability per one period of ", CURRENT_YEAR, " year: ")
print(min(profits))


# In[48]:


print("Maximum profitability per one period of ", CURRENT_YEAR, " year: ")
print(max(profits))


# In[49]:


print("Mean profitability per one period of ", CURRENT_YEAR, " year: ")
print(np.mean(profits))
