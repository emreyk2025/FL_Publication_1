from keras import layers, Sequential, optimizers
from lightgbm import LGBMRegressor
import xgboost as xgb

# For some reason the keras can not be imported directly from tensorflow as tensorflow.keras ... etc. In order to circumvent this
# issue I had to import keras and tensorflow seperately. from keras.layers import LSTM also doesn't work you have to call LSTM as
# layers.LSTM.

def get_LSTM_Simple(time_steps, num_features, learning_rate):
    model = Sequential()
    model.add(layers.Input(shape=(time_steps, num_features)))
    model.add(layers.LSTM(60, return_sequences=True))  
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(1))

    optimizer=optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mean_absolute_error'])
    return model

def get_LSTM_Bidirectional(time_steps, num_features, learning_rate):
    model = Sequential()
    model.add(layers.Input(shape=(time_steps, num_features)))
    model.add(layers.Bidirectional(layers.LSTM(60, return_sequences=True)))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(1))

    optimizer = optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mean_absolute_error'])
    return model


def get_LSTM_stacked(time_steps, num_features, learning_rate):
    model = Sequential()
    model.add(layers.Input(shape=(time_steps, num_features)))
    model.add(layers.LSTM(20, return_sequences=True)) 
    model.add(layers.Dropout(0.2))
    model.add(layers.LSTM(18, return_sequences=True))
    model.add(layers.Dropout(0.2))
    model.add(layers.LSTM(14, return_sequences=True))
    model.add(layers.Dropout(0.2))
    model.add(layers.LSTM(7))  
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(1))

    optimizer = optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mean_absolute_error'])
    return model

def get_GRU(time_steps, num_features, learning_rate):
    model = Sequential()
    model.add(layers.Input(shape=(time_steps, num_features)))
    model.add(layers.GRU(20, return_sequences=True))  # Corrected: GRU instead of LSTM
    model.add(layers.Dropout(0.2))
    model.add(layers.GRU(18, return_sequences=True))
    model.add(layers.Dropout(0.2))
    model.add(layers.GRU(14, return_sequences=True))
    model.add(layers.Dropout(0.2))
    model.add(layers.GRU(7))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(1))

    optimizer = optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mean_absolute_error'])
    
    return model


# The parameters for this model are given when this function is called for training in run.py
def get_XGBoost():    
    return xgb

def get_LightGBM(params):

    return LGBMRegressor(**params)


'''

Old comments

# Avoid subsample parameter, leave it as 1 (default), otherwise it will select a subsample of
# training data randomly for each tree which is not okay for a time series task. Similarly colsample
# which picks a portion of features to reduce overfitting.

# Define the XGBoost model

# Why XGBoost (and LightGBM) for time series forecasting: 
# https://medium.com/@geokam/time-series-forecasting-with-xgboost-and-lightgbm-predicting-energy-consumption-460b675a9cee

# There is also one other model with the name XGB Random Forest Regression: XGBRFRegressor

# Model Combination: LSTM + XGBoost

# Model Combination: LSTM + LightGBM

'''