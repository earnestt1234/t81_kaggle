#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 10:23:58 2022

@author: tom.earnest
"""

#%% Imports

import datetime
import os

import numpy as np
import pandas as pd

#%% PARAMETERS


SAVENAME = 'model2_predicts.csv'

#%% Overture
date = str(datetime.datetime.now())

print()
print("KAGGLE MODEL")
print("------------")
print('T81 Applications of Deep Learning')
print(f'Date: {date}')
print("Authors: Tom Earnest & Robert Jirsarie")

#%% Read data 

print()
print('---')
print("I. Reading Data")
df = pd.read_csv('data_features_added.csv')
print('---')

#%% Define helper functions

def timeseries_to_trainable(df, features, target, target_id, window=10, forecast=90):
    
    n = len(df)
    x = n - (window + forecast) + 1
    t = window

    arr = np.full((x, t, len(features)), np.nan)
    for z, feature in enumerate(features):
        for i in range(x):
            arr[i, :, z] = df[feature].iloc[i:i+window]


    forecasted = df[target].iloc[window + forecast -1:].to_numpy()
    identifier = df[target_id].iloc[window + forecast-1:].to_numpy()
    
    return arr, forecasted, identifier

def stack_training(arrays, forecasted):
    
    n_arrays = len(arrays)
    example = arrays[0]
    x, y, z = example.shape
    
    final_x_shape = (n_arrays * x, y, z)
    final_x = np.full(final_x_shape, np.nan)
    
    final_y_shape = (n_arrays * x, 1)
    final_y = np.full(final_y_shape, np.nan)
    print('---')
    for i, (arr, f) in enumerate(zip(arrays, forecasted)):
        start = x * i
        end = start + x
        final_x[start:end, :, :] = arr
        final_y[start:end, 0] = f
        
    return final_x, final_y

#%% Create training and testing arrays


features = df.columns.drop(['date', 'item_id', 'SUBMIT_ID']).to_list()

print()
print('---')
print("II. Creating tensors")

arrays_train = []
forecasts_train = []

arrays_test = []
forecasts_test = []

identifiers = []

print()
print("Seperating data by item...")

for item in df['item_id'].unique():
    print(f"  Item = {item}...")
    data = df[df['item_id'] == item]
    arr, forecast, identifier = timeseries_to_trainable(data, features=features,
                                                        target='item_count',
                                                        target_id='SUBMIT_ID',
                                                        window=90, forecast=92)
    arr_train = arr[np.isnan(identifier), :, :]
    forecast_train = forecast[np.isnan(identifier)]
    arr_test = arr[~ np.isnan(identifier), :, :]
    forecast_test = forecast[~ np.isnan(identifier)]
    
    arrays_train.append(arr_train)
    forecasts_train.append(forecast_train)
    arrays_test.append(arr_test)
    forecasts_test.append(forecast_test)
    
    identifiers.append(identifier[~np.isnan(identifier)])


print()
print("Regrouping separated data")    
X, Y = stack_training(arrays_train, forecasts_train)
X_PREDICT, _ = stack_training(arrays_test, forecasts_test)
PREDICT_IDS = np.concatenate(identifiers)
print("Done.  Reporting data shapes...")
print(f"All training features: {X.shape}")
print(f"All training labels: {Y.shape}")
print(f"All testing features: {X_PREDICT.shape}")
print('---')

#%% training/validation split

from sklearn.model_selection import train_test_split

print()
print('---')
print("III. Creating training/validation split.")
X_train, X_valid, Y_train, Y_valid = train_test_split(X, Y, test_size=0.2, random_state=0)

print()
print("Done.  Reporting data shapes...")
print(f"Training features: {X_train.shape}")
print(f"Training labels: {Y_train.shape}")
print(f"Validation features: {Y_train.shape}")
print(f"Validation labels: {Y_valid.shape}")
print('---')

#%% Build model

print()
print('---')
print("IV. Defining model.")

import tensorflow as tf 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D
from tensorflow.keras.layers import Dense, Flatten, Dropout

epochs = 500 
batch = 256
lr = 0.0003
adam = tf.keras.optimizers.Adam(lr)

model_cnn = Sequential()
model_cnn.add(Conv1D(filters=64, kernel_size=8, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])))
model_cnn.add(MaxPooling1D(pool_size=2))
model_cnn.add(Flatten())
model_cnn.add(Dense(50, activation='relu'))
model_cnn.add(Dropout(0.2))
model_cnn.add(Dense(20, activation='relu'))
model_cnn.add(Dropout(0.1))
model_cnn.add(Dense(1))
model_cnn.compile(loss='mse', optimizer=adam)
model_cnn.summary()

print()
print(f"Epochs: {epochs}")
print(f"Batch: {batch}")
print(f"Learning rate: {lr}")
print("Optimizer: Adam")

print('---')

#%% Train model

print()
print('---')
print("V. Training.")

from tensorflow.keras.callbacks import EarlyStopping

monitor = EarlyStopping(monitor='val_loss', min_delta=0, patience=25, verbose=2, mode='auto', restore_best_weights=True)

cnn_history = model_cnn.fit(X_train, Y_train, callbacks=[monitor], validation_data=(X_valid, Y_valid), epochs=epochs, verbose=2)

print('---')


#%% Get testing predictions

print()
print('---')
print("V. Saving test predictions.")
predicts = model_cnn.predict(X_PREDICT)

submit_df = pd.DataFrame({'id':PREDICT_IDS.astype(int),
                          'item_count': predicts.flatten()})
submit_df.to_csv(SAVENAME, index=False)
print('---')