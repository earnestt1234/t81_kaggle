
import os

import numpy as np
import pandas as pd

DATA_FOLDER = 'beach_demand_forecast'

#%% Read data

df_train = pd.read_csv(os.path.join(DATA_FOLDER, 'sales_train.csv'))
df_train['date'] = pd.to_datetime(df_train['date'])

df_test = pd.read_csv(os.path.join(DATA_FOLDER, 'sales_test.csv'))
df_test['date'] = pd.to_datetime(df_test['date'])
df_test.columns = ['SUBMIT_ID', 'date', 'item_id', 'price']

df = pd.concat([df_train, df_test])
df = df.sort_values(['date', 'item_id']).reset_index(drop=True)

#%% Add features from items.csv

items = pd.read_csv(os.path.join(DATA_FOLDER, 'items.csv'))

merger = items[['id', 'store_id', 'kcal']]
merger = merger.rename({'id':'item_id'}, axis=1)
merger = pd.get_dummies(merger, columns=['store_id'])
df = df.merge(merger, on='item_id')

#%% days relative to June 2020
# seems to be a lot of trends that drastically change around this month

xdate = pd.Timestamp('06-20-2020')
diff = df['date'] - xdate
df['diff_june_2020'] = diff.dt.days

#%% holidays

import holidays
us_holidays = holidays.US()

df['holiday'] = [1 if d in us_holidays else 0 for d in df['date']]

#%% day of week, season

df['weekday'] = df['date'].dt.dayofweek
df['quarter'] = df['date'].dt.quarter
df = pd.get_dummies(df, columns=['weekday', 'quarter'])

#%% Save

NAME = 'data_features_added.csv'
df.to_csv(NAME, index=False)


