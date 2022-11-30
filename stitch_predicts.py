# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 14:03:06 2022

@author: earne
"""

import matplotlib.pyplot as plt
import pandas as pd

data_path = 'data_features_added.csv'
predicts_path = 'Mean_Tom_Rob.csv'
# predicts_path = 'model4_predicts.csv'

df = pd.read_csv(data_path)
df['date'] = pd.to_datetime(df['date'])

predicts = pd.read_csv(predicts_path)
sales_test = pd.read_csv('beach_demand_forecast/sales_test.csv')
items = pd.read_csv('beach_demand_forecast/items.csv')
items = items.rename({'id':'item_id'}, axis=1)
store_link = items[['item_id', 'store_id']]

predicts = sales_test.merge(predicts, on='id')
predicts = predicts.merge(store_link, on='item_id')
predicts['date'] = pd.to_datetime(predicts['date'])

store = 1
item = 76

fig, ax = plt.subplots()
ax.set_xticks([])

store_col = f'store_id_{store}'
train = df[(df['item_id'] == item) & (df[store_col] == 1)]
ax.scatter(train['date'], train['item_count'], s=1)

test = predicts[(predicts['item_id'] == item) & (predicts['store_id'] == store)]
ax.scatter(test['date'], test['item_count'], s=1)