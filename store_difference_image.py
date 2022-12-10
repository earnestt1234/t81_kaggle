#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  4 17:12:18 2022

@author: earnestt1234
"""

import pandas as pd

df = pd.read_csv('data_features_added.csv')
df['date'] = pd.to_datetime(df['date'])

stores = df[[col for col in df.columns if 'store' in col]]
df['store'] = stores.idxmax(axis=1).str.replace('store_id_', '').astype(int)

total_sales = df.groupby(['store', 'item_id'], as_index=False)['item_count'].sum()
idx = total_sales.groupby('store')['item_count'].idxmax()
best_items = total_sales.loc[idx.values]

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

fig, ax = plt.subplots()

for i, row in best_items.iterrows():
    store = row['store']
    item = row['item_id']

    sub = df[(df['store'] == store) & (df['item_id'] == item)]
    x = sub['date']
    y = sub['item_count']

    ax.plot(x, y, label=f'Store {int(store)}')


ax.set_ylabel('Daily sales')
ax.set_xlabel('Date')
ax.legend()
ax.xaxis.set_major_locator(mdates.MonthLocator([1, 7]))
ax.set_title('Best selling items by store')

plt.tight_layout()
fig.savefig('store_difference.png', dpi=300)

