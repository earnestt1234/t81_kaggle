# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 08:42:47 2022

@author: earne
"""
import matplotlib.pyplot as plt
import pandas as pd

a = 'model4_predicts.csv'
b = 'model7_predicts.csv'


A = pd.read_csv(a)
B = pd.read_csv(b)

submission_key = pd.read_csv('beach_demand_forecast/sales_test.csv')
items = pd.read_csv('beach_demand_forecast/items.csv')
items = items.rename({'id':'item_id'}, axis=1)
submission_key = submission_key.merge(items, on='item_id')
submission_key['date'] = pd.to_datetime(submission_key['date'])

A = submission_key.merge(A, on='id')
B = submission_key.merge(B, on='id')

store = 1
item = 73

fig, ax = plt.subplots()

subA = A[A['item_id'].eq(item) & A['store_id'].eq(store)]
ax.plot(subA['date'], subA['item_count'], label=a)

subB = B[B['item_id'].eq(item) & B['store_id'].eq(store)]
ax.plot(subB['date'], subB['item_count'], label=b)

ax.legend()

