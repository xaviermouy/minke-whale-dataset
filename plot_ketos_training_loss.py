# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 10:07:31 2022

@author: xavier.mouy
"""

import pandas as pd
import matplotlib.pyplot as plt

infile = r'C:\Users\xavier.mouy\Documents\GitHub\Ketos\minke\20220401T140308\ketos_models\20s\log.csv'

df = pd.read_csv(infile)
pd.pivot_table(df.reset_index(), index='epoch', columns='dataset', values='loss').plot(subplots=False, grid=True, ylabel='Loss', xlabel='Epoch')
pd.pivot_table(df.reset_index(), index='epoch', columns='dataset', values='Precision').plot(subplots=False, grid=True, ylabel='Precision', xlabel='Epoch')
pd.pivot_table(df.reset_index(), index='epoch', columns='dataset', values='Recall').plot(subplots=False, grid=True, ylabel='Recall', xlabel='Epoch')
