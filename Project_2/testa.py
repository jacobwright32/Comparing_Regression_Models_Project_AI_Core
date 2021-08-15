#%%
import seaborn as sns
from sklearn import preprocessing
from sklearn import model_selection
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn
import os
from os import walk
from os import listdir
from os.path import isfile, join


mypath = 'United_Kingdom_Weather_Data/' 
all_files = [f for f in listdir(mypath) if isfile(join(mypath, f))]
df = pd.DataFrame()

for city in all_files:
    df2 = pd.read_csv(mypath + city, index_col=1)
    df = df.append(df2)
# %%
df.info()
# %%

for season in range (0,13):
    if 3 <= season <= 5:
        season_value = season
        season = 'spring'
        print(season)
        print(season_value)

    elif 6 <= season <= 8:
        season_value = season
        season = 'summer'
        print(season)
        print(season_value)

    elif 9 <= season <= 11:
        season_value = season
        season = 'autumn'
        print(season)
        print(season_value)

    elif season == 12 or season == 1 or season == 2:
        season_value = season
        season = ('winter')
        print(season)
        print(season_value)

    else:
        pass
# %%
