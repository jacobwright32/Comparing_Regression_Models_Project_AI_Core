#%%
from math import remainder
from numpy.core.numeric import True_
import seaborn as sns
from sklearn import preprocessing
from sklearn import model_selection
from sklearn import linear_model
from sklearn import datasets
from sklearn import neighbors
from sklearn import svm
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_transformer
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn
import os
from os import walk
from os import listdir
from os.path import isfile, join
import timeit
import itertools
import time

def data_loader(mypath, city, df):
    ''' Loads in the data'''
    df2 = pd.read_csv(mypath + city, index_col=1)
    df = df.append(df2)
    return df

def seasons(season):
    '''Maps the dates into seasons'''
    if 2 <= season <= 4:
        season_value = season
        season = 'spring'
        return season, season_value

    elif 5 <= season <= 7:
        season_value = season
        season = 'summer'
        return season, season_value

    elif 8 <= season <= 10:
        season_value = season
        season = 'autumn'
        return season, season_value

    elif season == 11:
        season_value = season
        season = ('winter')
        return season, season_value
    elif season == 12:
        season_value = season
        season = ('winter')
        return season, season_value
    elif season == 1:
        season_value = season
        season = ('winter')
        return season, season_value

    else:
        pass

def split_data(df):
    '''Splits and preprocesses the data ready for modelling'''
    random_seed = 42
    
    df.reset_index(inplace=True, drop=True)
    y = df['amount_of_rain']
    y.replace(to_replace={np.nan : 0}, inplace=True)
    df.drop(columns=['wind_direction','city','amount_of_rain'], inplace=True)
    for s in range(1,13):
        season, season_value = seasons(s)
        df.date = df.date.apply(lambda x: season if str(season_value) + '-' in x else x)
    column_trans = make_column_transformer((OneHotEncoder(),['date'] ),remainder='passthrough')
    df = column_trans.fit_transform(df)
    X = pd.DataFrame(df)
    X_train, X_val, y_train, y_val = train_test_split(X, y, random_state=random_seed)
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, random_state=random_seed)
    return X_train, X_val, X_test, y_train, y_val, y_test

def normaliser(X_train, X_val, X_test):
    '''Normalises the data'''    
    X_mean = np.mean(X_train, axis=0)
    X_std = np.std(X_train, axis=0)
    X_train = (X_train - X_mean) / X_std
    X_val = (X_val - X_mean) / X_std
    X_test = (X_test - X_mean) / X_std
    return X_train, X_val, X_test

class Base_model:
    '''Each model inherts the base class structure'''
    def __init__(self,params):
        '''intialises the parameters for each of the selected model passed into the Base_model'''
        self.keys = list(params.keys())
        self.values = params.values()
        self.grid_params = list(itertools.product(*self.values))
        

    def fit(self, model):
        '''Fits the given data to the selected model and outputs evaluation metrics'''
        results = []
        model_type = str(model).split('.')
        model_type = model_type[len(model_type)-1]
        model_type = model_type[:len(model_type) - 2]
        print(model_type)
        for r in self.grid_params:
                print(r)
                param = {self.keys[idx]:r[idx] for idx in range(len(r))}
                self.reg = model(**param)
                start = time.process_time()
                self.reg.fit(X_train, y_train)
                end = time.process_time()
                self.evaluation()
                # Train the model with grid search
                result = {
                    'model_type' : str(model_type),
                    'time_taken' : (end-start),
                    'params': param,
                    'train_score': self.reg.score(X_train, y_train),
                    'val_score': self.reg.score(X_val, y_val),
                    'test_score': self.reg.score(X_test, y_test),
                    'train_MSE' : self.MSE_train,
                    'val_MSE' : self.MSE_val,
                    'test_MSE' : self.MSE_test,
                    'train_MAE' : self.MAE_train,
                    'val_MAE' : self.MAE_val,
                    'test_MAE' : self.MAE_test,
                }

                results.append(result)

        return results

    def evaluation(self):
        '''Calculate the metrics for judging performance'''
        self.y_train_pred = self.reg.predict(X_train)
        self.y_val_pred = self.reg.predict(X_val)
        self.y_test_pred = self.reg.predict(X_test)
        self.MSE_train = mean_squared_error(y_train, self.y_train_pred)
        self.MSE_val = mean_squared_error(y_val, self.y_val_pred)
        self.MSE_test = mean_squared_error(y_test, self.y_test_pred)
        self.MAE_train = mean_absolute_error(y_train, self.y_train_pred)
        self.MAE_val = mean_absolute_error(y_val, self.y_val_pred)
        self.MAE_test = mean_absolute_error(y_test, self.y_test_pred)

def linear_parameters():
    '''Initalises the parameters'''
    params = {
    "fit_intercept": [True, False],
    'positive' : [True, False],
    "normalize": [True, False],
    }
    model = linear_model.LinearRegression
    return model, params

def ridge_parameters():
    '''Initalises the parameters'''
    params = {
    'solver' : ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga'],
    'alpha' : [0.15, 0.16, 0.17, 0.18, 0.19, 0.20, 0.21, 0.22, 0.23, 0.24, 0.25],
    "fit_intercept": [True, False],
    "normalize": [True, False],
    }
    model = linear_model.Ridge
    return model, params

def lasso_parameters():
    '''Initalises the parameters'''
    params = {
    'alpha' : [0.0008, 0.00085, 0.0009, 0.00095, 0.001],
    'selection' : ['cyclic', 'random'],
    'positive' : [True, False],
    "fit_intercept": [True, False],
    "normalize": [True, False],
    }
    model = linear_model.Lasso
    return model, params

def polynomial_parameters():
    params = {
    'degree': [1, 2, 3, 4, 5, 6]
    }
    model = PolynomialFeatures
    return model, params

def knn_parameters():
    '''Initalises the parameters'''
    params = {
    'leaf_size': [1, 2, 3 , 100],
    'n_neighbors': [50, 80, 90, 100 ],
    'weights' : ['uniform', 'distance'],
    'algorithm' : ['auto']
    }
    model = neighbors.KNeighborsRegressor
    return model, params

def svr_parameters():
    '''Initalises the parameters'''
    params = {
        'kernel' : ['linear', 'rbf'],
        'gamma': [0.1, 1],
        'C': [0.1]
    }
    model = SVR
    return model, params

def decision_tree_parameters():
    '''Initalises the parameters'''
    params = {
        'criterion' : ['mse', 'friedman_mse'],#, 'mae'],
        'splitter': ['best', 'random'],
        'max_depth': [1, 2, None],
        'min_samples_split': [2, 0.2],#, 0.4],
        'min_samples_leaf': [1, 0.1],#, 0.2],
        'min_weight_fraction_leaf': [0.0, 0.1,],# 0.3],
        'max_features': ['auto', 'sqrt', 'log2'],#, None],
        'random_state': [1, 2],#, 3, None],
        'max_leaf_nodes': [2, 3, 4],#, None],
        'min_impurity_decrease': [0.0, 0.1],#, 0.4],
        'ccp_alpha': [0.0, 0.1]
        }
    model = DecisionTreeRegressor
    return model, params

def random_forest_parameters():
    '''Initalises the parameters'''
    params = {
        'criterion' : ['mse'],#, 'mae'],
        'max_depth': [1, 2, 3, None],
        'min_samples_split': [2, 0.2],#, 0.4],
        'min_samples_leaf': [1, 0.1],#, 0.2],
        'min_weight_fraction_leaf': [0.0, 0.1,],# 0.3],
        'max_features': ['auto'],#, 'sqrt', 'log2'],
        'max_leaf_nodes': [2, 3],#4, None],
        'min_impurity_decrease': [0.0, 0.1],#, 0.4],
        'ccp_alpha': [0.0, 0.1],
        'bootstrap' : [True, False]
                }
    model = RandomForestRegressor
    return model, params


def model(model, params):
    reg = Base_model(params)
    scores = reg.fit(model)
    df_scores = pd.DataFrame(scores)
    df_scores.sort_values(by=['val_score'], ascending=False, inplace=True)
    df_scores = df_scores.head(1)
    return df_scores

mypath = 'United_Kingdom_Weather_Data/'
all_files = [f for f in listdir(mypath) if isfile(join(mypath, f))]
df = pd.DataFrame()

#loops throught the files of data loads them in, splits them and appends them to a dataframe 
for city in all_files:
    df = data_loader(mypath, city, df)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)


linear_scores = model(linear_parameters()[0], linear_parameters()[1])
linear_ridge_scores = model(ridge_parameters()[0], ridge_parameters()[1])
linear_lasso_scores = model(lasso_parameters()[0], lasso_parameters()[1])
knn__scores = model(knn_parameters()[0], knn_parameters()[1])
svr_scores = model(svr_parameters()[0], svr_parameters()[1])
decision_tree_scores = model(decision_tree_parameters()[0], decision_tree_parameters()[1])
random_forest_scores = model(random_forest_parameters()[0], random_forest_parameters()[1])

scores_df = pd.DataFrame(knn__scores)
scores_df = scores_df.append(linear_scores)
scores_df = scores_df.append(linear_ridge_scores)
scores_df = scores_df.append(linear_lasso_scores)
scores_df = scores_df.append(svr_scores)
scores_df = scores_df.append(decision_tree_scores)
scores_df = scores_df.append(random_forest_scores)
scores_df.plot(x='model_type', y=['val_score', 'val_MSE', 'val_MAE'], kind='bar', grid=True, title='Validation R_2 score, MSE and MAE for each top performing model')
plt.savefig(fname='plots/Three_score_metrics', bbox_inches='tight')
scores_df.plot(x='model_type', y=['val_score', 'time_taken'], kind='bar', grid=True, title='Validation score and time taken for each top performing model')
plt.savefig(fname='plots/validation_score', bbox_inches='tight')
scores_df.plot.scatter(x='time_taken', y='val_score', kind='scatter', grid=True, title='Validation R^2 score and time taken for each top performing model')
plt.savefig(fname='plots/validation_score_vs_time_taken', bbox_inches='tight')
scores_df.plot(x='model_type', y='val_score', kind='bar', grid=True, title='Validation R^2 score against time taken for each top performing model')
plt.savefig(fname='plots/validation_score_bar', bbox_inches='tight')
scores_df.plot(x='model_type', y='val_MSE', kind='bar', grid=True, title='Validation mean squared error')
plt.savefig(fname='plots/validation_mse_bar', bbox_inches='tight')
scores_df.plot(x='model_type', y='val_MAE', kind='bar', grid=True, title='Validation mean absolute error')
plt.savefig(fname='plots/validation_mae_bar', bbox_inches='tight')
scores_df.plot(x='model_type', y='time_taken', kind='bar', grid=True, title='Time taken')
plt.savefig(fname='plots/time_taken_bar')
scores_df.plot(x='model_type', y='time_taken', kind='line', grid=True, title='Time taken')
plt.xticks(rotation=90)
plt.savefig(fname='plots/time_taken_line',  bbox_inches='tight')
scores_df.plot(x='model_type', y='val_score', kind='line', title='Validation R^2 score')
plt.xticks(rotation=90)
plt.savefig(fname='plots/validation_score_line', bbox_inches='tight')
plt.show()

# %%
scores_df
# %%
