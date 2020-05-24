#!/usr/bin/env python
# coding: utf-8
"""
Created on Fri May  8 08:32:32 2020

@author: paras
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import joblib
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import MinMaxScaler

from scipy.stats import chi2_contingency

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PowerTransformer

from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn import metrics as skmetrics
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import classification_report,confusion_matrix
import xgboost as xgb


#///////////////////////////////////////////////////////////////////////
raw_df = pd.read_csv("cardio_train.csv", sep=";")
#///////////////////////////////////////////////////////////////////////
print(raw_df.duplicated().sum())
raw_df.drop_duplicates(inplace=True)

def feature_outlier_removal(data, feature, min_q, max_q):
    feature_min_outlier_mask = data[feature] > data[feature].quantile(min_q)
    feature_max_outlier_mask = data[feature] < data[feature].quantile(max_q)
    data = data[(feature_min_outlier_mask) & (feature_max_outlier_mask)]
    print(feature, "min: ", min(data[feature]))
    print(feature, "max: ", max(data[feature]))
    return data

def numerical_outlier_removal(data):
    data = feature_outlier_removal(data, "price", 0.01, 0.999)
    data = feature_outlier_removal(data, "sqfeet", 0.002, 0.999)
    return data

raw_df = numerical_outlier_removal(df)
print("\nOutliers Removed :", df.shape[0] - raw_df.shape[0])
print("Data Shape: ", raw_df.shape[0])
#///////////////////////////////////////////////////////////////////////
raw_df = raw_df[raw_df['beds'] <= 6] 
raw_df = raw_df[raw_df['baths'] <= 3.5] 
print("Data Shape: ", raw_df.shape[0])
#///////////////////////////////////////////////////////////////////////
lat_min_mask = raw_df['lat'] >= 19.50139
lat_max_mask = raw_df['lat'] <= 64.85694
raw_df = raw_df[(lat_min_mask) & (lat_max_mask)]

long_min_mask = raw_df['long'] >= -161.75583
long_max_mask = raw_df['long'] <= -68.01197
raw_df = raw_df[(long_min_mask) & (long_max_mask)]

print("lat min: ", min(raw_df.lat))
print("lat max: ", max(raw_df.lat))
print("long min: ", min(raw_df.long))
print("long max: ", max(raw_df.long))
print("Data Shape: ", raw_df.shape[0])
#///////////////////////////////////////////////////////////////////////
def Lat_long_outlier_removal(data):
    data = feature_outlier_removal(data, "lat", 0.01, 0.999)
    data = feature_outlier_removal(data, "long", 0.01, 0.999)
    return data

lat_long_df = Lat_long_outlier_removal(raw_df)
print("\nOutliers Removed :", raw_df.shape[0] - lat_long_df.shape[0])
print("Data Shape: ", lat_long_df.shape[0])
#///////////////////////////////////////////////////////////////////////
#///////////////////////////////////////////////////////////////////////
missing = raw_df.isnull().sum()
missing = missing[missing > 0]
print(missing)
missing.sort_values(inplace=True)
try:
    missing.plot.bar()
except:
    pass
#///////////////////////////////////////////////////////////////////////
decide_cols = ["beds", "baths", "cats_allowed", "dogs_allowed", 
               "smoking_allowed", "wheelchair_access", 
               "electric_vehicle_charge", "comes_furnished", "price" ]

X_train = raw_df[decide_cols][raw_df["laundry_options"].isna()==False]
y_train = raw_df["laundry_options"][raw_df["laundry_options"].isna()==False]
X_test = raw_df[decide_cols][raw_df["laundry_options"].isna()==True]
 
neigh = KNeighborsClassifier(n_neighbors=5)
neigh.fit(X_train, y_train)
laundry_pred = neigh.predict(X_test)
print(laundry_pred)
print(laundry_pred.size)

# filling missing values
raw_df["laundry_options"][raw_df["laundry_options"].isna()==True] = laundry_pred

#after imputation
print(raw_df["laundry_options"].value_counts())
print(raw_df["laundry_options"].isna().sum())
#///////////////////////////////////////////////////////////////////////
decide_cols = ["beds", "baths", "cats_allowed", "dogs_allowed", 
               "smoking_allowed", "wheelchair_access", "electric_vehicle_charge",
               "comes_furnished", "price" ]

X_train = raw_df[decide_cols][raw_df["parking_options"].isna()==False]
y_train = raw_df["parking_options"][raw_df["parking_options"].isna()==False]
X_test = raw_df[decide_cols][raw_df["parking_options"].isna()==True]
 
neigh = KNeighborsClassifier(n_neighbors=7)
neigh.fit(X_train, y_train)
laundry_pred = neigh.predict(X_test)
print(laundry_pred)
print(laundry_pred.size)

# filling missing values
raw_df["parking_options"][raw_df["parking_options"].isna()==True] = laundry_pred

#after imputation
print(raw_df["parking_options"].value_counts())
print(raw_df["parking_options"].isna().sum())
#///////////////////////////////////////////////////////////////////////
print(raw_df.isnull().sum())
raw_df.dropna(inplace=True)

clean_df = raw_df.copy()
clean_df.describe()
#///////////////////////////////////////////////////////////////////////
#///////////////////////////////////////////////////////////////////////
try:
    clean_df = clean_df.drop(['url', 'region_url', 'image_url'], axis=1)
except:
    pass
#///////////////////////////////////////////////////////////////////////
def sqfeet_range_column(data, feature='sqfeet'):
    if data[feature] < 300:
        return 'single room'
    if data[feature] >= 300 and data[feature] < 500:
        return 'mini'
    if data[feature] >= 500 and data[feature] < 1000:
        return 'small'
    if data[feature] >= 1000 and data[feature] < 1500:
        return 'medium'
    if data[feature] >= 1500 and data[feature] < 2000:
        return 'large'
    if data[feature] >= 2000 and data[feature] < 2500:
        return 'extra large'
    if data[feature] >=2500:
        return 'mansion'
    

clean_df['sqfeet_range'] = clean_df.apply(sqfeet_range_column, axis=1)
clean_df.sqfeet_range.value_counts()
#///////////////////////////////////////////////////////////////////////
kmeans = KMeans(n_clusters=8, random_state=0)
lat_long_pred = kmeans.fit_predict(clean_df[["lat", "long"]])
print(lat_long_pred.size)
clean_df['lat_long_cluster'] = lat_long_pred

clean_df = clean_df.reset_index(drop=True)
#///////////////////////////////////////////////////////////////////////
fig, axes = plt.subplots(figsize=(10,10))
plt.scatter(x=clean_df['lat'], y=clean_df['long'], c=lat_long_pred)
plt.show()
#///////////////////////////////////////////////////////////////////////
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer 
sid_obj = SentimentIntensityAnalyzer()

description_dict = {"description_negative":[], "description_neutral": [], "description_positive":[]}


loop = clean_df.shape[0]
for i in tqdm(range(loop), position=0, leave=True):
    desc = str(clean_df.description[i])
    sentiment_dict = sid_obj.polarity_scores(desc) 
    description_dict["description_negative"].append(sentiment_dict["neg"])
    description_dict["description_neutral"].append(sentiment_dict["neu"])
    description_dict["description_positive"].append(sentiment_dict["pos"]) 
#///////////////////////////////////////////////////////////////////////
desc_df = pd.DataFrame(description_dict)
print(desc_df.shape)
desc_df.head()

#///////////////////////////////////////////////////////////////////////
clean_df = pd.concat([clean_df, desc_df], axis=1)
clean_df = clean_df.drop(["description"], axis=1)
#///////////////////////////////////////////////////////////////////////
clean_df.corr()
f, ax = plt.subplots(figsize=(16, 16))
sns.heatmap(clean_df.corr(), annot=True, linewidths=0.5, square=True, 
            vmax=0.3, center=0, cmap=sns.cubehelix_palette())
#///////////////////////////////////////////////////////////////////////
#///////////////////////////////////////////////////////////////////////
df = clean_df.copy()
df.dropna(inplace=True)
df.shape
#///////////////////////////////////////////////////////////////////////
df = pd.get_dummies(df,drop_first=False)
print(df.head())
#///////////////////////////////////////////////////////////////////////
df_X = df.drop(["id", "price"], axis=1)
df_y = df.loc[:, "price"]
#///////////////////////////////////////////////////////////////////////
scaler = MinMaxScaler()
df_X = scaler.fit_transform(df_X)
print(df_X)
#///////////////////////////////////////////////////////////////////////
X_train, X_test, y_train, y_test = train_test_split(df_X, df_y, 
                                                    test_size=0.2, 
                                                    random_state=42)
#///////////////////////////////////////////////////////////////////////
def calculate_regression_metrics(y_test, predictions):
    mse = skmetrics.mean_squared_error(y_test, predictions)
    mae = skmetrics.mean_absolute_error(y_test, predictions)
    r2_error = skmetrics.r2_score(y_test, predictions)

    result = {'mean_squared_error': mse, 'mean_absolute_error': mae, 'r2_score': r2_error}
    return result 
#///////////////////////////////////////////////////////////////////////
regressor_model = RandomForestRegressor()
regressor_model.fit(X_train, y_train)
pred = regressor_model.predict(X_test)
pred = pred.reshape(-1, 1)

print(pred)
print("//////////////////////////////////////")
print(y_test)
print("//////////////////////////////////////")

calculate_regression_metrics(y_test, pred)
#///////////////////////////////////////////////////////////////////////

#///////////////////////////////////////////////////////////////////////

#///////////////////////////////////////////////////////////////////////

#///////////////////////////////////////////////////////////////////////

#///////////////////////////////////////////////////////////////////////

#///////////////////////////////////////////////////////////////////////
joblib.dump(model, './pickles/classifier_model.pkl') 



