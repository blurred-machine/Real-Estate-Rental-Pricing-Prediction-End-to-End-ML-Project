# -*- coding: utf-8 -*-
"""
Created on Mon May 25 18:06:28 2020

@author: paras
"""

from flask import Flask, request
import flask
import joblib

import pandas as pd
import numpy as np

app = Flask(__name__)

@app.route('/')
def index():
    return flask.render_template('index.html')


def clean_data(raw_df):     
    lat_min_mask = raw_df['lat'] >= 19.50139
    lat_max_mask = raw_df['lat'] <= 64.85694
    raw_df = raw_df[(lat_min_mask) & (lat_max_mask)]
    long_min_mask = raw_df['long'] >= -161.75583
    long_max_mask = raw_df['long'] <= -68.01197
    raw_df = raw_df[(long_min_mask) & (long_max_mask)]    
    raw_df.dropna(inplace=True)
    return raw_df


def feature_engineer_data(clean_df):
    try:
        clean_df = clean_df.drop(['id', 'url', 'region_url', 'image_url', 'description'], axis=1)
    except:
        print("Custom Error: drop columns did not execute!!")
    
    lat_long_pred = lat_long_classifier.predict(clean_df[["lat", "long"]])
        
    clean_df['lat_long_cluster'] = lat_long_pred
    clean_df = clean_df.reset_index(drop=True)
    clean_df = clean_df.reindex(sorted(clean_df.columns), axis=1)
    clean_df.fillna(-1)
    clean_df = pd.get_dummies(clean_df,drop_first=False)
    return clean_df

def scale_data(df):
    new_df = min_max_scaler.transform(df)
    return new_df

def prdict_results(df):
    random_regressor_pred = random_regressor.predict(df)
    return random_regressor_pred


@app.route('/predict', methods=['POST'])
def predict():
    form_data = request.form.to_dict()
    print("FORM DATA: //////////////")
    print(form_data)
    
    df_input = pd.DataFrame.from_records([form_data])
    df_input = pd.DataFrame(df_input)
    print("INPUT DATAFRAME: //////////////")
    print(df_input)       
    
    int_cols = ['id', 'sqfeet', 'beds', 'cats_allowed', 
                'dogs_allowed', 'smoking_allowed', 
                'wheelchair_access', 'electric_vehicle_charge', 
                'comes_furnished']
    float_cols = ['baths', 'lat', 'long']
    
    df_input[int_cols] = df_input[int_cols].astype('int64')
    df_input[float_cols] = df_input[float_cols].astype('float64')
    print(df_input.info())
    
    print('CLEANING DATA..............')
    clean_df = clean_data(df_input)
    print('FEATURING DATA.............')
    df_featured = feature_engineer_data(clean_df)  
    
    
    print("DATA COLUMNS: //////////////")
    print(data_columns)
    sample_df = pd.DataFrame(columns = data_columns)
    main_df = sample_df.append(df_featured)
    main_df = main_df.fillna(0)
    print("MAIN DATAFRAME: //////////////")
    print(main_df)
    print(main_df.info())
    print()
    
    print('SCALING DATA.............')
    df_scaled = scale_data(main_df)
    
    pred_val = ""
    pred_val = round(prdict_results(df_scaled), 2)
    print("PREDICTION: ////////////////")
    print(pred_val)
    msg = f"Wohoo! AI predicts the price of this property to be around {pred_val[0]} $"

    
    return flask.render_template('index.html', predicted_value="{}".format("Prediction: "+str(pred_val[0])+" $"), any_message=msg)


if __name__ == '__main__':
    random_regressor = joblib.load("./pickles/random_regressor.pkl")
    min_max_scaler = joblib.load("./pickles/min_max_scaler.pkl")
    data_columns = joblib.load("./pickles/data_columns.pkl")
    lat_long_classifier = joblib.load("./pickles/lat_long_classifier.pkl")
    
    app.run(host='0.0.0.0', port=8088)