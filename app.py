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


@app.route('/predict', methods=['POST'])
def predict():
    form_data = request.form.to_dict()
    print("FORM DATA")
    print(form_data)
    
    df_input = pd.DataFrame.from_records([form_data])
    df_input = pd.DataFrame(df_input)
    print("INPUT DATAFRAME")
    print(df_input)       
    df_input = df_input.astype('int')
    print(df_input.info())
    
    sample_df = pd.DataFrame(columns = data_columns)
    main_df = sample_df.append(df_input)
    main_df = main_df.fillna(0)
    print("MAIN DATAFRAME")
    print(main_df)
    print(main_df.info())
    print()
    
    return flask.render_template('index.html', predicted_value="{}".format(str(pred_val)), any_message=msg)


if __name__ == '__main__':
    random_regressor = joblib.load("./pickles/random_regressor.pkl")
    min_max_scaler = joblib.load("./pickles/min_max_scaler.pkl")
    data_columns = joblib.load("./pickles/data_columns.pkl")
    
    app.run(host='0.0.0.0', port=8088)
    
    