"""
Contains functions for getting data from the datasets in a usable format.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder


"""
Get data with all features included
"""
def get_full_data():
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')

    s = (train_df.dtypes == 'object')
    object_cols = list(s[s].index)

    ordinal_encoder = OrdinalEncoder()
    label_train_df = train_df.copy()
    label_train_df[object_cols] = ordinal_encoder.fit_transform(train_df[object_cols])
    label_train_df['energy_star_rating'] = label_train_df['energy_star_rating'].fillna(label_train_df['energy_star_rating'].mean())
    label_train_df['direction_max_wind_speed'] = label_train_df['direction_max_wind_speed'].fillna(1.0)
    label_train_df['direction_peak_wind_speed'] = label_train_df['direction_peak_wind_speed'].fillna(1.0)
    label_train_df['max_wind_speed'] = label_train_df['max_wind_speed'].fillna(1.0)
    label_train_df['days_with_fog'] = label_train_df['days_with_fog'].fillna(label_train_df['days_with_fog'].mean())
    label_train_df = label_train_df.fillna(0)

    y = label_train_df['site_eui'].values
    X = label_train_df.drop(columns=['site_eui', 'id'])
    X = X.values

    return X, y


"""
Get train and test data with the average of some correlated variables
"""
def get_data_avg_vars():
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')

    s = (train_df.dtypes == 'object')
    object_cols = list(s[s].index)

    ordinal_encoder = OrdinalEncoder()
    label_train_df = train_df.copy()
    label_train_df[object_cols] = ordinal_encoder.fit_transform(train_df[object_cols])
    label_train_df['energy_star_rating'] = label_train_df['energy_star_rating'].fillna(label_train_df['energy_star_rating'].mean())
    label_train_df['direction_max_wind_speed'] = label_train_df['direction_max_wind_speed'].fillna(1.0)
    label_train_df['direction_peak_wind_speed'] = label_train_df['direction_peak_wind_speed'].fillna(1.0)
    label_train_df['max_wind_speed'] = label_train_df['max_wind_speed'].fillna(1.0)
    label_train_df['days_with_fog'] = label_train_df['days_with_fog'].fillna(label_train_df['days_with_fog'].mean())
    label_train_df['Avg_min_temp_winter'] = (label_train_df['january_min_temp'] + label_train_df['february_min_temp'] + label_train_df['march_min_temp'] + label_train_df['april_min_temp'] + label_train_df['october_min_temp'] + label_train_df['november_min_temp'] + label_train_df['december_min_temp'])/7
    label_train_df['Avg_max_temp_winter'] = (label_train_df['january_max_temp'] + label_train_df['february_max_temp'] + label_train_df['march_max_temp'] + label_train_df['april_max_temp'] + label_train_df['october_max_temp'] + label_train_df['november_max_temp'] + label_train_df['december_max_temp'])/7
    label_train_df['Avg_temp_winter'] = (label_train_df['january_avg_temp'] + label_train_df['february_avg_temp'] + label_train_df['march_avg_temp'] + label_train_df['april_avg_temp'] + label_train_df['october_avg_temp'] + label_train_df['november_avg_temp'] + label_train_df['december_avg_temp'])/7
    label_train_df['Avg_min_temp_summer'] = (label_train_df['may_min_temp'] + label_train_df['june_min_temp'] + label_train_df['july_min_temp'] + label_train_df['august_min_temp'] + label_train_df['september_min_temp'])/5 
    label_train_df['Avg_max_temp_summer'] = (label_train_df['may_max_temp'] + label_train_df['june_max_temp'] + label_train_df['july_max_temp'] + label_train_df['august_max_temp'] + label_train_df['september_max_temp'])/5
    label_train_df['Avg_temp_summer'] = (label_train_df['may_avg_temp'] + label_train_df['june_avg_temp'] + label_train_df['july_avg_temp'] + label_train_df['august_avg_temp'] + label_train_df['september_avg_temp'])/5 
    label_train_df['Avg_days_below30F'] = (label_train_df['days_below_30F'] + label_train_df['days_below_20F'] + label_train_df['days_below_10F'] + label_train_df['days_below_0F'])/4
    label_train_df = label_train_df.fillna(0)
    months_cols = list(label_train_df.iloc[:,5:41].columns) + ['days_below_30F','days_below_20F', 'days_below_10F','days_below_0F','direction_max_wind_speed', 'direction_peak_wind_speed','snowdepth_inches','avg_temp','days_above_90F']
    Xtraindf = label_train_df.drop(columns=months_cols, axis=1)

    # Calculate average variables
    label_test_df = test_df.copy()
    label_test_df[object_cols] = ordinal_encoder.transform(test_df[object_cols])
    label_test_df['energy_star_rating'] = label_test_df['energy_star_rating'].fillna(label_test_df['energy_star_rating'].mean())
    label_test_df['direction_max_wind_speed'] = label_test_df['direction_max_wind_speed'].fillna(1.0)
    label_test_df['direction_peak_wind_speed'] = label_test_df['direction_peak_wind_speed'].fillna(1.0)
    label_test_df['max_wind_speed'] = label_test_df['max_wind_speed'].fillna(1.0)
    label_test_df['days_with_fog'] = label_test_df['days_with_fog'].fillna(label_test_df['days_with_fog'].mean())
    label_test_df['Avg_min_temp_winter'] = (label_test_df['january_min_temp'] + label_test_df['february_min_temp'] + label_test_df['march_min_temp'] + label_test_df['april_min_temp'] + label_test_df['october_min_temp'] + label_test_df['november_min_temp'] + label_test_df['december_min_temp'])/7
    label_test_df['Avg_max_temp_winter'] = (label_test_df['january_max_temp'] + label_test_df['february_max_temp'] + label_test_df['march_max_temp'] + label_test_df['april_max_temp'] + label_test_df['october_max_temp'] + label_test_df['november_max_temp'] + label_test_df['december_max_temp'])/7
    label_test_df['Avg_temp_winter'] = (label_test_df['january_avg_temp'] + label_test_df['february_avg_temp'] + label_test_df['march_avg_temp'] + label_test_df['april_avg_temp'] + label_test_df['october_avg_temp'] + label_test_df['november_avg_temp'] + label_test_df['december_avg_temp'])/7
    label_test_df['Avg_min_temp_summer'] = (label_test_df['may_min_temp'] + label_test_df['june_min_temp'] + label_test_df['july_min_temp'] + label_test_df['august_min_temp'] + label_test_df['september_min_temp'])/5 
    label_test_df['Avg_max_temp_summer'] = (label_test_df['may_max_temp'] + label_test_df['june_max_temp'] + label_test_df['july_max_temp'] + label_test_df['august_max_temp'] + label_test_df['september_max_temp'])/5
    label_test_df['Avg_temp_summer'] = (label_test_df['may_avg_temp'] + label_test_df['june_avg_temp'] + label_test_df['july_avg_temp'] + label_test_df['august_avg_temp'] + label_test_df['september_avg_temp'])/5 
    label_test_df['Avg_days_below30F'] = (label_test_df['days_below_30F'] + label_test_df['days_below_20F'] + label_test_df['days_below_10F'] + label_test_df['days_below_0F'])/4
    label_test_df = label_test_df.fillna(0)
    months_cols = list(label_test_df.iloc[:,5:41].columns) + ['days_below_30F','days_below_20F', 'days_below_10F','days_below_0F','direction_max_wind_speed', 'direction_peak_wind_speed','snowdepth_inches','avg_temp','days_above_90F']
    Xtestdf = label_test_df.drop(columns=months_cols, axis=1)

    y = Xtraindf['site_eui'].values
    X = Xtraindf.drop(columns=['site_eui', 'id'])
    cols = X.columns
    X = X.values

    return X, y