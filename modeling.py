# -*- coding: utf-8 -*-
"""
1. Defining helper functions
2. Feature engineering
3. Feature selection
4. Building model
5. Making predictions
"""

###
### Load required packages and initial setup
###
import pandas as pd
import numpy as np
import scipy.stats as stat
from sklearn.svm import LinearSVR

import warnings
warnings.filterwarnings("ignore")

seed = 1000
np.random.seed(seed)

import crowdai

###
### Helper Function
###

## Function: Calculate the metrics
def calc_metrics(time_period, predicted_rank):
    
    #subset actual values for prediction time_period
    actuals = model_data.loc[(model_data["time_period"] == time_period) & (model_data["Train"] == 1),:]
    
    #join predictions onto actuals
    actuals["Rank_F6M_pred"] = predicted_rank
    
    #calculate spearman correlation
    spearman = stat.spearmanr(actuals["Rank_F6M"],actuals["Rank_F6M_pred"])[0]
    
    # calculate NDCG = DCG of Top 20% / Ideal DCG of Top 20%
    # subset top 20% predictions
    t20 = actuals.loc[actuals["Rank_F6M_pred"] <= np.nanpercentile(actuals["Rank_F6M_pred"],20),:]
    t20["discount"] = np.amax(actuals["Rank_F6M_pred"])/(np.amax(actuals["Rank_F6M_pred"])+actuals["Rank_F6M_pred"])
    t20["gain"] = t20["Norm_Ret_F6M"]*t20["discount"]
    DCG = np.sum(t20["gain"])
    
    #subset top 20% actuals
    i20 = actuals.loc[actuals["Rank_F6M"] <= np.nanpercentile(actuals["Rank_F6M"],20),:]
    i20["discount"] = np.amax(actuals["Rank_F6M"])/(np.amax(actuals["Rank_F6M"])+actuals["Rank_F6M"])
    i20["gain"] = i20["Norm_Ret_F6M"]*i20["discount"]
    IDCG = np.sum(i20["gain"])
    
    NDCG = DCG/IDCG
    
    # return time_period, spearman correlation, NDCG
    return pd.DataFrame([(time_period,spearman,NDCG)],columns = ["time_period","spearman","NDCG"])

## Function: Train the model
def Training(data, model, train_start_period, prediction_period, features):

    train_window_start = time_periods_index[time_periods == train_start_period][0]
    train_window_end = time_periods_index[time_periods == prediction_period][0]
    
    data_training = data.iloc[range(train_window_start,train_window_end),:]
    
    # fit using training data only (Train == 1)
    model.fit(data_training.loc[data_training["Train"] == 1, features].values, 
              data_training.loc[data_training["Train"] == 1, "Norm_Ret_F6M"].values)
    
    return model

## Function: Model testing (can use the most recent n periods)
def model_testing(model, data, features, imputed_value, n = 0):
    
    # imputation method, can be changed
    data_imputed = data.fillna(imputed_value)
    
    train_results = pd.DataFrame(columns=["time_period","spearman","NDCG"])
    
    if n == 0:
        for time in time_periods[11:]:
            model = Training(data_imputed, model, "1996_2", time, features)
            
            if(time != "2017_1"):
                train_predictions = model.predict(data_imputed.loc[(data_imputed["time_period"] == time) & (data_imputed["Train"] == 1),
                                                                   features].values)
                train_rank_predictions = stat.rankdata(train_predictions*(-1),method="average")
    
                train_results = train_results.append(calc_metrics(time_period = time, predicted_rank = train_rank_predictions))
        
            print("Time period " + time + " completed.")
    
    else:
        for time in time_periods[11:]:
            time_start = time_periods[max(0, (np.where(time_periods == time)[0][0] - n))]
            model = Training(data_imputed, model, time_start, time, features)
        
            
            if(time != "2017_1"):
                train_predictions = model.predict(data_imputed.loc[(data_imputed["time_period"] == time) & (data_imputed["Train"] == 1),
                                                           features].values)
                train_rank_predictions = stat.rankdata(train_predictions*(-1),method="average")
        
                train_results = train_results.append(calc_metrics(time_period = time, predicted_rank = train_rank_predictions))
        
            print("Time period " + time + " completed.")
    
    return train_results

## Function: Model training and make predictions for submission
def model_training_and_prediction_for_submission(model, data, features, imputed_value):
    
    # imputation method, can be changed
    data_imputed = data.fillna(imputed_value)
    
    pred_template_tmp = pred_template.copy()
    pred_template_tmp["Return"] = 0
    
    for time in time_periods[11:]:
        model = Training(data_imputed, model, "1996_2", time, features)
        
        # prediction for submission
        test_predictions = model.predict(data_imputed.loc[(data_imputed["time_period"] == time) & (data_imputed["Train"] == 0),
                                                          features].values)
        test_rank_predictions = stat.rankdata(test_predictions*(-1),method="average")
        pred_template.loc[pred_template["time_period"] == time,"Rank_F6M"] = test_rank_predictions
        
        pred_template_tmp.loc[pred_template_tmp["time_period"] == time,"Return"] = test_predictions
        pred_template_tmp.loc[pred_template_tmp["time_period"] == time,"Rank_F6M"] = test_rank_predictions
        
        print("Time period " + time + " completed.")
        
    return pred_template_tmp

###
### Load data
###
df = pd.read_csv("...")
pred_template = pd.read_csv("...")

###
### Feature engineering
###

# array containing each time period
time_list = np.unique(df["time_period"].values)

# feature name for each x
feature_names = ["X" + str(i) + "_" for i in range(1, 71)]

## Feature: Mean of each feature(X) from all months
new_feature_name = "mean_all_periods"
for feature in feature_names:
    df[feature + new_feature_name] = df.filter(regex=(feature+"\d")).apply(lambda x: np.nanmean(x), axis = 1)
    
## Feature: Median of each feature(X) from all months
new_feature_name = "median_all_periods"
for feature in feature_names:
    df[feature + new_feature_name] = df.filter(regex=(feature+"\d")).apply(lambda x: np.nanmedian(x), axis = 1)   

## Feature: Standard deviation of each feature(X) from all months
new_feature_name = "std_all_periods"
for feature in feature_names:
    df[feature + new_feature_name] = df.filter(regex=(feature+"\d")).apply(lambda x: np.nanstd(x), axis = 1)

## Feature: Max of each the feature(X) from all months
new_feature_name = "max_all_periods"
for feature in feature_names:
    df[feature + new_feature_name] = df.filter(regex=(feature+"\d")).apply(lambda x: np.nanmax(x), axis = 1)

## Feature: Min of each the feature(X) from all months
new_feature_name = "min_all_periods"
for feature in feature_names:
    df[feature + new_feature_name] = df.filter(regex=(feature+"\d")).apply(lambda x: np.nanmin(x), axis = 1)

## Feature: Change of the last value from the first value for each feature(X)(the first to the last)
new_feature_name = "change_all_periods"
for feature in feature_names:
    df[feature + new_feature_name] = df.filter(regex=(feature+"\d")).apply(lambda x: \
      (x[~np.isnan(x)][-1] - x[~np.isnan(x)][0]) if np.sum(~np.isnan(x)) > 1 else np.nan, axis = 1)

## Feature: Change of the last value from previous value for each feature(the second-last to the last)
new_feature_name = "change_second_last_to_last_all_periods"
for feature in feature_names:
    df[feature + new_feature_name] = df.filter(regex=(feature+"\d")).apply(lambda x: \
      (x[~np.isnan(x)][-1] - x[~np.isnan(x)][-2]) if np.sum(~np.isnan(x)) > 1 else np.nan, axis = 1)
    
## Feature: Range of each feature(X) from all months
new_feature_name = "range_all_periods"
for feature in feature_names:
    df[feature + new_feature_name] = df.filter(regex=(feature+"\d")).apply(lambda x: np.nanmax(x) - np.nanmin(x), axis = 1)
    
## Feature: Mean of the differences of each feature(X) from all months
new_feature_name = "mean_diff_all_periods"
for feature in feature_names:
    df[feature + new_feature_name] = df.filter(regex=(feature+"\d")).apply(lambda x: np.nanmean(np.diff(x[~np.isnan(x)])), axis = 1)

## Feature: Median of the differences of each feature(X) from all months
new_feature_name = "median_diff_all_periods"
for feature in feature_names:
    df[feature + new_feature_name] = df.filter(regex=(feature+"\d")).apply(lambda x: np.nanmedian(np.diff(x[~np.isnan(x)])), axis = 1)

## Feature: Standard deviation of the differences of each feature(X) from all months
new_feature_name = "std_diff_all_periods"
for feature in feature_names:
    df[feature + new_feature_name] = df.filter(regex=(feature+"\d")).apply(lambda x: np.nanstd(np.diff(x[~np.isnan(x)])), axis = 1)

## Feature: Max of the differences of each feature(X) from all months
new_feature_name = "max_diff_all_periods"
for feature in feature_names:
    df[feature + new_feature_name] = df.filter(regex=(feature+"\d")).apply(lambda x: np.nanmax(np.diff(x[~np.isnan(x)])) if np.sum(~np.isnan(x)) > 1 else np.nan, axis = 1)

## Feature: Min of the differences of each feature(X) from all months
new_feature_name = "min_diff_all_periods"
for feature in feature_names:
    df[feature + new_feature_name] = df.filter(regex=(feature+"\d")).apply(lambda x: np.nanmin(np.diff(x[~np.isnan(x)])) if np.sum(~np.isnan(x)) > 1 else np.nan, axis = 1)

###
### Feature selection
###

# Data for modeling
model_data = df.copy(deep = True)

# Time periods and indexes
time_periods = np.unique(model_data["time_period"], return_index=True)[0]
time_periods_index = np.unique(model_data["time_period"], return_index=True)[1]

## Correlation between features and targets for each period (the most recent 10 periods) and feature
n = 10
correlation_list = []
for i in range(11, len(time_list)):
    correlation_list_tmp = []
    df_tmp = df.loc[(df["time_period"].isin(time_list[(i - n):i]))&(df["Train"] == 1), :]
    for j in df_tmp.columns[5:]:
        correlation_list_tmp.append(stat.pearsonr(df_tmp[j].values[~np.isnan(df_tmp[j].values)],
                                           df_tmp["Norm_Ret_F6M"].values[~np.isnan(df_tmp[j].values)])[0])
    correlation_list.append(correlation_list_tmp)
    print(i)

correlation_df_by_cumul_period = pd.DataFrame(correlation_list, columns = df.columns[5:])
correlation_df_by_cumul_period.index = time_list[range(10, len(time_list)-1)]    

## Correlation between features and targets for each prediction period and feature
correlation_list = []
for i in range(11, len(time_list)-1):
    correlation_list_tmp = []
    df_tmp = df.loc[(df["time_period"] == time_list[i])&(df["Train"] == 1), :]
    for j in df_tmp.columns[5:]:
        correlation_list_tmp.append(stat.pearsonr(df_tmp[j].values[~np.isnan(df_tmp[j].values)],
                                           df_tmp["Norm_Ret_F6M"].values[~np.isnan(df_tmp[j].values)])[0])   
    correlation_list.append(correlation_list_tmp)
    print(i)

correlation_df_each_by_period = pd.DataFrame(correlation_list, columns = df.columns[5:])
correlation_df_each_by_period.index = time_list[range(11, len(time_list)-1)]

## Sum of the number of different signs between corresponding the most recent 10 periods and prediction periods for each feature
correlation_diff_sign_list = []
for i in correlation_df_by_cumul_period.columns:
    correlation_diff_sign_list.append(np.sum(correlation_df_each_by_period.loc[:, i].\
            values*correlation_df_by_cumul_period.loc[:"2016_1", i].values<0))
    print(i)

## Rank the feature according to the performance

# Get performance for each feature, filter the features with negative performance and rank them
num_top = np.sum(np.array(sorted(correlation_diff_sign_list))<=10)
feature_list_tmp = list(correlation_df_by_cumul_period.columns[np.argsort(correlation_diff_sign_list)][:num_top])
feature_importance_list = []

for ind, i in enumerate(feature_list_tmp):
    model = LinearSVR(random_state = seed) 
    imputed_value = model_data[i].mean()
    outcome = model_testing(model, model_data, [i, ], imputed_value, n = 0)
    feature_importance_list.append(outcome)
    print(ind)

feature_importance_list_positive = np.array([x.mean().mean() for x in feature_importance_list])[np.array([x.mean().mean() for x in feature_importance_list])>0]
feature_list_positive = np.array(feature_list_tmp)[np.array([x.mean().mean() for x in feature_importance_list])>0]
feature_list_tmp_ranked = list(feature_list_positive[np.argsort(feature_importance_list_positive*(-1))])

feature_list_tmp_ranked_tmp = feature_list_tmp_ranked[:]

# Get correlation using each training period
df_corr_list = []
for i in range(11, len(time_list)):
    df_corr_list.append(model_data.loc[model_data["time_period"].isin(time_list[:i]),
            feature_list_tmp_ranked].fillna(model_data.loc[model_data["time_period"].isin(time_list[:i]),
            feature_list_tmp_ranked].mean()).corr())
    print(i)

df_corr = df_corr_list[0].copy(deep = True)
df_corr[:] = 0
for i in range(len(df_corr_list)):
    df_corr += df_corr_list[i]

df_corr = df_corr/len(df_corr_list)

# Filter out the features with high correlation with other features starting from the features with low performance
corr_threshold = 0.8

for i in feature_list_tmp_ranked[::-1]:
    print(i)
    if (df_corr.loc[:, i].abs() >= corr_threshold).sum() > 1:
        feature_list_tmp_ranked_tmp.remove(i)
        df_corr = model_data.loc[:, feature_list_tmp_ranked_tmp].fillna(model_data.loc[:, feature_list_tmp_ranked_tmp]).corr()
  
###
### Modeling
###
    
## LinearSVR model
model = LinearSVR(random_state = seed)
feature_list = feature_list_tmp_ranked_tmp[:26]
imputed_value = 0

pred_temp = model_training_and_prediction_for_submission(model, model_data, feature_list, imputed_value)

# File name for submission 
file_name = "..."

# Output the file
pred_template.to_csv(file_name, index = False)
