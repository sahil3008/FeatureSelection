#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 19:42:47 2020

@author: sahil308
"""
import pandas_profiling
import os
import pandas as pd
import numpy as np
from datetime import datetime, date 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, RobustScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
import xgboost
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import VarianceThreshold
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score, median_absolute_error
from sklearn import metrics
from math import sqrt
from sklearn.utils import shuffle
from sklearn.linear_model import Lasso, LassoCV, LinearRegression
from pprint import pprint
from sklearn.model_selection import StratifiedKFold
from scipy import stats
# Graphics customized color map
from matplotlib.colors import LinearSegmentedColormap

os.chdir('/Users/sahil308/Desktop/ResearchCali')

# Reading the combined data file
data = pd.read_csv('final_data.csv', low_memory=False)
data.head()
data.columns
data.dtypes
#scaled_features.isnull().sum()
#principalDf.isnull().sum()

print('Unique values for each feature:\n',data.isnull().sum())
print('Unique values for each feature:\n',data.nunique())

# Keeping only  the relevant variables  
data = data[['host_since', 'host_is_superhost', 'host_has_profile_pic',
       'host_identity_verified', 'is_location_exact', 'property_type', 'room_type',
       'accommodates', 'bathrooms', 'bedrooms', 'beds','amenities',
       'price', 'security_deposit', 'cleaning_fee', 'guests_included',
       'extra_people', 'minimum_nights', 'number_of_reviews', 'number_of_reviews_ltm',
       'review_scores_rating', 'review_scores_accuracy', 'review_scores_cleanliness',
       'review_scores_checkin', 'review_scores_communication', 'review_scores_location',
       'review_scores_value', 'requires_license', 'instant_bookable', 'cancellation_policy',
       'require_guest_profile_picture', 'require_guest_phone_verification',
       'calculated_host_listings_count', 'reviews_per_month', 'market']]

# Adding a filter to remove observations with displaced variables
filt = (data['host_is_superhost'] == 'f') | (data['host_is_superhost'] == 't')
data = data[filt]

# Removing observations with missing values for any of the below variables
data.dropna(axis='index', how='any', subset=['price', 'bathrooms', 'bedrooms', 'beds'], inplace=True)

# Formatting amount variables
amount_cols = ['price', 'security_deposit', 'cleaning_fee', 'extra_people']
for i in amount_cols:
    data[i] = data[i].str.replace(r'[^-+\d.]', '').astype(float)
    
# Removing observations with incorrect information
data = data.loc[~((data['price'] == 0) | (data['bathrooms'] == 0)), :]

# Removing observations which have target variable more than 3 standard deviations away from mean
outlier_price = data['price'].mean() + 3*data['price'].std()
data = data[data['price']< outlier_price]

#data['price'].agg(['mean', 'median','std'])

# Missing Value imputation
data['security_deposit'].fillna(data['security_deposit'].median(), inplace=True)
data['cleaning_fee'].fillna(data['cleaning_fee'].median(), inplace=True)
data['reviews_per_month'].fillna(0, inplace=True)

cols = ['review_scores_rating','review_scores_accuracy','review_scores_cleanliness', 'review_scores_checkin',
       'review_scores_communication', 'review_scores_location', 'review_scores_value']   
data.loc[data['reviews_per_month'] == 0, cols] = 0

median_review = data[cols].median()
data[cols] = data[cols].fillna(median_review )

# Binary encoding of variables
binary_cols = ['host_is_superhost', 'host_identity_verified','is_location_exact',
            'requires_license', 'instant_bookable', 'require_guest_profile_picture',
            'require_guest_phone_verification', 'host_has_profile_pic']
for i in binary_cols:
    data[i] = data[i].replace({'t': 1, 'f':0})
    
#Converting host_since to number of years
data['host_since'] = pd.to_datetime(data['host_since'])
data['host_since'] = ( pd.datetime.now().timestamp() - data['host_since'].apply(lambda x:x.timestamp()))/(60*60*24*365.24)

# Removing , from market
data['market'] = data['market'].str.replace(',', '')

# Replacing categories
data['property_type'].replace({
    'Condominium': 'Apartment',
    'Guesthouse': 'House',
    'Townhouse': 'House',
    'Guest suite': 'House',
    'Bungalow': 'Bungalow/Loft',
    'Loft': 'Bungalow/Loft'
    }, inplace=True)

data['cancellation_policy'].replace({
    'super_strict_30': 'strict_14_with_grace_period',
    'super_strict_60': 'strict_14_with_grace_period',
    'strict': 'strict_14_with_grace_period',
    'luxury_moderate': 'moderate'
    }, inplace=True)

# Scaling
numeric_cols = ['number_of_reviews', 'number_of_reviews_ltm', 'review_scores_rating', 'review_scores_accuracy',
                'review_scores_cleanliness', 'review_scores_checkin', 'review_scores_communication',
                'review_scores_location', 'review_scores_value', 'reviews_per_month','accommodates',
                'bathrooms','bedrooms','beds', 'security_deposit', 'cleaning_fee', 'guests_included',
                'extra_people', 'minimum_nights','calculated_host_listings_count', 'host_since']
scaled_features = data.copy()
features = scaled_features[numeric_cols]
scaler = StandardScaler().fit(features.values)
features = scaler.transform(features.values)
#Assign the result to those two columns:
scaled_features[numeric_cols] = features

# Creating instance of one-hot-encoder
data_dummy = pd.get_dummies(scaled_features[['property_type', 'room_type', 'cancellation_policy', 'market']]).max(level=0, axis=1)
data_dummy.columns

# Keeping only those dummy variables which together represent majority of data
dummy_vars = data_dummy[['property_type_House', 'property_type_Apartment', 'property_type_Villa',
                         'property_type_Bungalow/Loft','room_type_Entire home/apt', 'room_type_Private room', 
                         'room_type_Shared room','cancellation_policy_strict_14_with_grace_period', 
                         'cancellation_policy_moderate', 'cancellation_policy_flexible',
                         'market_South Bay CA', 'market_San Francisco', 'market_San Diego',
                         'market_East Bay CA', 'market_Los Angeles']]

#Dropping the original variables from the dataframe
scaled_features.drop(columns=['property_type', 'room_type', 'cancellation_policy', 'market'], inplace=True)

#Merging the original dataset with dummy variables
final_data = pd.merge(scaled_features, dummy_vars, left_index = True, right_index = True, how = 'left')

# Correlation Heatmap
principalDf.corr().style.background_gradient(cmap='coolwarm') 

import seaborn as sns

Var_Corr = principalDf.corr()
# plot the heatmap and annotation on it
sns.heatmap(Var_Corr, xticklabels=Var_Corr.columns, yticklabels=Var_Corr.columns, annot=True)

# Creating a copy of dataset to be used in further steps
final_data2 = final_data.copy()
    
# Creating a variable for each amenity in the amenity list        
amenities_data = set()

def amenities_1(val):
    val_data = val.replace("}", "").replace("{", "").replace(" ", "_").replace("'", "").replace('"', "").split(',')
    for j in val_data:
        if ('translation_missing' not in j and j != ''):
            amenities_data.add(j)

final_data2['amenities'].apply(amenities_1)
#print(amenities_data)

def amenities_2(val, amenity):
    val_data = val.replace("{", "").replace("}", "").replace("'", "").replace('"', "").replace(" ", "_").split(',')
    for j in val_data:
        if (j == amenity):
            return 1
    return 0

for amenity in amenities_data:
    final_data2.insert(len(list(final_data2)), amenity, 0)
    final_data2[amenity] = final_data2['amenities'].apply(lambda x: amenities_2(x, amenity))

#final_data2.iloc[:,44].name

# Dropping the original variable amenities
final_data2.drop(columns=['amenities'], axis=1, inplace=True)
final_data.drop(columns=['amenities'], axis=1, inplace=True)

#Creating a new dataframe for amenities in amenity list and keeping only those amenities which are there in 
#more than 90% of the listings
df = final_data2.iloc[:, 45:]
df1 = df.loc[:, df.sum() > 7356]
df1.columns = df1.columns.str.lower()

# Merging the amenities dataset with the final dataset
principalDf = pd.merge(final_data, df1, left_index = True, right_index = True, how = 'left')
cols = list(principalDf.columns.values) #List of all of the columns
cols.pop(cols.index('price')) #Remove price from list
principalDf = principalDf[cols+['price']] 

# Low Variance filter with threshhold 0.95
threshold_n=0.95
sel = VarianceThreshold(threshold=(threshold_n* (1 - threshold_n) ))
sel_var=sel.fit_transform(principalDf)
principalDf = principalDf[principalDf.columns[sel.get_support(indices=True)]]

principalDf.columns[sel.get_support(indices=True)].to_list()

# Removing Correlated features
correlated_features = set()
# Separating X and y
X = principalDf.drop('price', axis=1)
y = principalDf.price
correlation_matrix = X.corr()

for i in range(len(correlation_matrix.columns)):
    for j in range(i):
        if abs(correlation_matrix.iloc[i, j]) > 0.7:
            colname = correlation_matrix.columns[i]
            correlated_features.add(colname)
            if colname in principalDf.columns:
                    del principalDf[colname]
 
# Log Transform of Target variables
principalDf['log_price'] = np.log(principalDf['price'])   

# Shuffling the dataset to avoid any bias
principalDf = shuffle(principalDf)

#Separating X and y variables
Selected_X = principalDf.drop(columns= ['price', 'log_price'], axis=1)
y_untransformed = principalDf['price']
y_log = principalDf['log_price']

# Using forest for variable importance
extra_tree_forest = ExtraTreesRegressor(n_estimators = 100, 
                                         max_features = 10, random_state=42) 
  
# Training the model 
extra_tree_forest.fit(Selected_X, y_log) 
  
# Computing the importance of each feature 
feature_importance = extra_tree_forest.feature_importances_ 

feature_name = Selected_X.columns.tolist()
# Zipping the two containers based on indexes- feature importance and feature name to create a dataframe
df_filter_feat = list(zip(extra_tree_forest.feature_importances_, feature_name))
# Creating a dataframe from the above created list
df_filter_feat  = pd.DataFrame(df_filter_feat ,columns=["Importance","Feature_Name"])
# Keeping only those features that has an importance score of greater than zero
df_imp_feat = df_filter_feat [df_filter_feat .Importance != 0]
# creating a sorted dataframe based on descending order of feature importance
df_imp_feat_sort = df_imp_feat.sort_values(by='Importance', ascending = False)

# taking the sorted dataframe into a csv file to inspect feature importance values
df_imp_feat_sort.to_csv('feature importance0.8.csv', index = False)
df_imp_feat_sort = pd.read_csv('feature importance.csv')
# Keeping only the top 15 features based on feature importance values
df_kept_feat= df_imp_feat_sort.head(15)
features_high = df_kept_feat.Feature_Name.tolist()
kept_cols = principalDf[features_high]


# Linear Regression
x_train , x_test, Y_train, Y_test = train_test_split(kept_cols, y_log, test_size = 0.30, random_state = 42)    
lm = LinearRegression()
lm.fit(x_train, Y_train)
y_pred_tr = lm.predict(x_train)
y_pred = lm.predict(x_test)

print("R-squared value  of Train by Linear Regression:", round(r2_score(Y_train, y_pred_tr),3))
print("Mean Squared Error:", mean_squared_error(Y_train, y_pred_tr))

print("R-squared value  of Train by Linear Regression:", round(r2_score(Y_test, y_pred),3))
print("Mean Squared Error:", mean_squared_error(Y_test, y_pred))


print(lm.intercept_, lm.coef_)
lin_reg_coef = pd.DataFrame(list(zip(Selected_X,(lm.coef_))),columns=['Feature','Coefficient'])
lin_reg_coef.sort_values(by='Coefficient',ascending=False)


# Train test split
X_train , X_test, y_train, y_test = train_test_split(kept_cols,y_log, test_size = 0.30, random_state = 42)    

#Hyperparameters tuning
print('Parameters currently used:\n')
pprint(xgb_reg.get_params())

# Create the parameter grid based on the results of random search 
param_grid = {'xgbregressor__learning_rate': [0.1, 0.05], 
              'xgbregressor__max_depth': [3, 5, 7],
              'xgbregressor__n_estimators': [200, 300, 400, 500]}

grid_search = GridSearchCV(estimator = xgb_reg,
                           param_grid = param_grid, 
                           cv = 3, n_jobs = -1, verbose = 2, 
                           scoring = 'neg_median_absolute_error')

grid_search.fit(X_train , y_train)

grid_search.best_params_

# Fitting the XGBoost model
xgb_reg = xgb.XGBRegressor(random_state = 42,learning_rate = 0.1, max_depth = 5, n_estimators = 500)
xgb_reg.fit(X_train, y_train)
xgb_trn = xgb_reg.predict(X_train)
xgb_val = xgb_reg.predict(X_test)

# Printing the results
print("\nTraining MSE:", round(mean_squared_error(y_train, xgb_trn),4))
print("Validation MSE:", round(mean_squared_error(y_test, xgb_val),4))
print("\nTraining r2:", round(r2_score(y_train, xgb_trn),4))
print("Validation r2:", round(r2_score(y_test, xgb_val),4))
print("Median Absolute Error: " + str(round(median_absolute_error(np.exp(y_test), np.exp(xgb_val)), 2))) 
print("Median Absolute Error: " + str(round(median_absolute_error(np.exp(y_train), np.exp(xgb_trn)), 2))) 
mse_train = mean_squared_error(np.exp(y_train), np.exp(xgb_trn))
rmse_train = round(sqrt(mean_squared_error(np.exp(y_train), np.exp(xgb_trn))))
print(mse_train, rmse_train)
mse_test = mean_squared_error(np.exp(y_test), np.exp(xgb_val))
rmse_test = round(sqrt(mean_squared_error(np.exp(y_test), np.exp(xgb_val))))
print(mse_test, rmse_test)


mse_train = mean_squared_error(y_train, xgb_trn)
mse_test = mean_squared_error(y_test, xgb_val)


# Plotting feature importances
plt.barh(df_kept_feat.Feature_Name, df_kept_feat.Importance) 
plt.title("Importance of features in the XGBoost model", fontsize=14)
plt.xlabel("Feature importance")
plt.show()
plt.savefig('features16.png')

#Fitting the random forest model
X_train1 , X_test1, y_train1, y_test1 = train_test_split(kept_cols, y_log, test_size = 0.30, random_state = 10)  
regressor = RandomForestRegressor(random_state = 42, max_depth = 5, n_estimators = 500)
regressor.fit(X_train1, y_train1)
y_trdata = regressor.predict(X_train1)
y_pred = regressor.predict(X_test1)

#Printing the results
print('Mean Squared Error:', metrics.mean_squared_error(y_train1, y_trdata))
print('Mean Squared Error:', metrics.mean_squared_error(y_test1, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test1, y_pred)))
print("\nTraining r2:", round(r2_score(y_train1, y_trdata),4))
print("Validation r2:", round(r2_score(y_test1, y_pred),4))

# Lasso Model and Feature Selection
X_train,X_test,y_train,y_test=train_test_split(Selected_X,y_log, test_size=0.3, random_state=42)
lasso001 = Lasso(alpha=0.01, max_iter=10e5)
lasso001.fit(X_train,y_train)
x_train_pred = lasso001.predict(X_train)
x_test_pred = lasso001.predict(X_test)

#Printing the results of Lasso
#mse_train = mean_squared_error(np.exp(y_train), np.exp(x_train_pred))
#mse_test = mean_squared_error(np.exp(y_test), np.exp(x_test_pred))
mse_train = mean_squared_error(y_train, x_train_pred)
print("MSE Training:", mse_train)
mse_test = mean_squared_error(y_test, x_test_pred)
print("MSE Validation: ", mse_test)

print("\nTraining r2:", round(r2_score(y_train, x_train_pred),4))
print("Validation r2:", round(r2_score(y_test, x_test_pred),4))

rmse_train = np.sqrt(mse_train)
rmse_test = np.sqrt(mse_test)

coeffs_used = np.sum(lasso001.coef_!=0)
print("number of features used: ", coeff_used)



# Log Price Distribution and QQ plot
sns.distplot(principalDf['log_price'], kde=True,);
fig = plt.figure()
res = stats.probplot(principalDf['log_price'], plot=plt)
print("Skewness: %f" % principalDf['log_price'].skew())
print("Kurtosis: %f" % principalDf['log_price'].kurt())

# Price Distribution and QQ plot
sns.distplot(principalDf['price'], kde=True,);
fig = plt.figure()
res = stats.probplot(principalDf['price'], plot=plt)
print("Skewness: %f" % principalDf['price'].skew())
print("Kurtosis: %f" % principalDf['price'].kurt())











    



#data2 = data[data['price'].isnull()]
#data2 = data[data['beds'].isnull()]
#data10 = data[data['reviews_per_month'].isnull()]
#data10 = data[data['reviews_per_month']==0]
#data10 = data[data['review_scores_rating'].isnull()]
#data10['number_of_reviews'].value_counts()
#data3 = data[data['price'] == 0]
#data2['room_type'].value_counts()
#print 'Number of Beds 0:', len(data[data['beds'] == 0])
#data.to_csv('datadescripwcity.csv', index=False)
#pd.DataFrame(transformed).to_csv("transfile.csv")
#print 'Number of Unique Beds: ', np.unique(data['beds'])
#for i in range(1, 17):
#print 'Beds {}:'.format(i), len(data[data['beds'] == i])
    

    
    




	

	
