# Importing required modules/dependencies
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from datetime import datetime
import xgboost
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score
np.random.seed(1337) # For reproducibility of same results on every run from the neural network model from keras library
from keras import Sequential
from keras.layers import Dense
# Graphics customized color map
from matplotlib.colors import LinearSegmentedColormap

# Reading the dataset into a dataframe
df = pd.read_csv('./DScasestudy.txt', delim_whitespace=True)
print(df)
# Creating a new df after shuffling the observations so as to create random distribution of responders/nonresponders since the data has all one 1's in serial order followed by zero's
df1 = shuffle(df, random_state =10)




# check if null values are present
print("The status of null values is ", df1.isnull().values.any())

# Checking the balance of dataset - The ratio of 1 and 0 is 1:3.3 in the current data distribution which is not completely imbalanced.
response_positive_cnt = len(df1.loc[df1['response']==1])
response_negative_cnt = len(df1.loc[df1['response']== 0])
print("The distribution of response variable (count of 1 and 0) in the dataset is {} and {} respectively".format(response_positive_cnt, response_negative_cnt))

# Creating a dataframe of independent features after dropping the response variable
df_ind_feat = df1.drop(['response'], axis=1)

# Iterating over the dataframe containing independent features and dropping the features from the complete dataframe that has only one distinct value as these variables have zero variance
# and hence will not contribute towards the predictive power  of the model
for col in df_ind_feat:
    if len(df_ind_feat[col].unique()) == 1:
        df_ind_feat.drop(col, inplace=True, axis=1)



# Taking all independent variables that were not dropped in a list
col_list = list(df_ind_feat.columns)


# taking independent and dependent features in different data frame/series
X = df1[col_list]
y= df1['response']



# Reducing the dimensionality even further by determining feature importance scores using ExtraTreesClassifier
model = ExtraTreesClassifier(random_state = 10)
model.fit(X,y.ravel())
#plot graph of feature importances for better visualization- only top 20 features
feat_importances = pd.Series(model.feature_importances_)
feat_importances.nlargest(20).plot(kind='barh')
plt.savefig('feature.png')


# Creating a list of feature importance score and feature names
df_filter_feat = list(zip(model.feature_importances_, col_list))
# Creating a dataframe from the above created list
df_filter_feat  = pd.DataFrame(df_filter_feat ,columns=["Importance","Feature_Name"])
# Keeping only those features that has an importance score of greater than zero
df_imp_feat = df_filter_feat [df_filter_feat .Importance != 0]
# creating a sorted dataframe based on descending order of feature importance
df_imp_feat_sort = df_imp_feat.sort_values(by='Importance', ascending = False)

# taking the sorted dataframe into a csv file to inspect feature importance values
df_imp_feat_sort.to_csv('feature importance filt.csv', index = False)

# Keeping only the top 20 features based on feature importance values
df_kept_feat= df_imp_feat_sort.head(20)


# Setting the index by a variable name for a df
df_kept_feat = df_kept_feat1.set_index('Feature_Name')
# Plotting the graph from df
df_kept_feat.plot(kind='barh', title ="Feature Importance", figsize=(12, 8), fontsize = 14 )
plt.title("Feature Importance", fontweight = 'bold')
# Footnote
plt.annotate('              The plot shows the importance scores of various features which were used in the final model. ', (0,0), (0, -40), xycoords='axes fraction', fontsize = 12,  textcoords='offset points', va='top')
plt.savefig('Feature Importance.png')

# Creating a list of independent variables that were kept in the dataframe and will be used in final model
new_col_list = df_kept_feat['Feature_Name'].tolist()
# Inserting the dependent variable in the feature list
new_col_list.insert(0,'response')
# Creating a new dataframe by just keeping the final 20 independent and  the dependent feature
df_final = df1[new_col_list]
print(df_final)



# Define the plot size for heat map so that labels dont cut off. All below params doesnt help if labels cut off
plt.figure(figsize=(10 ,9))
# Checking correlations of the features using a heatmap
sns.set(font_scale=1.2)
# specifying params required by bar graph from seaborn - depending on how big the figure is the size in annot_kwargs dict param of sns.heatmap should be
rc={'font.size': 32, 'axes.labelsize': 34, 'legend.fontsize': 34.0,
    'axes.titlesize': 34, 'xtick.labelsize': 40, 'ytick.labelsize': 40}
corr_map = sns.heatmap(df_final.corr(),  xticklabels=2, annot=True,  fmt=".2f",annot_kws={"size": 9})

figure = corr_map.get_figure()
figure.savefig('corr_map.png', dpi=400)


# Checking the VIF to check multicolliearity among the selected 20 features
M = add_constant(df_final)
vif= pd.Series([variance_inflation_factor(M.values, i)
               for i in range(M.shape[1])],
              index=M.columns)
vif.to_csv('vif.csv')

# Independent features list
new_ind_feat = new_col_list[1:]
# Creating a dataframes for independent features and a series for dependent feature
X = df_final[new_ind_feat]
y= df_final['response']




# Splitting the data in train test in a ratio of 70:30 for honest assessment
X_train , X_test, y_train, y_test = train_test_split(X,y, test_size = 0.30, random_state = 10)


# Creating a random forest model

rf_model = RandomForestClassifier(random_state = 10)
# fit the model
rf_model.fit(X_train, y_train.ravel())
# Predicting the outcome on test set
predicted_y = rf_model.predict(X_test)

# Check accuracy of the model - upto 3 decimal points

print("Accuracy of Random Forest= {0:.3f}".format(accuracy_score(y_test, predicted_y)))

# Checking precision, recall and f1 because the data has more observations with negative outcome and accuracy may not alone give the best representation of power of the model.
# since we do not have a perfect balance of the response variable in the data, F1 score is the best metric.
print("Classification Report of Random forest = ",classification_report(y_test, predicted_y))
print("Confusion matrix of Random Forest= ",  confusion_matrix(y_test, predicted_y), sep = '\n')

#                                                              Creating an   XGBoost Model- State of art ML Algorithm for structurd data

# Hyperparameter optimization

params = {
    "learning_rate" : [0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
    "max_depth" : [3,4,5,6,8,10,12,15],
    "min_child_weight" : [1,3,5,7],
    "gamma" : [0.0, 0.1, 0.2, 0.3, 0.4],
    "colsample_bytree" : [0.3, 0.4, 0.5,0.7]
}

classifier = xgboost.XGBClassifier()
random_search = RandomizedSearchCV(classifier, param_distributions = params, n_iter = 5, scoring = 'roc_auc', n_jobs = -1, cv=5, verbose =3)

# Defining a function to calculate time taken for finding the best estimators
def timer(start_time = None):
    if not start_time:
        start_time = datetime.now()
        return start_time
    elif start_time:
        thour, temp_sec = divmod((datetime.now()- start_time).total_seconds(), 3600)
        tmin, tsec = divmod(temp_sec,60)
        print('\n Time taken: %i hours %i minutes and %i seconds.' %(thour, tmin, round(tsec,2)))


# Calling the timer function
start_time = timer(None) # Timing starts
random_search.fit(X_train, y_train.ravel())
timer(start_time) # Timer ends for start_time here
#the below statement gives me the best estimators after optimzing the hyperparameters
print(random_search.best_estimator_)
# Substituting the best hyperparameters values in the classifier
classifier = xgboost.XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
              colsample_bynode=1, colsample_bytree=0.7, gamma=0.0,
              learning_rate=0.2, max_delta_step=0, max_depth=15,
              min_child_weight=3, missing=None, n_estimators=100, n_jobs=1,
              nthread=None, objective='binary:logistic', random_state=0,
              reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
              silent=None, subsample=1, verbosity=1)


# Cross validation score
score = cross_val_score(classifier, X_train, y_train.ravel(), cv =10)
print("The cross val score of each fold of cross validation is", score)
# Taking the mean of all 10 rounds of cross validation
print("The mean cross val score from XGboost is", score.mean())

# Since CV may not give a best representation of predictive power of the model, also look at confusion matrix and classification for F1 score, precision and recall
# Fitting the XGboost as well
classifier.fit(X_train, y_train.ravel())
# Predicting the outcome on test set
predicted_y = classifier.predict(X_test)

# Check accuracy of the model - upto 3 decimal points

print("Accuracy of XGBoost= {0:.3f}".format(accuracy_score(y_test, predicted_y)))

# Checking precision, recall and f1 because the data has more observations with negative outcome and accuracy may not alone give the best representation of power of the model

print("Classification Report of XGBoost = ",classification_report(y_test, predicted_y))
print("Confusion matrix of XGboost= ",  confusion_matrix(y_test, predicted_y), sep = '\n')

# Building a neural network model as well to just compare its performance with xgboost

classifier = Sequential()
#First Hidden Layer: Initialization of weights using random normal initializer to generate tensors with normal distribution
classifier.add(Dense(4, activation='relu', kernel_initializer='random_normal', input_dim=20))
#Second  Hidden Layer
classifier.add(Dense(4, activation='relu', kernel_initializer='random_normal'))
#Output Layer
classifier.add(Dense(1, activation='sigmoid', kernel_initializer='random_normal'))

#Compiling the neural network
classifier.compile(optimizer ='adam',loss='binary_crossentropy', metrics =['accuracy'])

#Fitting the data to the training dataset
classifier.fit(X_train,y_train, batch_size=10, epochs=100)

eval_model=classifier.evaluate(X_train, y_train)
print("Loss function and accuracy is ", eval_model)

y_pred=classifier.predict_classes(X_test)
# Using the scikit learn metrics API
Conf_matrrix_NN = confusion_matrix(y_test, y_pred)
print("The confusion matrix of neural net is", Conf_matrrix_NN)
classification_rep_NN = classification_report(y_test, y_pred)
print("The classification report of neural net is", classification_rep_NN)
Accuracy_NN = accuracy_score(y_test, y_pred)
print("The Accuracy of neural net is", Accuracy_NN)

# taking the data fields as a list of tuple
data = [(0.912, 0.918,0.8760 ),
 (0.81, 0.87, 0.80), (0.70, 0.67, 0.67), (0.75, 0.75,0.73)]

# Creating a dataframe from list of tuples.
df = pd.DataFrame(data, columns = ['Random Forest Classifier','XGboost', 'Neural Network' ])
print(df)
width = 0.55
# setting customized colors for individual bars as a list
my_color = ['#FF6600', '#8ba850', '#3d97db' ]

# customized color map
cmap_name= 'my_list'
# Creating a color map from list of specified colors
cm = LinearSegmentedColormap.from_list(cmap_name, my_color)
# Plotting a plot from df (does not support fontweight)
ax = df.plot(kind='bar', title ="Comparison of Algorithms", figsize=(16, 8), legend=True, fontsize=18, position = 0.5, cmap = cm )

# For annotating the values over individual bars (patches command)
for p in ax.patches:
    #  Identify the bar height
    z= np.round(p.get_height(),decimals=3)
    print(z)
    # formatting the decimal to % with 1 decimal point
    z1 = "{0:.1f}%".format(z * 100)
    print(z1)
    # defining the font attributes for labels
    ax.annotate(z1, (p.get_x()+p.get_width()/2., p.get_height()), ha='center', va='center', xytext=(0, 7), textcoords='offset points', fontweight = 'bold')

# y axis ticks
vals = ax.get_yticks()
ax.set_yticklabels(['{:,.0%}'.format(x) for x in vals])
# xaxis tick labels- specify list and iterate over it
check = ["Accuracy", "Precision", "Recall", "F1 Score"]
ax.set_xticklabels([ z for z in check], fontsize =12, rotation = 360)
# setting the labels and title
ax.set_ylabel("Scores",fontsize=12, fontweight = 'bold' )
ax.set_xlabel("Evaulation metric",fontsize=12, fontweight = 'bold')
plt.title("Comparison of Algorithms", fontweight = 'bold')
# Footnote
plt.annotate('                                   The plot shows comparisons of  different algorithms with respect to minority class. ', (0,0), (0, -50), xycoords='axes fraction', fontsize = 14,  textcoords='offset points', va='top')
plt.savefig("comp.png")