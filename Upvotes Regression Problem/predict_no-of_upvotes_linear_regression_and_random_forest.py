# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 14:40:32 2019

@author: expert
"""

# Prediction of number of upvotes
# data used on uploaded with it 

#importing packages
import pandas
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

#reading training and test data
training_data = pandas.read_csv("train_NIR5Yl1.csv")
testing_data = pandas.read_csv("test_8i3B3FC.csv")
submission_format = pandas.read_csv("sample_submission_OR5kZa5.csv")

training_data.head(15)

#There are 10 different tags
len(training_data.Tag.unique())

#Some views are lesser than upvotes
(training_data[training_data['Views'] < training_data['Upvotes']])

#Univariate analysis
# NUMERICAL VARIABLES

#tag c question has the most reputation
training_data[training_data['Reputation']==training_data.Reputation.max()]

#Maximum Upvotes in Tag c is 360073 and answers is 61
training_data[training_data.Tag == 'c'].Answers.max()

tags = list(training_data.Tag.unique())
#maximum upvotes belong to the tag j 
for row in tags:
    print(training_data[training_data['Tag']==row].Upvotes.max() , row)

#Tag h has been answered more times
for row in tags:
    print(training_data[training_data['Tag']==row].Answers.max() , row)

#Tag j has the maximum number of views
for row in tags:
    print(training_data[training_data['Tag']==row].Views.max() , row)

# Categorical variable analysis
training_data['Tag'].value_counts().plot.bar(title ="Tag")
#There are lot of question belonges to Tag c and j

#MISSING VALUE IDENTIFICATION

training_data.isnull().sum()
#There are no missing values 

#find the correlation for numerical variables
correlated_data = training_data.corr(method="pearson")

# ID and username can be neglected 
actual_training_data_features = training_data[['Reputation','Answers','Views']]
actual_training_data_target = training_data['Upvotes']
actual_testing_data = testing_data[['Reputation','Answers','Views']]

#One hot encoding for the column Tag for training
new_dataframe = pandas.get_dummies(training_data.Tag)
actual_training_data_features = pandas.concat([actual_training_data_features,new_dataframe],axis=1,ignore_index=True)

#One hot encoding for the column Tag for testing
new_dataframe = pandas.get_dummies(testing_data.Tag)
actual_testing_data = pandas.concat([actual_testing_data,new_dataframe],axis=1,ignore_index=True)

#Build a model LR
regression = LinearRegression()
regression.fit(actual_training_data_features,actual_training_data_target)
regression.intercept_

#RandomForest Model
ranForest = RandomForestRegressor(n_estimators = 90, random_state = 42)
ranForest.fit(actual_training_data_features,actual_training_data_target)

#predict linear regression
prediction_results = regression.predict(actual_testing_data)

#predict RandomForest regression
prediction_results_RF = ranForest.predict(actual_testing_data)

#writing in submission format
submission_format = pandas.DataFrame()
submission_format['ID'] = pandas.DataFrame(testing_data.ID)
submission_format['Upvotes'] = pandas.DataFrame(prediction_results_RF)
submission_format.to_csv("sample_submission_OR5kZa5.csv")
