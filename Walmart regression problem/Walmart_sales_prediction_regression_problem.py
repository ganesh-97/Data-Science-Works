# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 11:43:18 2019

@author: expert
"""

#Walmart slaes prediction regression problem

import pandas
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.cross_validation import train_test_split
from sklearn.metrics import mean_squared_error
import math

#Reading training and testing data
training_data = pandas.read_csv("Train_UWu5bXk.csv")
testing_data = pandas.read_csv("Test_u94Q5KV.csv")
sample_submission = pandas.read_csv("SampleSubmission_TmnO39y.csv")
training_data.head(10)
training_data = pandas.concat([training_data,testing_data],ignore_index=True)
#display the dtypes 
training_data.dtypes

#display categorical variables
training_data.select_dtypes(include = ['object']).columns

#display numerical variables
training_data.select_dtypes(include=['int','float']).columns

#univariate analysis for categorical variables
#'Item_Identifier', 'Item_Fat_Content', 'Item_Type', 'Outlet_Identifier',
 #      'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type'

#since the Item_identifier and outlet identifier has nothing to do with the
#target variable, it's ommited
training_data['Item_Fat_Content'].value_counts().plot.bar(title = "Item_Fat_Content")
#There are so many Low Fat content products in all those super markets
training_data['Item_Type'].value_counts().plot.bar(title="Item_Type")
#It is identified that there are so many fruits and vegetables when compared to other 
#categories
training_data['Outlet_Size'].value_counts().plot.bar(title="Outlet_Size")
#Most of the outlets are medium in size
training_data['Outlet_Location_Type'].value_counts().plot.bar(title="Outlet_Location_Type")
#Most of the outlets comes under Tier 3
training_data['Outlet_Type'].value_counts().plot.bar(title='Outlet_Type')
#Most of the outlets are supermarket

#Univaiate analysis on numerical variables
#'Item_Weight', 'Item_Visibility', 'Item_MRP',
 #      'Outlet_Establishment_Year', 'Item_Outlet_Sales'

training_data['Item_Weight'].value_counts().plot.hist(title="Item_Weight")
#Most of the product's weight lies within 0 to 20
training_data['Item_Visibility'].value_counts().plot.hist(title="Item_Visibility",bins=5)
training_data['Item_MRP'].describe()
# Most of the prices are 185rupees
training_data['Item_MRP'].value_counts().plot.hist(title="Item_MRP") 
#target variable
training_data['Item_Outlet_Sales'].value_counts().plot.hist(title="Item_Outlet_Sales") 

#BIVARIATE ANALYSIS
#Numerical variables correlation
numerical_correlated_data = training_data.select_dtypes(include=['int','float']).corr(method="pearson")
#it is inferred that the column 'Item_MRP' is correlating with the target value

#Identified variables are 
#Item_MRP
#Item_Fat_Content
#Item_Type
#Outlet_Size
#Outlet_Location_Type
#Outlet_Type

#MISSING VALUES CHECKING

training_data['Item_MRP'].isnull().sum()
training_data['Outlet_Size'].isnull().sum()

actual_training_data = training_data[['Item_MRP','Item_Fat_Content','Item_Type','Outlet_Location_Type','Outlet_Type']]
actual_testing_data = testing_data[['Item_MRP','Item_Fat_Content','Item_Type','Outlet_Location_Type','Outlet_Type']]

#changing the categorical to dummy variables
columns = ['Item_Fat_Content','Item_Type','Outlet_Location_Type','Outlet_Type']

for column in columns:
    new_dataframe = pandas.get_dummies(testing_data[column])
    actual_testing_data = pandas.concat([actual_testing_data, new_dataframe], axis=1)
    del actual_testing_data[column]
    new_dataframe = pandas.DataFrame()

#Linear regression
regression = LinearRegression()
regression.fit(actual_training_data,training_data['Item_Outlet_Sales'])
predicted_data = regression.predict(actual_testing_data)

math.sqrt(mean_squared_error(predicted_data,training_data['Item_Outlet_Sales']))

#Decision Tree
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(actual_training_data,training_data['Item_Outlet_Sales'])
predicted_data = regressor.predict(actual_testing_data)
predicted_data[100]
math.sqrt(mean_squared_error(predicted_data,training_data['Item_Outlet_Sales']))
