# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 11:08:50 2019

@author: expert
"""

import pandas
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.decomposition import PCA

training_file = pandas.read_csv("train.csv")
testing_file = pandas.read_csv("test.csv")
sample_submission_format = pandas.read_csv("Sample_Submission_Tm9Lura.csv")
training_file.columns

#describing each columns 
def dataset_description(dataset):
    for column in training_file.columns:
        print(dataset[column].describe())
        print()
        
#UserID and ProductId are omitted
training_features = training_file[['Gender', 'Age', 'Occupation', 'City_Category','Stay_In_Current_City_Years', 'Marital_Status', 'Product_Category_1',   'Product_Category_2', 'Product_Category_3']]
training_target = training_file['Purchase']        
testing_file = testing_file[['Gender', 'Age', 'Occupation', 'City_Category','Stay_In_Current_City_Years', 'Marital_Status', 'Product_Category_1',   'Product_Category_2', 'Product_Category_3']]

categorical_columns = training_features.select_dtypes(include=['object']).columns
categorical_columns
numerical_columns = training_features.select_dtypes(include=['int','float']).columns
numerical_columns
numerical_data = training_file[['Occupation', 'Marital_Status','Product_Category_1','Product_Category_2', 'Product_Category_3','Purchase']]

training_features.head(5)

for column in categorical_columns:
    print(training_features[column].value_counts())
    print()

#There are more males compared to female and the maximum purchase done
#by the people between the age 26-35
# and City B has the most purchase 
# and people staying for 1 year has done the purchasing compared to 4+ and less than a year

#Univariate Analysis Categorical variable

training_file['Stay_In_Current_City_Years'].value_counts().plot.bar(title="Stay_In_Current_City_Years")

#Numerical variables 
#'Occupation', 'Marital_Status', 'Product_Category_1',
 #      'Product_Category_2', 'Product_Category_3'
training_file['Product_Category_3'].isnull().value_counts()
training_file['Product_Category_3'].value_counts().plot.bar(title ="Product_Category_2")

#The maximum purchase is done by married person since most of the 
#purchase has been done by the age group of people 26-35

#Mostly The product category '5,1 and 8' is been purchased
# In Product_Category_2 there are 173638 entries are null

len(training_file[training_file['Product_Category_2']==8])
training_file['Product_Category_3'].mean()

#Finding correlation within numerical variables
correlated_data = numerical_data.corr(method="pearson")

#Transformed data

actual_training_data = training_features.copy()
actual_target = training_target.copy()

actual_training_data['Product_Category_2'] = actual_training_data['Product_Category_2'].fillna(actual_training_data['Product_Category_2'].max())
actual_training_data['Product_Category_3'] = actual_training_data['Product_Category_3'].fillna(actual_training_data['Product_Category_3'].max())

testing_file['Product_Category_2']=testing_file['Product_Category_2'].fillna(testing_file['Product_Category_2'].max())
testing_file['Product_Category_3']=testing_file['Product_Category_3'].fillna(testing_file['Product_Category_3'].max())

#Encoding

#LABEL ENCODING

label_encoded_training_data = actual_training_data.copy()
label_encoder = LabelEncoder()
for column in categorical_columns:
    label_encoded_training_data[column] = label_encoder.fit_transform(label_encoded_training_data[column])

label_encoded_testing_data = testing_file.copy()
label_encoder = LabelEncoder()
for column in categorical_columns:
    label_encoded_testing_data[column] = label_encoder.fit_transform(label_encoded_testing_data[column])

#One Hot Encoding

OHE_training_data = actual_training_data.copy()
for column in categorical_columns:
    OHE_training_data = pandas.concat([OHE_training_data, pandas.get_dummies(actual_training_data[column])], axis=1)

for column in categorical_columns:
    del OHE_training_data[column]

OHE_testing_data = testing_file.copy()
for column in categorical_columns:
    OHE_testing_data = pandas.concat([OHE_testing_data, pandas.get_dummies(OHE_testing_data[column])], axis=1)

for column in categorical_columns:
    del OHE_testing_data[column]

#Standardising the data

std = StandardScaler()
label_encoded_training_data =  std.fit_transform(label_encoded_training_data)
label_encoded_testing_data =  std.fit_transform(label_encoded_testing_data)


#Apply PCA in OHE_training and testing data

pca = PCA(n_components = 3)
OHE_training_data = pca.fit_transform(OHE_training_data)
OHE_testing_data = pca.fit_transform(OHE_testing_data)

label_encoded_training_data =  pca.fit_transform(label_encoded_training_data)
label_encoded_testing_data =  pca.fit_transform(label_encoded_testing_data)

#Building models

#LinearRegression

linear = LinearRegression()
linear.fit(label_encoded_training_data,actual_target)

y_predicted = linear.predict(label_encoded_testing_data)

#DecisionTreeRegression and prediction

regressor_decision = DecisionTreeRegressor()
regressor_decision.fit(label_encoded_training_data,actual_target)

y_predicted = regressor_decision.predict(label_encoded_testing_data)

#RandomForest and prediction

randomforest =  RandomForestRegressor(n_estimators=100, min_samples_leaf=52,n_jobs=1,random_state=50)
randomforest.fit(label_encoded_training_data,actual_target)

y_predicted = randomforest.predict(label_encoded_testing_data)


#RMSE Score are embedded with it after trying different models
label_encoded_prediction_LR = y_predicted #4664
label_encoded_prediction_DR = y_predicted #3282
label_encoded_prediction_RF = y_predicted #2910
OHE_prediction_LR = y_predicted #5833
OHE_prediction_DR = y_predicted #7779
OHE_prediction_RF = y_predicted #6955

#writing_to_submission_format

sample_submission_format['User_ID'] = testing_file['User_ID']
sample_submission_format['Product_ID'] = testing_file['Product_ID']
sample_submission_format['Purchase'] = label_encoded_prediction_RF
sample_submission_format.to_csv("Sample_Submission_Tm9Lura.csv")


dataset_description(training_file)

