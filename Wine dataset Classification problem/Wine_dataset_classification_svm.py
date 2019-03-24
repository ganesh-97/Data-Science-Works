# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 12:08:45 2019

@author: expert
"""

############################ WINE DATASET ###################################

#importing packages
import pandas
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn import svm
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

#reading the data
white_wine_data = pandas.read_csv("winequality-white.csv", sep=";")
red_wine_data = pandas.read_csv("winequality-red.csv",sep=";")

#concatinating white and red wine data
complete_data = pandas.concat([white_wine_data,red_wine_data], ignore_index=True)
training_data_features = complete_data.loc[:,:'alcohol']
target_variable = complete_data['quality']

#no of rows and columns
complete_data.shape

#information about the data
complete_data.info()

#UNIVARIATE ANALYSIS
#It is found that there are only numerical variables
#'fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
#'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
#'pH', 'sulphates', 'alcohol', 'quality'

complete_data.select_dtypes(include = ['int','float']).columns

#check for missing values
columns = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar','chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density','pH', 'sulphates', 'alcohol', 'quality']
for columns_name in columns:
    print(complete_data[columns_name].value_counts().isnull().sum())

#checking for outliers
complete_data['volatile acidity'].value_counts().plot.box(title = 'volatile acidity')

#finding correlation
correlated_data = complete_data.corr()

#splitting training and testing data
X_train,X_test,Y_train,Y_test = train_test_split(training_data_features, target_variable, test_size =0.3,random_state = 7)

#Standardising the data
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

#applying PCA
pca = PCA()
X_train = pca.fit_transform(X_train)
X_test =pca.fit_transform(X_test)

#svm model building
svm_classifier = svm.SVC(kernel="linear")
svm_classifier.fit(X_train,Y_train)

#Randomforest Classifier
randomclassifier = RandomForestClassifier()
randomclassifier.fit(X_train,Y_train)

#Naive bayes
naive = GaussianNB()
naive.fit(X_train,Y_train)

#randomforest classifier prediction
Y_predicted = randomclassifier.predict(X_test)

#svm prediction
Y_predicted = svm_classifier.predict(X_test)

#naive prediction
Y_predicted = naive.predict(X_test)

#Evaluation
print(confusion_matrix(Y_predicted,Y_test))
print(accuracy_score(Y_predicted,Y_test))
print(classification_report(Y_predicted,Y_test))
