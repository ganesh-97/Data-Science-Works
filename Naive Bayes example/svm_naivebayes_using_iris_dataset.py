# -*- coding: utf-8 -*-
"""
Created on Tue Feb 26 11:10:41 2019

@author: expert
"""

#IRIS DATASET , used SVM and NAIVE BAYES

##### IMPORT ALL NECESSARY PACKAGES

import pandas
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

##### READ IRIS DATASET #####

iris_dataset = datasets.load_iris()

##### DISPLAY ITS FEATURES AND TARGET #####

iris_dataset.target
iris_dataset.feature_names
iris_dataset.target_names
iris_dataset.data.shape

##### CREATING A DATAFRAME AND CHECK FOR MISSING VALUES #####

iris_dataframe = pandas.DataFrame(data = iris_dataset.data, columns = iris_dataset.feature_names)
iris_dataframe['Class'] = iris_dataset.target
iris_dataframe.tail(10)

iris_dataframe.isnull().sum()

##### FIND CORRELATION #####

correlated_data = iris_dataframe.corr(method = "pearson")

##### SPLIT TRAINING AND TESTING DATA #####

X_train,X_test,Y_train,Y_test = train_test_split(iris_dataset.data, iris_dataset.target,test_size = 0.2,random_state = 7)

##### STANDARDISE THE TRAINING DATA #####

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

##### APPLY PCA #####

pca = PCA()
X_train = pca.fit_transform(X_train)
X_test = pca.fit_transform(X_test)

##### CREARTE A SVM MODEL ######

svm_clf = svm.SVC(kernel ="linear")
svm_clf.fit(X_train,Y_train)

##### CREATE NAIVE MODEL #####

naive_model = GaussianNB()
naive_model.fit(X_train,Y_train)

##### PREDICTING USING SVM MODEL ######

Y_pred = svm_clf.predict(X_test)
print(confusion_matrix(Y_pred,Y_test))
print(accuracy_score(Y_pred,Y_test))
print(classification_report(Y_pred,Y_test))

##### PREDICTING USING NAIVE BAYES MODEL #####

Y_pred = naive_model.predict(X_test)
print(confusion_matrix(Y_pred,Y_test))
print(accuracy_score(Y_pred,Y_test))
print(classification_report(Y_pred,Y_test))
