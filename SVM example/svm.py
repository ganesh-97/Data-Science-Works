# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 15:29:12 2019

@author: expert
"""

##### CANCER DATASET #######

####### IMPORTING ALL NECESSARY LIBRARIES ######

from sklearn import datasets
from sklearn import svm
from sklearn.cross_validation import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np
#import plotly.plotly as py
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

##### LOADING CANCER DATASET FROM SKLEARN ######

cancer_dataset = datasets.load_breast_cancer()

##### VIEW THE FEATURE NAMES AND TARGET AND SHAPE

cancer_dataset.feature_names
cancer_dataset.target
cancer_dataset.data.shape

##### SPLITTING THE DATASETS TO TRAINING AND TESTING #####

X_train,X_test,Y_train,Y_test = train_test_split(cancer_dataset.data,cancer_dataset.target, test_size = 0.3, random_state = 7)

py.iplot(X_train)
##### STANDARDISE THE DATA SINCE PCA WORKS BEST IN STANDARDISED DATA #####

sc = StandardScaler()

X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

plt.hist(X_train,10)
##### APPLY PCA TO THE TRANSFORMED DATA #####

pca = PCA(n_components = 3)

X_train = pca.fit_transform(X_train)
X_test = pca.fit_transform(X_test)

plt.hist(X_train,10)

##### DISPLAY THE VARIANCE #####

explained_variance = pca.explained_variance_ratio_

##### CREATE A MODEL ( SVM ) AND FIT THE DATA #####

svm = svm.SVC(kernel ="linear")
svm.fit(X_train,Y_train)
len(svm.support_vectors_[:])

##### DISPLAY THE HYPERPLANE AND SVM PLOT #####

print(len(X_train[:,1]))

# get the separating hyperplane
w = svm.coef_[0]
a = -w[0] / w[1]
xx = np.linspace(-5, 5)
yy = a * xx - (svm.intercept_[0]) / w[1]

# plot the parallels to the separating hyperplane that pass through the
# support vectors
b = svm.support_vectors_[0]
yy_down = a * xx + (b[1] - a * b[0])
b = svm.support_vectors_[-1]
yy_up = a * xx + (b[1] - a * b[0])

# plot the line, the points, and the nearest vectors to the plane
plt.plot(xx, yy, 'k-')
plt.plot(xx, yy_down, 'k--')
plt.plot(xx, yy_up, 'k--')

plt.scatter(svm.support_vectors_[:, 0], svm.support_vectors_[:, 1],s=80, facecolors='none')
plt.scatter(X_train[:, 0], X_train[:, 1], c=Y_train, cmap=plt.cm.Paired)

plt.axis('tight')
plt.show()

##### CREATE A MODEL ( NAIVE BAYES ) AND FIT THE DATA #####

from sklearn.naive_bayes import GaussianNB

model = GaussianNB()
model.fit(X_train,Y_train)

##### PREDICT #####

y_pred = svm.predict(X_test)
from sklearn.metrics import confusion_matrix 
print(confusion_matrix(y_pred,Y_test))

##### VIEW ACCURACY SCORE #####

accuracy_score = metrics.accuracy_score(y_pred,Y_test)
