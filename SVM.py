#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 21:42:50 2019

@author: lakshyasharma
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('train.csv')
X = dataset.iloc[:, [2, 4,5,6,7,9]].values
y = dataset.iloc[:,1].values

# Splitting the dataset into the Training set and Test set
X_train=X
y_train=y
dataset_2=pd.read_csv('test.csv')
X_test=dataset_2.iloc[:,[1,3,4,5,6,8]].values
dataset_3=pd.read_csv('gender_submission.csv')
y_test=dataset_3.iloc[:,1]

# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder
labelencoder_X = LabelEncoder()
X_train[:, 1] = labelencoder_X.fit_transform(X_train[:, 1])

#X test
from sklearn.preprocessing import LabelEncoder
labelencoder_X = LabelEncoder()
X_test[:, 1] = labelencoder_X.fit_transform(X_test[:, 1])

# Taking care of missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X_train[:, 0:7])
X_train[:,0:7] = imputer.transform(X_train[:,0:7])

#X test
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X_test[:, 0:7])
X_test[:,0:7] = imputer.transform(X_test[:,0:7])
#y train
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(y_train)
y_train = imputer.transform(y_train)

#y test
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(y_test)
y_test = imputer.transform(y_test)



# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting SVM to the Training set
from sklearn.svm import SVC
classifier =SVC(kernel='linear')
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
acc = accuracy_score(y_test, y_pred)

# Print the accuracy and the confusion matrix
print("the confusion matrix is\n", cm)
print("the test accuracy is \n", acc)
