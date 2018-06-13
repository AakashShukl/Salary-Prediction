# -*- coding: utf-8 -*-
"""
Created on Tue Jun 12 23:40:35 2018

@author: AAKASH
"""
#libraries 
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#dataset
dataset=pd.read_csv('Data.csv')
#Object form has been converted to data frame
X1=pd.DataFrame(X)

#Dividing into matrix of features and independent variable
X=dataset.iloc[:,:-1].values
y=dataset.iloc[:,3]

#Handling missing data using Imputer class from libraries sklearn

from sklearn.preprocessing import Imputer
imputer=Imputer(missing_values="NaN",strategy="mean",axis=0)
imputer=imputer.fit(X[:,1:3])
X[:,1:3]=imputer.transform(X[:,1:3])

#Encoding into Categorical data

from sklearn.preprocessing import LabelEncoder
labelencoder_X=LabelEncoder()
X[:,0]=labelencoder_X.fit_transform(X[:,0])
labelencoder_Y=LabelEncoder()
y=labelencoder_Y.fit_transform(y)

#Creating a dummy variable because machine should not find any corelation between
from sklearn.preprocessing import OneHotEncoder
onehotencoder=OneHotEncoder(categorical_features=[0])
X=onehotencoder.fit_transform(X).toarray()


#dividing the dataset in trainingset and test set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)


#Feature Scaling
from sklearn.preprocessing import StandardScaler
standardScalar=StandardScaler()
X_train=standardScalar.fit_transform(X_train)
X_test=standardScalar.fit_transform(X_test)


#SalaryData

dataset=pd.read_csv("Salary_Data.csv")
X=dataset.iloc[:,:-1].values
y=dataset.iloc[:,1]
                                              

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

# Fitting Simple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)

#Visualising the train set result
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.savefig('train.png')
plt.show()

# Visualising the Test set results
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.savefig('test.png')
plt.show()