# -*- coding: utf-8 -*-
"""
Created on Thu Dec 20 11:50:29 2018

@author: KIIT
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values

#Categorical data 
from sklearn.preprocessing import LabelEncoder , OneHotEncoder
labelencoder_x = LabelEncoder()
X[:, 3]=labelencoder_x.fit_transform(X[:, 3])
onehotencoder=OneHotEncoder(categorical_features=[3])
X=onehotencoder.fit_transform(X).toarray()

# To AVoid Dummy Variable Trap
X=X[:,1:]
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/5, random_state = 0)

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

# Apply Multiple Linear Regression
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,y_train)

# Prdicting the test set results
y_pred=regressor.predict(X_test)

# Building the optimal model using Backward elimination
"""Now we will remove less significantly data that has less impact on the results"""
"""We will form a team of a independent variables  which have a great impact on a results"""
# Preparing Backward elimination
import statsmodels.formula.api as sm
# Statsmodels use to calculate p values
X = np.append(arr=np.ones((50,1)).astype(int),values=X,axis=1)
X_opt=X[:,[0,1,2,3,4,5]]
regressor_OLS = sm.OLS(endog=y,exog=X_opt).fit()#Ordinary Least Squares
regressor_OLS.summary()
# Removing 2 
X_opt=X[:,[0,1,3,4,5]]
regressor_OLS = sm.OLS(endog=y,exog=X_opt).fit()#Ordinary Least Squares
regressor_OLS.summary()
# Removing 1
X_opt=X[:,[0,3,4,5]]
regressor_OLS = sm.OLS(endog=y,exog=X_opt).fit()#Ordinary Least Squares
regressor_OLS.summary()

# Removing 4
X_opt=X[:,[0,3,5]]
regressor_OLS = sm.OLS(endog=y,exog=X_opt).fit()#Ordinary Least Squares
regressor_OLS.summary()

# Removing 5  bcoz its p value is .06
X_opt=X[:,[0,3]]
regressor_OLS = sm.OLS(endog=y,exog=X_opt).fit()#Ordinary Least Squares
regressor_OLS.summary()


