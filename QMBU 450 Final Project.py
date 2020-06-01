# -*- coding: utf-8 -*-
"""
Created on Sun May 10 01:57:41 2020

@author: user
"""
# Explanatory Variables: World Governance Indicators (WGI)
# VA: Voice and Accountability
# CoC: Control of Corruption
# GE: Government Effectiveness
# PSNV: Political Stability No Violence
# RQ: Regulatory Quality
# RoL: Rule of Law

# Dependent Variable:
# GINI: GINI Index, measures inequality with a range from 0 to 100. Higher the index, higher the inequality.

# Read data by using pandas
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Model 1 (with 160 observations)

file='C:/Users/user/Documents/Final_Project_Data.xlsx'
x1 = pd.ExcelFile(file)
print(x1.sheet_names)
df1 = x1.parse('Sheet1')

# Look at the data
df1.head(161)

# Split the data into dependent and explanatory variables

#Create dataset from the selected column
explanatory = df1[['VA', 'CoC', 'GE', 'PSNV', 'RQ','RoL']]

# Create dataset from the selected column
dependent = df1[['GINI']]

# For doing Principal Component Analysis

# Standartization
from sklearn.preprocessing import StandardScaler
explanatory_std = StandardScaler().fit_transform(explanatory)

# look at your standartized matrix
explanatory_std

# For creating the covariance matrix:

# compute your covarianca matrix by using numpy library
columns = explanatory_std.T 
matrix = np.cov(columns)

# Look at your matrix
print(matrix)

# # Compute eigenvectors and eigenvalues from Covariance Matrix

# compute eigen values and vectors
eig_vals, eig_vecs = np.linalg.eig(matrix)

# print your computation
print('Eigenvectors \n%s' %eig_vecs)
print('\nEigenvalues \n%s' %eig_vals)

# compute percentage of variances that explained by our values
eig_vals[0] / sum(eig_vals)

# compute percentage of variances that explained by our values
eig_vals[1] / sum(eig_vals)

# compute percentage of variances that explained by our values
eig_vals[2] / sum(eig_vals)

# select your first pc
pca = explanatory_std.dot(eig_vecs.T[0])

# look at the array
pca

# Create dataset by using component
pcamain = pd.DataFrame(pca, columns=['PC1'])

# Add your main dependent variable to have both component and dependent variable in one data
pcamain['GINI'] = dependent

# Look at the data
pcamain.head(10)

# PCA in short
from  sklearn  import  decomposition
pca = decomposition.PCA(n_components=2)
pcashort = pca.fit_transform(explanatory_std)

# Create dataset from matrix by using pandas
pcashort2 = pd.DataFrame(pcashort, columns=['PC1','PC2'])

# Look at the data
pcashort2

# Add the dependent into the dataset
finalDf1 = pd.concat([pcashort2, dependent], axis = 1)

# look at the data
finalDf1

# Run regression to see the importance of the principal component (PC1)
import statsmodels.formula.api as sm
regression = sm.ols(formula="GINI ~ PC1", data=finalDf1).fit()
print(regression.summary())

# Run regression to see the importance of the principal component (the six WGI variables)
import statsmodels.formula.api as sm
regression = sm.ols(formula="GINI ~ VA + CoC+GE+PSNV+RQ+RoL", data=df1).fit()
print(regression.summary())

# To create scatter plot:

x= pcamain[['PC1']]
y= pcamain[['GINI']]
plt.scatter(x,y, color = "b", marker = "o", s = 30)
plt.ylabel('GINI')
plt.xlabel('PC1')
plt.show()

# Random Forest Regression (with PC1)

# define the x and y values

x = pcamain.iloc[:, 0:1].values  
print(x) 
y = pcamain.iloc[:, 1].values
print(y)

# Fitting Random Forest Regression to the dataset 
# import the regressor 
from sklearn.ensemble import RandomForestRegressor 

# create regressor object 
regressor = RandomForestRegressor(n_estimators = 160, random_state = 0) 

# fit the regressor with x and y data 
regressor.fit(x, y)
# Find R2 value of the model
r_sq = regressor.score(x,y)
print('R2:', r_sq)

# test the output by changing values
Y_pred = regressor.predict(np.array([6.5]).reshape(1, 1))  

# Visualising the Random Forest Regression results 

X_grid = np.arange(min(x), max(x), 0.01)  

# reshape for reshaping the data into a len(X_grid)*1 array,  
# i.e. to make a column out of the X_grid value                   
X_grid = X_grid.reshape((len(X_grid), 1)) 

# Scatter plot for original data 
plt.scatter(x, y, color = 'blue')   

# plot predicted data 
plt.plot(X_grid, regressor.predict(X_grid),  
         color = 'green')  
plt.title('Random Forest Regression') 
plt.xlabel('PC1') 
plt.ylabel('GINI') 
plt.show()

# For the training and test split:

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size= 48, random_state=0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.ensemble import RandomForestRegressor

regressor = RandomForestRegressor(n_estimators=160, random_state=0)
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)

# For MAE, MSE, and RMSE values

from sklearn import metrics

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

# For R2 values
r_sq_train = regressor.score(X_train, y_train)
print('R2 (Training Data):', r_sq_train)
r_sq_test = regressor.score(X_test, y_test)
print('R2 (Test Data):', r_sq_test)

# Model 2 (with 56 observations)

# Note: For running the training and test data, please run these two files seperately:
# file='C:/Users/user/Documents/FP_3_Train.xlsx'
# file='C:/Users/user/Documents/FP_3_Test.xlsx'

file='C:/Users/user/Documents/FP_3.xlsx'
x1 = pd.ExcelFile(file)
print(x1.sheet_names)
df1 = x1.parse('Sheet1')

# Look at the data
df1.head(56)

# Split the data into dependent and explanatory variables

#Create dataset from the selected column
explanatory = df1[['VA', 'CoC', 'GE', 'PSNV', 'RQ','RoL']]

# Create dataset from the selected column
dependent = df1[['GINI']]

# For doing Principal Component Analysis

# Standartization
from sklearn.preprocessing import StandardScaler
explanatory_std = StandardScaler().fit_transform(explanatory)

# look at your standartized matrix
explanatory_std

# For creating the covariance matrix:

# compute your covarianca matrix by using numpy library
columns = explanatory_std.T 
matrix = np.cov(columns)

# Look at your matrix
print(matrix)

# # Compute eigenvectors and eigenvalues from Covariance Matrix

# compute eigen values and vectors
eig_vals, eig_vecs = np.linalg.eig(matrix)

# print your computation
print('Eigenvectors \n%s' %eig_vecs)
print('\nEigenvalues \n%s' %eig_vals)

# compute percentage of variances that explained by our values
eig_vals[0] / sum(eig_vals)

# compute percentage of variances that explained by our values
eig_vals[1] / sum(eig_vals)

# compute percentage of variances that explained by our values
eig_vals[2] / sum(eig_vals)

# select your first pc
pca = explanatory_std.dot(eig_vecs.T[0])

# look at the array
pca

# Create dataset by using component
pcamain = pd.DataFrame(pca, columns=['PC1'])

# Add your main dependent variable to have both component and dependent variable in one data
pcamain['GINI'] = dependent

# Look at the data
pcamain.head(10)

# PCA in short
from  sklearn  import  decomposition
pca = decomposition.PCA(n_components=2)
pcashort = pca.fit_transform(explanatory_std)

# Create dataset from matrix by using pandas
pcashort2 = pd.DataFrame(pcashort, columns=['PC1','PC2'])

# Look at the data
pcashort2

# Add the dependent into the dataset
finalDf1 = pd.concat([pcashort2, dependent], axis = 1)

# look at the data
finalDf1

# Run regression to see the importance of the principal component (PC1)
import statsmodels.formula.api as sm
regression = sm.ols(formula="GINI ~ PC1", data=finalDf1).fit()
print(regression.summary())

# Run regression to see the importance of the principal component (the six WGI variables)
import statsmodels.formula.api as sm
regression = sm.ols(formula="GINI ~ VA + CoC+GE+PSNV+RQ+RoL", data=df1).fit()
print(regression.summary())

# To create scatter plot:

x= pcamain[['PC1']]
y= pcamain[['GINI']]
plt.scatter(x,y, color = "b", marker = "o", s = 30)
plt.ylabel('GINI')
plt.xlabel('PC1')
plt.show()

# Random Forest Regression (with PC1)

# define the x and y values

x = pcamain.iloc[:, 0:1].values  
print(x) 
y = pcamain.iloc[:, 1].values
print(y)

# Fitting Random Forest Regression to the dataset 
# import the regressor 
from sklearn.ensemble import RandomForestRegressor 

# create regressor object 
regressor = RandomForestRegressor(n_estimators = 55, random_state = 0) 

# fit the regressor with x and y data 
regressor.fit(x, y)
# Find R2 value of the model
r_sq = regressor.score(x,y)
print('R2:', r_sq)

# test the output by changing values
Y_pred = regressor.predict(np.array([6.5]).reshape(1, 1))  

# Visualising the Random Forest Regression results 

X_grid = np.arange(min(x), max(x), 0.01)  

# reshape for reshaping the data into a len(X_grid)*1 array,  
# i.e. to make a column out of the X_grid value                   
X_grid = X_grid.reshape((len(X_grid), 1)) 

# Scatter plot for original data 
plt.scatter(x, y, color = 'blue')   

# plot predicted data 
plt.plot(X_grid, regressor.predict(X_grid),  
         color = 'green')  
plt.title('Random Forest Regression') 
plt.xlabel('PC1') 
plt.ylabel('GINI') 
plt.show()

# For the training and test split:

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size= 16, random_state=0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.ensemble import RandomForestRegressor

regressor = RandomForestRegressor(n_estimators=55, random_state=0)
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)

# For MAE, MSE, and RMSE values

from sklearn import metrics

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

# For R2 values
r_sq_train = regressor.score(X_train, y_train)
print('R2 (Training Data):', r_sq_train)
r_sq_test = regressor.score(X_test, y_test)
print('R2 (Test Data):', r_sq_test)

# Model 3 (With 8 Countries)

# Files for the countries:
# file='C:/Users/user/Documents/FP_Moldova.xlsx'
# file='C:/Users/user/Documents/FP_CostaRica.xlsx'
# file='C:/Users/user/Documents/FP_Honduras.xlsx'
# file='C:/Users/user/Documents/FP_ElSalvador.xlsx'
# file='C:/Users/user/Documents/FP_Denmark.xlsx'
# file='C:/Users/user/Documents/FP_Finland.xlsx'
# file='C:/Users/user/Documents/FP_Belarus.xlsx'
# file='C:/Users/user/Documents/FP_Indonesia.xlsx'

# The model below is an example:

file='C:/Users/user/Documents/FP_Moldova.xlsx'
x1 = pd.ExcelFile(file)
print(x1.sheet_names)
df1 = x1.parse('Sheet1')

# Look at the data
df1.head(17)

# Split the data into dependent and explanatory variables

#Create dataset from the selected column
explanatory = df1[['VA', 'CoC', 'GE', 'PSNV', 'RQ','RoL']]

# Create dataset from the selected column
dependent = df1[['GINI']]

# For doing Principal Component Analysis

# Standartization
from sklearn.preprocessing import StandardScaler
explanatory_std = StandardScaler().fit_transform(explanatory)

# look at your standartized matrix
explanatory_std

# For creating the covariance matrix:

# compute your covarianca matrix by using numpy library
columns = explanatory_std.T 
matrix = np.cov(columns)

# Look at your matrix
print(matrix)

# # Compute eigenvectors and eigenvalues from Covariance Matrix

# compute eigen values and vectors
eig_vals, eig_vecs = np.linalg.eig(matrix)

# print your computation
print('Eigenvectors \n%s' %eig_vecs)
print('\nEigenvalues \n%s' %eig_vals)

# compute percentage of variances that explained by our values
eig_vals[0] / sum(eig_vals)

# compute percentage of variances that explained by our values
eig_vals[1] / sum(eig_vals)

# compute percentage of variances that explained by our values
eig_vals[2] / sum(eig_vals)

# select your first pc
pca = explanatory_std.dot(eig_vecs.T[0])

# look at the array
pca

# Create dataset by using component
pcamain = pd.DataFrame(pca, columns=['PC1'])

# Add your main dependent variable to have both component and dependent variable in one data
pcamain['GINI'] = dependent

# Look at the data
pcamain.head(10)

# PCA in short
from  sklearn  import  decomposition
pca = decomposition.PCA(n_components=2)
pcashort = pca.fit_transform(explanatory_std)

# Create dataset from matrix by using pandas
pcashort2 = pd.DataFrame(pcashort, columns=['PC1','PC2'])

# Look at the data
pcashort2

# Add the dependent into the dataset
finalDf1 = pd.concat([pcashort2, dependent], axis = 1)

# look at the data
finalDf1

# Run regression to see the importance of the principal component (PC1)
import statsmodels.formula.api as sm
regression = sm.ols(formula="GINI ~ PC1", data=finalDf1).fit()
print(regression.summary())

# Run regression to see the importance of the principal component (the six WGI variables)
import statsmodels.formula.api as sm
regression = sm.ols(formula="GINI ~ VA + CoC+GE+PSNV+RQ+RoL", data=df1).fit()
print(regression.summary())

# To create scatter plot:

x= pcamain[['PC1']]
y= pcamain[['GINI']]
plt.scatter(x,y, color = "b", marker = "o", s = 30)
plt.ylabel('GINI')
plt.xlabel('PC1')
plt.show()

# Random Forest Regression (with PC1)

# define the x and y values

x = pcamain.iloc[:, 0:1].values  
print(x) 
y = pcamain.iloc[:, 1].values
print(y)

# Fitting Random Forest Regression to the dataset 
# import the regressor 
from sklearn.ensemble import RandomForestRegressor 

# create regressor object 
regressor = RandomForestRegressor(n_estimators = 17, random_state = 0) 

# fit the regressor with x and y data 
regressor.fit(x, y)
# Find R2 value of the model
r_sq = regressor.score(x,y)
print('R2:', r_sq)

# test the output by changing values
Y_pred = regressor.predict(np.array([6.5]).reshape(1, 1))  

# Visualising the Random Forest Regression results 

X_grid = np.arange(min(x), max(x), 0.01)  

# reshape for reshaping the data into a len(X_grid)*1 array,  
# i.e. to make a column out of the X_grid value                   
X_grid = X_grid.reshape((len(X_grid), 1)) 

# Scatter plot for original data 
plt.scatter(x, y, color = 'black')   

# plot predicted data 
plt.plot(X_grid, regressor.predict(X_grid),  
         color = 'yellow')  
plt.title('Random Forest Regression') 
plt.xlabel('PC1') 
plt.ylabel('GINI') 
plt.show()