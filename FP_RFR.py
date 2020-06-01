# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 21:37:18 2020

@author: user
"""

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
print(r_sq)

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