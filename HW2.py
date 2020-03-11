# Hypothesis: there is a reverse relationship between GDP per capita and unemployment rate.
# SL.UEM.TOTL.ZS = Unemployment rate in the world (Variable 1)
# NY.GDP.PCAP.PP.CD = GDP per capita in the world (Variable 2)
# For finding whether this hypothesis is true or not, we are going to make a regression analysis and find whether there is a linear relationship between two variables.

import wbdata
import datetime
import pandas as pd
import numpy as np
import regression
import math
import matplotlib.pyplot as pyplot
from sklearn.linear_model import LinearRegression

data_date = (datetime.datetime(2010, 1, 1), datetime.datetime(2015, 1, 1))
Variable_1 = wbdata.get_data("SL.UEM.TOTL.ZS", data_date = data_date, pandas = True)
Variable_2 = wbdata.get_data("NY.GDP.PCAP.PP.CD", data_date = data_date, pandas = True)

# We need to reshape Variable 1 and adjust Variable 1 and 2 for NaN values 

a = np.array(Variable_1).reshape(-1,1)
np.isnan(a)
np.where(np.isnan(a))
np.nan_to_num(a)
x = np.nan_to_num(a)

b = np.array(Variable_2)
np.isnan(b)
np.where(np.isnan(b))
np.nan_to_num(b)
y = np.nan_to_num(b)

# Solution 1 via writing the formulas by hand

x1 = np.matrix(x) # matrix of Variable 1
x1.I # Inverse of Variable 1
beta = x1.I @ x.T @ y
k = 1
n = np.size(x)
y_estimate = x @ beta
e = y.T - y_estimate
sigma_square = (e.T @ e) / (n - k - 1)
beta_Variance = sigma_square @ x1.I
std = math.sqrt(beta_Variance) # standard deviation
lower_bound = beta - 1.96 * std
upper_bound = beta + 1.96 * std

#  Solution 1: Scatter Plot Graph

pyplot.scatter(x,y, color = "b", marker = "o", s = 30)
pyplot.ylabel('GDP Per Capita (Current US$)')
pyplot.xlabel('Unemployment (% of total)')
pyplot.show()

# Solution 2 via Linear Regression function

model = LinearRegression()
model.fit(x,y)
r_sq = model.score(x,y)
print('coefficient of determination:',r_sq)
print('intercept:', model.intercept_)
print('slope:', model.coef_)

y_pred = model.predict(x)
print('predicted response:',ypred,sep = '/n')

# csv file writing

countries = [i['id'] for i in wbdata.get_country(incomelevel="HIC", display=False)]
indicators = {"SL.UEM.TOTL.ZS": "unemployment_rate", "NY.GDP.PCAP.PP.CD": "gdppc"}
df = wbdata.get_dataframe(indicators, country=countries, convert_date=True)
df.to_csv('hw2_csv')
df.describe()
