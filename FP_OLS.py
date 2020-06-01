# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 21:34:47 2020

@author: user
"""

# OLS Regression:

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