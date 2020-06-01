# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 21:31:10 2020

@author: user
"""

# Principal Component Analysis (PCA):

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