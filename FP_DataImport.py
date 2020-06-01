# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 21:23:25 2020

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
df1 = x1.parse('Model_1')

# Look at the data
df1.head(161)