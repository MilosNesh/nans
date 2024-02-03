import pandas as pd
import seaborn as sb
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from utils_nans1 import *

df = pd.read_csv('data/raspberry.csv')

cols = ['Hectograms Per Hectare', 'Hectares', 'Thousand US Dollars PPP = 2004–2006']
x = df[cols]
y = df['Metric Tons']
       
x_train, x_val, y_train, y_val = train_test_split(x, y, train_size=0.7, shuffle=True, random_state=42)

model = get_fitted_model(x_train, y_train)

l = linear_assumption(model, sm.add_constant(x), y, plot= True)
print("Linearnost: ",l)
i = independence_of_errors_assumption(model, sm.add_constant(x),y, plot= True)
print("Independens: ",i)
n = normality_of_errors_assumption(model, sm.add_constant(x), y, plot= True)
print("Normality", n)
e = equal_variance_assumption(model, sm.add_constant(x), y, plot= True)
print("Equal", e)
print(model.summary())
r2 = get_rsquared_adj(model, x_val, y_val)
print(r2)