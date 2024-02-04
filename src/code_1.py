import pandas as pd
import seaborn as sb
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import utils_nans1 as un
from sklearn.model_selection import train_test_split
from utils_nans1 import *
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Ucitavanje podataka

df = pd.read_csv('data/raspberrybd.csv', sep=',')
print(df.head)

#   * Thousand US Dollars PPP = 2004–2006,
#   * Hectograms Per Hectare,
#   * Hectares,
#   * Metric Tons, 
#   * US Dollars Per Metric Ton

cols = ['Hectograms Per Hectare', 'Thousand US Dollars PPP = 2004–2006', 'Hectares', 'Metric Tons']
x = df[cols]
y = df['US Dollars Per Metric Ton']

# Matrica korelacije 
correlation_matrix = df.corr()
sb.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.1)
plt.title('Matrica korelacije')
plt.show()

###   Linearna regresija   ###  

x_train, x_val, y_train, y_val = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=42)

model = get_fitted_model(x_train, y_train)

# Testiranje LINE pretpostavki

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

y_pred = model.predict(sm.add_constant(x_val, has_constant='add'))

mse = mean_squared_error(y_val, y_pred)
print(f"Mean Squared Error(LinearRegressor): {mse}")     
     

###   RandomForestRegressor   ### 

X_train, X_test, y_train, y_test = train_test_split(x, y, shuffle=True,train_size=0.8, random_state=42)

# Inicijalizacija i treniranje RandomForestRegressor modela
rf_model = RandomForestRegressor(max_depth= 5, min_samples_leaf= 1, min_samples_split= 2, n_estimators= 100, random_state=42)
rf_model.fit(X_train, y_train)

# Predviđanje na test skupu 
y_pred = rf_model.predict(X_test)

# Evaluacija modela
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error(RandomForestRegressor): {mse}")


###   Neuronska mreza   ###

X_train, X_test, y_train, y_test = train_test_split(x, y, shuffle=True,train_size=0.8, random_state=42)

# Normalizacija podataka
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Kreiranje i obuka modela
model = MLPRegressor(activation='relu',
                    hidden_layer_sizes=(100, 50),
                    learning_rate_init=0.009,
                    max_iter=1500, random_state=42)

model.fit(X_train, y_train.ravel())

# Predviđanje na test skupu 
y_pred = model.predict(X_test)

# Evaluacija modela
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error on Test Data: {mse}")


###   HuberRegressor   ###

from sklearn.linear_model import HuberRegressor

X_train, X_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=42)

from sklearn.model_selection import cross_val_score
huber_reg = HuberRegressor()

# Unakrsna validacija sa neg_mean_squared_error kao metrikom
# scoring='neg_mean_squared_error' znači da ćemo dobiti negativne vrednosti MSE
scores = cross_val_score(huber_reg, x, y.ravel(), scoring='neg_mean_squared_error', cv=5)

# Dobijanje vrednosti parametra epsilon iz HuberRegressor modela
epsilon_value = huber_reg.epsilon

print(f"Vrednost epsilon parametra: {epsilon_value}")

# Inicijalizacija i treniranje robusnog linearnog regresora
huber_reg = HuberRegressor(epsilon = epsilon_value)
huber_reg.fit(X_train, y_train)

# Predikcija na test skupu
y_pred = huber_reg.predict(X_test)

# Evaluacija modela
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error(HuberRegressor): {mse}')


# Proba podaci za Cile
x_chile = np.array([[40669, 36503.896415, 4810, 19562]])
#x_chile = np.array([[40669, 36503.896415]])
y_pravo_chile = 3638.7

y_pred_chile = rf_model.predict(x_chile)
print(f"(RF) Stvarna cijena maline u Chileu 2018. god je {y_pravo_chile}, dok je nasa predikcija {y_pred_chile}")

# y_pred_chile = huber_reg.predict(x_chile)
# # print(f"(HR)Stvarna cijena maline u Chileu 2018. god je {y_pravo_chile}, dok je nasa predikcija {y_pred_chile}")

scaler = StandardScaler()
x_chile = scaler.fit_transform(x_chile)
y_pred_chile = model.predict(x_chile)
print(f"(NN)Stvarna cijena maline u Chileu 2018. god je {y_pravo_chile}, dok je nasa predikcija {y_pred_chile}") 
print()

# Cile 2012
x_chile = np.array([[38661,33890.864673, 4473, 17293]])
#
#x_chile = np.array([[38661,33890.864673]])
y_pravo_chile = 4158.9

y_pred_chile = rf_model.predict(x_chile)
print(f"(RF)Stvarna cijena maline u Chileu 2012. god je {y_pravo_chile}, dok je nasa predikcija {y_pred_chile}")

# y_pred_chile = huber_reg.predict(x_chile)
# print(f"Stvarna cijena maline u Chileu 2012. god je {y_pravo_chile}, dok je nasa predikcija {y_pred_chile}")

scaler = StandardScaler()
x_chile = scaler.fit_transform(x_chile)
y_pred_chile = model.predict(x_chile)
print(f"(NN)Stvarna cijena maline u Chileu 2012. god je {y_pravo_chile}, dok je nasa predikcija {y_pred_chile}")
print()

# Cile 2015
x_chile = np.array([[39539, 35793.556008, 4625, 18287]])
#
#x_chile = np.array([[39539, 18287]])
y_pravo_chile = 4672.4

y_pred_chile = rf_model.predict(x_chile)
print(f"(RF)Stvarna cijena maline u Chileu 2015. god je {y_pravo_chile}, dok je nasa predikcija {y_pred_chile}")

# y_pred_chile = huber_reg.predict(x_chile)
# print(f"Stvarna cijena maline u Chileu 2012. god je {y_pravo_chile}, dok je nasa predikcija {y_pred_chile}")

scaler = StandardScaler()
x_chile = scaler.fit_transform(x_chile)
y_pred_chile = model.predict(x_chile)
print(f"(NN)Stvarna cijena maline u Chileu 2015. god je {y_pravo_chile}, dok je nasa predikcija {y_pred_chile}")

