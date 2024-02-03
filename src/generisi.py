import pandas as pd
import seaborn as sb
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import utils_nans1 as un
from sklearn.model_selection import train_test_split
from utils_nans1 import *
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error as mse
#BiH
df = pd.read_excel('data/Raspberries Producer Price in Bosnia and Herzegovina.xlsx')
# df = pd.read_csv('data/Raspberries Price BiH.csv', sep=',')
df = df.set_index('Date')
df.plot()
plt.show()

plot_acf(df['US Dollars Per Metric Ton'])
plt.show()

plot_pacf(df['US Dollars Per Metric Ton'], lags=9)
plt.show()

from statsmodels.tsa.seasonal import seasonal_decompose

result = seasonal_decompose(df['US Dollars Per Metric Ton'], model='additive', period=1)
result.plot()
plt.show()

df['log(Price)'] = np.log10(df['US Dollars Per Metric Ton'])

pv = adfuller(df['log(Price)'])[1]
print("p vslues: ", pv)

stac = df['log(Price)'].diff().diff().dropna()
pv = adfuller(stac)[1]
print("pv2: ", pv)
plt.plot(stac)
plt.show()

#           Predikcija...

tp = int(len(df['US Dollars Per Metric Ton'])*0.8)
train = df[:tp].copy()
valid = df[tp:].copy()

train['log(Price)'] = np.log10(train['US Dollars Per Metric Ton'])
train['stacionary'] = train['log(Price)'].diff()

pv = adfuller(train['stacionary'].dropna())[1]
if pv < 0.05:
    print("Postoji stacionarnost")
else:
    print("Ne postoji stacionarnost. P vrijednost iznosi ", pv)

plot_acf(train['stacionary'].dropna(), lags=6)
plt.show()

plot_pacf(train['stacionary'].dropna(), lags=6, method='ols')
plt.show()

p, d, q = 3, 1, 0
ar_model = ARIMA(train['log(Price)'], order=(p, d, 0)).fit()
print(ar_model.summary())

y_train_pred = ar_model.predict(start=train.index[p+1], end=train.index[-1])

plt.plot(train['log(Price)'], color='b', linewidth=4, alpha=0.3, label='train')
plt.plot(y_train_pred, color='darkorange', label='AR model prediction')
plt.title('predikcije za log10(Price)')
plt.legend()
plt.show()

valid['log(Price)'] = np.log10(valid['US Dollars Per Metric Ton'])

y_valid_pred = ar_model.predict(start=valid.index[0], end=valid.index[-1])

plt.plot(train['log(Price)'], color='b', linewidth=4, alpha=0.3, label='train')
plt.plot(valid['log(Price)'], color='mediumblue', linewidth=4, alpha=0.3, label='val')

plt.plot(y_valid_pred, color='darkorange', label='AR model prediction')
plt.title('predikcije za log(Price)')
plt.legend()
plt.show()

y_pred = ar_model.predict(start=train.index[p+1], end=valid.index[-1])
y_pred = np.power(10, y_pred)

plt.plot(train['US Dollars Per Metric Ton'], color='b', linewidth=4, alpha=0.3, label='train')
plt.plot(valid['US Dollars Per Metric Ton'], color='mediumblue', linewidth=4, alpha=0.3, label='val')

plt.plot(y_pred, color='darkorange', label='AR model prediction')
plt.title('predikcije za log(Price)')
plt.legend()
plt.show()

y_valid_pred = np.power(10, y_valid_pred)

print("Mean Sqared Error: ", mse(valid['US Dollars Per Metric Ton'], y_valid_pred)) #372234.884

# df['log(Price)'] = np.log10(df['US Dollars Per Metric Ton'])
# stl = STL(df['log(Price)']).fit()
# stl.plot()
# plt.show()

