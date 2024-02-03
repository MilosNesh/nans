#   Iscrtavanje podataka o broju proizvedenih tona i zakljucivanje na osnovu grafika

import pandas as pd
import matplotlib.pyplot as plt

#   BiH
df = pd.read_excel('data/BiH.xlsx')
x = df['Date']
y = df['Metric Tons']
plt.plot(x, y, c='g', label='BiH')

#   Poljska 
dfp = pd.read_excel('data/Poland.xlsx')
xp = dfp['Date']
yp = dfp['Metric Tons']
plt.plot(xp, yp, label='Poland')

#   USA
dfus = pd.read_excel('data/US.xlsx')
x_us = dfus['Date']
y_us = dfus['Metric Tons']
plt.plot(x_us, y_us, c='r', label='USA')

#   Spanija
dfs = pd.read_excel('data/Spain.xlsx')
x_s = dfs['Date']
y_s = dfs['Metric Tons']
plt.plot(x_s, y_s, c='y', label='Spain')

#   Bulgaria
dfb = pd.read_excel('data/Bulgaria.xlsx')
x_b = dfb['Date']
y_b = dfb['Metric Tons']
plt.plot(x_b, y_b, c='purple', label='Bulgaria')

#   Canada
dfc = pd.read_excel('data/Canada.xlsx')
x_c = dfc['Date']
y_c = dfc['Metric Tons']
plt.plot(x_c, y_c, c='orange', label='Canada')

plt.legend()
plt.show()

#   Azerbejdzan 
dfa = pd.read_excel('data/Azerbaijan.xlsx')
x_a = dfa['Date']
y_a = dfa['Metric Tons']
plt.plot(x_a, y_a, c='orange', label='Azerbaijan')

#   Mexico
dfm = pd.read_excel('data/Mexico.xlsx')
x_m = dfm['Date']
y_m = dfm['Metric Tons']
plt.plot(x_m, y_m, c='purple', label='Mexico')

#   Ukrajina
dfuk = pd.read_excel('data/Ukraine.xlsx')
x_uk = dfuk['Date']
y_uk = dfuk['Metric Tons']
plt.plot(x_uk, y_uk, c='y', label='Ukraine')

plt.legend()
plt.show()

#   Srbija              popalve 2014.

dfsr = pd.read_excel('data/Raspberries Production in Serbia.xlsx')
x_sr = dfsr['Date'] 
y_sr = dfsr['Metric Tons']
plt.plot(x_sr, y_sr, c='r', label='Serbia')

#   Cile
# Cile cunami 2014.
dfc = pd.read_excel('data/Raspberries Production in Chile.xlsx')
x_c = dfc['Date']
y_c = dfc['Metric Tons']
plt.plot(x_c, y_c, c='y', label='Chile')

#   Poljska 
dfp = pd.read_excel('data/Poland.xlsx')
xp = dfp['Date']
yp = dfp['Metric Tons']
plt.plot(xp, yp, label='Poland')

plt.legend()
plt.show()

#https://www.b92.net/biz/vesti/srbija/malina-iz-cilea-je-u-stvari-poljska-816271   clanak 

#   Mexico
dfm = pd.read_excel('data/Mexico.xlsx')
x_m = dfm['Date']
y_m = dfm['Metric Tons']
plt.plot(x_m, y_m, c='purple', label='Mexico')
plt.legend()
plt.show()