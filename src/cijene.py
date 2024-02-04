# ovde iscrtavam grafike cijena maline i poredim ih da bi dosao do odredjenjih zakljucka

import pandas as pd
import matplotlib.pyplot as plt

#   BiH
df = pd.read_excel('data/BiH.xlsx')
x = df['Date']
y = df['US Dollars Per Metric Ton']
plt.plot(x, y, c='g', label='BiH')

#   Poljska 
dfp = pd.read_excel('data/Poland.xlsx')
xp = dfp['Date']
yp = dfp['US Dollars Per Metric Ton']
plt.plot(xp, yp, label='Poland')

#   USA
dfus = pd.read_excel('data/US.xlsx')
x_us = dfus['Date']
y_us = dfus['US Dollars Per Metric Ton']
plt.plot(x_us, y_us, c='r', label='USA')

#   Spanija
dfs = pd.read_excel('data/Spain.xlsx')
x_s = dfs['Date']
y_s = dfs['US Dollars Per Metric Ton']
plt.plot(x_s, y_s, c='y', label='Spain')

#   Bulgaria
dfb = pd.read_excel('data/Bulgaria.xlsx')
x_b = dfb['Date']
y_b = dfb['US Dollars Per Metric Ton']
plt.plot(x_b, y_b, c='purple', label='Bulgaria')

#   Canada
dfc = pd.read_excel('data/Canada.xlsx')
x_c = dfc['Date']
y_c = dfc['US Dollars Per Metric Ton']
plt.plot(x_c, y_c, c='orange', label='Canada')

plt.legend()
plt.savefig('image/2008Price.png')
plt.show()

#   Gore navedenih 6 drzava imaju istu najvecu cijenu maline u istoj godini i to u 2008.
#   2008. godine velika recesija te to moze biti razlog za skok u cijeni maline

#   Azerbejdzan 
dfa = pd.read_excel('data/Azerbaijan.xlsx')
x_a = dfa['Date']
y_a = dfa['US Dollars Per Metric Ton']
plt.plot(x_a, y_a, c='orange', label='Azerbaijan')

#   Mexico
dfm = pd.read_excel('data/Mexico.xlsx')
x_m = dfm['Date']
y_m = dfm['US Dollars Per Metric Ton']
plt.plot(x_m, y_m, c='purple', label='Mexico')

#   Ukrajina
dfuk = pd.read_excel('data/Ukraine.xlsx')
x_uk = dfuk['Date']
y_uk = dfuk['US Dollars Per Metric Ton']
plt.plot(x_uk, y_uk, c='y', label='Ukraine')

plt.legend()
plt.show()


#   Mexico VS USA

dfus = pd.read_excel('data/US.xlsx')
x_us = dfus['Date']
y_us = dfus['US Dollars Per Metric Ton']
plt.plot(x_us, y_us, c='r', label='USA')

dfm = pd.read_excel('data/Mexico.xlsx')
x_m = dfm['Date']
y_m = dfm['US Dollars Per Metric Ton']
plt.plot(x_m, y_m, c='purple', label='Mexico')

plt.legend()
plt.title('Mexico VS USA')
plt.savefig('image/MexicoUSAPrice.png')
plt.show()

#   Periodu 2000-2005 je bio najturbolentniji u odnosu na cijenu. Cijena je imala nejvece oscilacije u tom periodu.
#   sa 1.6k-3.3k-6.4k-3.6k-7k-2.7k      2-3-4-5

#   Mexico 2002 - 2 uragana   ap
#   Mexico 2003 - nista prorodno  da
#   Mexico 2004 - uragan cunami   ap
#   Mexico 2005 - 3 uragana, vulkan  da

#   1997-2010 su u vecini cijene u Meksiku i SAD-u bile u suprotnosti

#   BiH   2005: 1.2KM moji dobili 0.4KM
df = pd.read_excel('data/BiH.xlsx')
x = df['Date']
y = df['US Dollars Per Metric Ton']
plt.plot(x, y, c='g', label='BiH')
plt.legend()
plt.show()


#   BiH dodatno

df = pd.read_csv('data/Raspberries Price BiH.csv')
x = df['Date']
y = df['US Dollars Per Metric Ton']
plt.plot(x, y, c='g', label='BiH')
plt.legend()
plt.show()

# Mexico

dfm = pd.read_excel('data/Mexico.xlsx')
x_m = dfm['Date']
y_m = dfm['US Dollars Per Metric Ton']
plt.plot(x_m, y_m, c='purple', label='Mexico')

plt.legend()
plt.savefig('image/MexicoPrice.png')
plt.show()

# Hungary

dfm = pd.read_excel('data/Hungary.xlsx')
x_m = dfm['Date']
y_m = dfm['US Dollars Per Metric Ton']
plt.plot(x_m, y_m, c='purple', label='Hungary')
plt.legend()
plt.savefig('image/HungaryPrice.png')
plt.show()

# Poland VS Hungary
dfp = pd.read_excel('data/Poland.xlsx')
xp = dfp['Date']
yp = dfp['US Dollars Per Metric Ton']
plt.plot(xp, yp, label='Poland')
dfm = pd.read_excel('data/Hungary.xlsx')
x_m = dfm['Date']
y_m = dfm['US Dollars Per Metric Ton']
plt.plot(x_m, y_m, c='purple', label='Hungary')
plt.legend()
plt.savefig('image/HungaryPolandPrice.png')
plt.show()