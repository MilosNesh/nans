from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
import pandas as pd

# Podaci
df = pd.read_csv('data/raspberry.csv')
cols = ['Hectograms Per Hectare', 'Hectares', 'Thousand US Dollars PPP = 2004–2006', 'Metric Tons']
X = df[cols]
y = df['US Dollars Per Metric Ton']

# Podjela podataka
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, shuffle=True, random_state=42)

param_grid = {
    'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
    'activation': ['relu', 'tanh', 'logistic'],
    'learning_rate_init': [0.001, 0.01, 0.1],
    'max_iter': [1000, 2000, 3000]
}

model = MLPRegressor(random_state=42)
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_

print(best_params)

import json
# Ime fajla u koji ćemo upisati parametre
file_name = 'param_nn.json'

# Upisivanje parametara u JSON fajl
with open(file_name, 'w') as json_file:
    json.dump(best_params, json_file)

print(f"Parametri su upisani u fajl: {file_name}")