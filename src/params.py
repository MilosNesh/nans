from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import pandas as pd

# Podaci
df = pd.read_csv('data/raspberry.csv')
cols = ['Hectograms Per Hectare', 'Metric Tons']
X = df[cols]
y = df['US Dollars Per Metric Ton']

# Podjela podataka
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, shuffle=True, random_state=42)

# Definisanje modela
model = RandomForestRegressor()

# Definisanje opsega vrednosti parametara koje želite isprobati
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 10, 15],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Inicijalizacija GridSearchCV objekta
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error')

# Pokretanje pretrage
grid_search.fit(X_train, y_train)

# Prikaz najboljih parametara
print("Najbolji parametri:", grid_search.best_params_)

import json
# Ime fajla u koji ćemo upisati parametre
file_name = 'parametri.json'

best = grid_search.best_params_

# Upisivanje parametara u JSON fajl
with open(file_name, 'w') as json_file:
    json.dump(best, json_file)

print(f"Parametri su upisani u fajl: {file_name}")

#'max_depth': 5, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 100