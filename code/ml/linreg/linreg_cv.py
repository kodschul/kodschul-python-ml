import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import r2_score, accuracy_score, mean_squared_error

'''
Den perfekten Model zu erkennen  und besten Alpha Wert zu bestimmen.
'''

# 1. Daten laden
df = pd.read_csv("../data/people_weight_stats.csv")

print(f"{len(df)} Data loaded for training")

# 2. Features erstellen
X = df[["Height"]]
y = df['Weight']

lin_reg = LinearRegression()

# 3. Model trainieren + cross validation + testing
scores = cross_val_score(lin_reg, X, y, cv=5, scoring="r2")

print("R²-Scores für jeden Fold:", scores)
print("Durchschnittlicher R²:", np.mean(scores))
