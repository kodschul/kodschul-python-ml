import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import r2_score, accuracy_score, mean_squared_error

# 1. Daten laden
df = pd.read_csv("../data/SOCR-HeightWeight.csv")
print(f"{len(df)} Data loaded for training")


df["Height"] = (df['Height(Inches)'] * 2.54).round().astype(int)
df["Weight"] = (df['Weight(Pounds)'] * 0.453592).round().astype(int)


# 2. Features erstellen
X = df[["Height"]]
y = df['Weight']

linreg = LinearRegression()
lasso = Lasso(alpha=0.5)
ridge = Ridge(alpha=0.5)

# 3. Model trainieren + cross validation + testing
print("LINREG")
scores_linreg = cross_val_score(linreg, X, y, cv=5, scoring="r2")
print("R²-Scores für jeden Fold:", scores_linreg)
print("Durchschnittlicher R²:", np.mean(scores_linreg))

print("LASSO")
scores_lasso = cross_val_score(lasso, X, y, cv=5, scoring="r2")
print("R²-Scores für jeden Fold:", scores_lasso)
print("Durchschnittlicher R²:", np.mean(scores_lasso))

print("RIDGE")
scores_ridge = cross_val_score(ridge, X, y, cv=5, scoring="r2")
print("R²-Scores für jeden Fold:", scores_ridge)
print("Durchschnittlicher R²:", np.mean(scores_ridge))
