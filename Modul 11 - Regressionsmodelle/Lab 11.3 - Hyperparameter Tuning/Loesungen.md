# Lösungen zu Lab 11.3 – Hyperparameter‑Tuning (Grid‑ und Randomized Search)

## Lösung zu Aufgabe 1 – KNN‑Klassifikation optimieren

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

# 1. Daten laden
iris = load_iris()
X, y = iris.data, iris.target

# 2. Basismodell definieren
knn = KNeighborsClassifier()

# 3. Parameter‑Grid
param_grid = {'n_neighbors': [1, 3, 5, 7, 9, 11]}

# 4. GridSearchCV mit 5‑facher CV
grid = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy')
grid.fit(X, y)

print("Bestes n_neighbors:", grid.best_params_['n_neighbors'])
print(f"Mittlere Accuracy: {grid.best_score_:.3f}")
```

Grid Search probiert alle angegebenen Nachbarzahlen aus und wählt diejenige mit der höchsten durchschnittlichen Accuracy aus. Auf dem Iris‑Datensatz liegt der optimale `k` meist zwischen 3 und 5.

## Lösung zu Aufgabe 2 – Ridge‑Regression mit Grid Search

```python
from sklearn.datasets import make_regression
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV

# 1. Datensatz erzeugen
X, y = make_regression(n_samples=200, n_features=5, noise=15, random_state=0)

# 2. Modell definieren
ridge = Ridge()

# 3. Parameter‑Grid
alphas = [0.01, 0.1, 1.0, 10.0, 100.0]
param_grid = {'alpha': alphas}

# 4. GridSearchCV mit neg_mean_squared_error
grid = GridSearchCV(ridge, param_grid, cv=5,
                    scoring='neg_mean_squared_error')
grid.fit(X, y)

# Da MSE negativ maximiert wird, ist der beste alpha der mit dem größten Score (= kleinster Fehler)
best_alpha = grid.best_params_['alpha']
best_mse = -grid.best_score_
print("Bestes alpha:", best_alpha)
print(f"Geringster MSE: {best_mse:.2f}")
```

Das kleinste `alpha` führt zu der geringsten Regularisierung und kann bei einem einfachen Datensatz den geringsten Fehler ergeben. Bei komplexeren Daten kann ein moderates `alpha` von Vorteil sein, um Overfitting zu reduzieren.

## Lösung zu Aufgabe 3 – Randomized Search für Random Forest

```python
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
import numpy as np

# 1. Datensatz laden
data = load_breast_cancer()
X, y = data.data, data.target

# 2. Modell definieren
rf = RandomForestClassifier(random_state=42)

# 3. Parameter‑Raum
param_dist = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 5, 10, 20],
    'max_features': ['sqrt', 'log2', None]
}

# 4. RandomizedSearchCV
random_search = RandomizedSearchCV(
    rf, param_distributions=param_dist, n_iter=20,
    cv=5, scoring='accuracy', random_state=42)

random_search.fit(X, y)

# 5. Beste Parameter und Accuracy
print("Beste Parameterkombination:", random_search.best_params_)
print(f"Mittlere Accuracy: {random_search.best_score_:.3f}")
```

Randomized Search probiert nur eine zufällige Stichprobe von Parameterkombinationen aus (hier 20), was Rechenzeit spart. Die beste Kombination liefert die höchste durchschnittliche Accuracy in der Kreuzvalidierung.
