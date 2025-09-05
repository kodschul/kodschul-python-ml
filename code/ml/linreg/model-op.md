# Lineare Regression optimieren mit Scikit-Learn

Dieses Dokument zeigt **Schritt für Schritt**, wie man ein lineares Regressionsmodell in Python mit `scikit-learn` aufbaut, trainiert und optimiert.  
Jeder Schritt enthält eine kurze Erklärung **was gemacht wird** und **warum**.

---

## 1) Daten vorbereiten

```python
import numpy as np
import pandas as pd

# Beispiel: df ist dein DataFrame
X = df.drop(columns=["target"]).values
y = df["target"].values
```

**Warum?**

- Wir trennen die **Features (X)** von der **Zielvariable (y)**.
- Lineare Regression braucht numerische Daten. Deshalb vorher sicherstellen, dass keine Strings oder fehlende Werte enthalten sind.

---

## 2) Train/Test-Split

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

**Warum?**

- Damit wir das Modell fair bewerten können, teilen wir die Daten in **Trainingsdaten** und **Testdaten** auf.
- So vermeiden wir, dass wir nur die Trainingsleistung messen (Overfitting).

---

## 3) Baseline-Modell: einfache LinearRegression

```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

base = LinearRegression()
base.fit(X_train, y_train)

pred = base.predict(X_test)
print("Baseline R2:", r2_score(y_test, pred))
print("MAE:", mean_absolute_error(y_test, pred))
print("RMSE:", mean_squared_error(y_test, pred, squared=False))
```

**Warum?**

- Erst eine **Baseline** messen, um zu wissen, wie gut (oder schlecht) unser Modell aktuell performt.
- Metriken:

  - **R²**: Erklärte Varianz (wie viel % der Zielvariable erklärt wird).
  - **MAE/RMSE**: Durchschnittlicher Fehler in den Vorhersagen.

---

## 4) Pipeline + Skalierung

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

pipe_base = Pipeline([
    ("scaler", StandardScaler()),
    ("linreg", LinearRegression())
])
pipe_base.fit(X_train, y_train)
```

**Warum?**

- Skalierung (StandardScaler) bringt alle Features auf die gleiche Skala.
- Das ist besonders wichtig, wenn wir gleich **Regularisierung** (Ridge, Lasso) einsetzen, weil dort Koeffizienten direkt bestraft werden.

---

## 5) Regularisierung testen (Ridge, Lasso, ElasticNet)

```python
from sklearn.linear_model import Ridge, Lasso, ElasticNet

ridge = Pipeline([("scaler", StandardScaler()), ("model", Ridge())])
lasso = Pipeline([("scaler", StandardScaler()), ("model", Lasso(max_iter=10000))])
enet  = Pipeline([("scaler", StandardScaler()), ("model", ElasticNet(max_iter=10000))])
```

**Warum?**

- **Ridge (L2)**: Verhindert zu große Koeffizienten → stabilisiert das Modell.
- **Lasso (L1)**: Kann unnötige Features komplett auf 0 setzen → Feature-Selektion.
- **ElasticNet**: Mischung aus beiden.

---

## 6) Hyperparameter mit Cross-Validation suchen

```python
from sklearn.model_selection import GridSearchCV

param_ridge = {"model__alpha": [0.001, 0.01, 0.1, 1, 10, 100]}
grid_ridge = GridSearchCV(ridge, param_grid=param_ridge, cv=5, scoring="r2", n_jobs=-1)
grid_ridge.fit(X_train, y_train)

print("Best Ridge:", grid_ridge.best_params_, "CV-R2:", grid_ridge.best_score_)
best_ridge = grid_ridge.best_estimator_
```

**Warum?**

- Wir testen verschiedene Werte von `alpha` und finden per **Cross-Validation** den besten Wert.
- Cross-Validation nutzt mehrere Splits der Daten, um robustere Ergebnisse zu bekommen.

Das gleiche Verfahren kann für **Lasso** und **ElasticNet** wiederholt werden.

---

## 7) Bestes Modell auf dem Testset prüfen

```python
for name, mdl in [("Ridge", best_ridge)]:
    p = mdl.predict(X_test)
    print(name, "→ R2:", r2_score(y_test, p),
          "MAE:", mean_absolute_error(y_test, p),
          "RMSE:", mean_squared_error(y_test, p, squared=False))
```

**Warum?**

- Jetzt testen wir unser **optimiertes Modell** auf den Testdaten.
- Vergleich mit der Baseline zeigt, ob sich die Optimierung lohnt.

---

## 8) (Optional) Nichtlinearität: PolynomialFeatures

```python
from sklearn.preprocessing import PolynomialFeatures

poly_ridge = Pipeline([
    ("poly", PolynomialFeatures(degree=2, include_bias=False)),
    ("scaler", StandardScaler()),
    ("model", Ridge())
])
```

**Warum?**

- Falls die Beziehung nicht linear ist, können **Polynom-Features** helfen.
- Achtung: Gefahr von Overfitting bei zu hohem Grad.

---

## 9) Residuen prüfen

```python
import matplotlib.pyplot as plt

y_pred = best_ridge.predict(X_test)
resid = y_test - y_pred

plt.scatter(y_pred, resid)
plt.axhline(0, color="red")
plt.xlabel("Vorhersage")
plt.ylabel("Residuum")
plt.title("Residuen-Check")
plt.show()
```

**Warum?**

- Residuen (Fehler) sollten **zufällig verteilt** sein.
- Wenn Muster sichtbar sind, ist das Modell falsch spezifiziert (z. B. fehlende Features oder Nichtlinearität).

---

## 10) Feature-Wichtigkeit/Selektion

- **Lasso** kann automatisch unwichtige Features auf 0 setzen.
- **Korrelationen** prüfen und redundante Features entfernen.
- Optional: **VIF (Variance Inflation Factor)** zur Erkennung von Multikollinearität.

**Warum?**

- Weniger Features → stabileres Modell → weniger Overfitting.

---

# Fazit

- Mit **LinearRegression** bekommst du eine schnelle Baseline.
- Mit **Ridge/Lasso/ElasticNet** kannst du Überanpassung reduzieren.
- Mit **GridSearchCV + Cross-Validation** findest du die besten Hyperparameter.
- Mit **Residuenanalyse & Feature-Selektion** stellst du sicher, dass das Modell wirklich sinnvoll ist.

Das ist der grundlegende Workflow, um ein lineares Regressionsmodell in Python **sauber zu optimieren**.

```

```
