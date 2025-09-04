# Lösungen zu Lab 12.1 – K‑Nearest Neighbors: Bias/Varianz und Skalierungseffekte

## Lösung zu Aufgabe 1 – Bias‑Varianz‑Tradeoff

```python
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Daten laden und splitten
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2,
    random_state=42, stratify=iris.target)

k_values = [1, 3, 5, 7, 9, 11, 15]
train_acc = []
test_acc = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    train_acc.append(accuracy_score(y_train, knn.predict(X_train)))
    test_acc.append(accuracy_score(y_test, knn.predict(X_test)))

# Ergebnisse plotten
plt.plot(k_values, train_acc, label='Trainings‑Accuracy', marker='o')
plt.plot(k_values, test_acc, label='Test‑Accuracy', marker='o')
plt.xlabel("k (Anzahl Nachbarn)")
plt.ylabel("Accuracy")
plt.title("KNN‑Bias‑Varianz‑Tradeoff")
plt.legend()
plt.show()

for k, tr, te in zip(k_values, train_acc, test_acc):
    print(f"k={k:>2}: Training {tr:.3f}, Test {te:.3f}")
```

Bei kleinen `k` (z. B. 1) ist die Trainings‑Accuracy sehr hoch, aber die Test‑Accuracy kann sinken (Overfitting). Bei sehr großen `k` wird das Modell zu einfach (Underfitting). Ein mittlerer Bereich – häufig 3 bis 7 – liefert eine gute Balance.

## Lösung zu Aufgabe 2 – Skalierungseffekte

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# 1. Manuell erstellter Datensatz
data = pd.DataFrame({
    'Länge_cm': [150, 160, 170, 180, 190, 200],
    'Gewicht_kg': [50, 55, 65, 80, 90, 100],
    'Klasse': [0, 0, 0, 1, 1, 1]
})

X = data[['Länge_cm', 'Gewicht_kg']].values
y = data['Klasse'].values

# 2. Train/Test‑Split (4 train, 2 test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=2, random_state=42, stratify=y)

# 3. KNN ohne Skalierung
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
acc_unscaled = accuracy_score(y_test, knn.predict(X_test))
print(f"Accuracy ohne Skalierung: {acc_unscaled:.3f}")

# 4. Mit Skalierung
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

knn_scaled = KNeighborsClassifier(n_neighbors=3)
knn_scaled.fit(X_train_scaled, y_train)
acc_scaled = accuracy_score(y_test, knn_scaled.predict(X_test_scaled))
print(f"Accuracy mit Skalierung: {acc_scaled:.3f}")
```

Ohne Skalierung dominiert das Merkmal mit der größeren Skala (z. B. `Länge_cm`) die Distanzberechnung. Die Skalierung führt beide Merkmale auf einen vergleichbaren Bereich zurück, sodass sie gleichermaßen zum Distanzmaß beitragen. In vielen Fällen verbessert dies die Genauigkeit deutlich.

## Lösung zu Aufgabe 3 – KNN‑Regression

```python
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np

# 1. Datensatz erzeugen
X, y = make_regression(n_samples=100, n_features=1, noise=20, random_state=1)

# 2. Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)

# 3. Verschiedene k-Werte
k_vals = [1, 3, 5, 7, 9]
rmse_vals = []

for k in k_vals:
    knn_reg = KNeighborsRegressor(n_neighbors=k)
    knn_reg.fit(X_train, y_train)
    y_pred = knn_reg.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    rmse_vals.append(rmse)

# 4. RMSE anzeigen
plt.plot(k_vals, rmse_vals, marker='o')
plt.xlabel("k (Nachbarn)")
plt.ylabel("RMSE")
plt.title("KNN‑Regression: RMSE vs. k")
plt.show()

for k, rmse in zip(k_vals, rmse_vals):
    print(f"k={k:>2}: RMSE={rmse:.2f}")
```

Auch bei der Regression zeigt ein sehr kleines `k` eine hohe Varianz (das Modell passt sich stark an einzelne Punkte an), während ein sehr großes `k` zu hohem Bias führt. Ein mittlerer Wert (z. B. 5) liefert oft den niedrigsten RMSE.
