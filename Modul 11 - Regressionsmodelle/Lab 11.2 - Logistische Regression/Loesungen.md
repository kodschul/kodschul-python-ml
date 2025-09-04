# Lösungen zu Lab 11.2 – Logistische Regression: Klassifikation und Entscheidungsgrenzen

## Lösung zu Aufgabe 1 – Datensatz und Modelltraining

```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# 1. Datensatz erzeugen
X, y = make_classification(
    n_samples=200, n_features=2, n_informative=2, n_redundant=0,
    n_clusters_per_class=1, random_state=0)

# 2. Train/Test‑Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0)

# 3. Modell trainieren
clf = LogisticRegression()
clf.fit(X_train, y_train)
print("Koeffizienten:", clf.coef_)
print("Achsenabschnitt:", clf.intercept_)

# 4. Bewertung
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
print(f"Accuracy: {acc:.3f}, Precision: {prec:.3f}, Recall: {rec:.3f}, F1‑Score: {f1:.3f}")
```

Das Modell lernt eine lineare Entscheidungsgrenze in dem zweidimensionalen Raum. Die Koeffizienten geben die Steigung der Grenze an, der Achsenabschnitt verschiebt die Grenze.

## Lösung zu Aufgabe 2 – Entscheidungsgrenze visualisieren

```python
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np

# Gitter erzeugen
x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                     np.linspace(y_min, y_max, 300))
grid = np.c_[xx.ravel(), yy.ravel()]

# Vorhersage der Wahrscheinlichkeiten für Klasse 1
probs = clf.predict_proba(grid)[:, 1].reshape(xx.shape)

# Kontur bei Schwellenwert 0.5
plt.contourf(xx, yy, probs, levels=20, cmap="coolwarm", alpha=0.8)
contour = plt.contour(xx, yy, probs, levels=[0.5], colors='black', linewidths=1)
contour.collections[0].set_label("Entscheidungsgrenze (p=0.5)")

# Trainingspunkte plotten
plt.scatter(X_train[y_train==0, 0], X_train[y_train==0, 1], c='blue', label='Klasse 0')
plt.scatter(X_train[y_train==1, 0], X_train[y_train==1, 1], c='red', label='Klasse 1')
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Logistische Regression: Entscheidungsfläche")
plt.legend()
plt.show()
```

Die Konturlinie (p = 0.5) teilt den Raum in zwei farbige Regionen, die jeweils einer Klasse zugeordnet werden. Der sanfte Farbverlauf zeigt die Wahrscheinlichkeiten, die das Modell berechnet.

## Lösung zu Aufgabe 3 – Schwellenwert und ROC‑Kurve

```python
from sklearn.metrics import roc_curve, roc_auc_score

# 1. Vorhergesagte Wahrscheinlichkeiten
y_prob = clf.predict_proba(X_test)[:, 1]

# 2. Schwellenwert 0.7
threshold = 0.7
y_pred_07 = (y_prob >= threshold).astype(int)

# Metriken bei 0.7
acc07 = accuracy_score(y_test, y_pred_07)
prec07 = precision_score(y_test, y_pred_07)
rec07 = recall_score(y_test, y_pred_07)
f107 = f1_score(y_test, y_pred_07)
print(f"Schwellenwert 0.7 – Accuracy: {acc07:.3f}, Precision: {prec07:.3f}, Recall: {rec07:.3f}, F1: {f107:.3f}")

# Vergleich mit 0.5
# (Die Werte aus Aufgabe 1 gelten für 0.5.)

# 3. ROC‑Kurve und AUC
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
auc = roc_auc_score(y_test, y_prob)

plt.plot(fpr, tpr, label=f'ROC‑Kurve (AUC = {auc:.3f})')
plt.plot([0,1],[0,1],'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC‑Kurve der logistischen Regression")
plt.legend()
plt.show()
```

Ein höherer Schwellenwert wie 0.7 führt oft zu einer höheren Precision (weniger falsch‑positive Vorhersagen), aber zu einer niedrigeren Recall (mehr falsch‑negative). Die ROC‑Kurve zeigt das Verhältnis zwischen True‑Positive‑Rate und False‑Positive‑Rate für alle möglichen Schwellenwerte; die Fläche unter der Kurve (AUC) fasst die Trennleistung des Modells zusammen. Eine AUC von 1 bedeutet perfekte Trennung, eine AUC von 0.5 entspricht Zufall.
