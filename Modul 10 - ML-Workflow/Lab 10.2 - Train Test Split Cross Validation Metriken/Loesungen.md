# Lösungen zu Lab 10.2 – Train/Test‑Split, Cross‑Validation und Metriken

## Lösung zu Aufgabe 1 – Regressionsaufgabe mit Train/Test‑Split

```python
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np

# 1. Datensatz erzeugen
X, y = make_regression(n_samples=200, n_features=1, noise=10.0, random_state=42)

# 2. Train/Test‑Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# 3. Modell trainieren
reg = LinearRegression()
reg.fit(X_train, y_train)

# 4. Bewertung
y_pred = reg.predict(X_test)
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print(f"R^2: {r2:.3f}")
print(f"MSE: {mse:.2f}")
```

Das R‑Quadrat zeigt, wie viel Varianz in den Testdaten durch das Modell erklärt wird. Ein MSE näher bei 0 bedeutet bessere Vorhersagen; aufgrund des Rauschens und der kleinen Datengröße ist der MSE hier nicht exakt 0.

## Lösung zu Aufgabe 2 – Klassifikationsmetriken und Confusion Matrix

```python
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

# 1. Daten laden und splitten
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42, stratify=iris.target)

# 2. KNN‑Modell trainieren
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# 3. Vorhersagen und Metriken
y_pred = knn.predict(X_test)
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# 4. Confusion Matrix als Heatmap
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(4, 3))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=iris.target_names,
            yticklabels=iris.target_names)
plt.xlabel("Vorhergesagte Klasse")
plt.ylabel("Wahre Klasse")
plt.title("Confusion Matrix des KNN‑Klassifikators")
plt.tight_layout()
plt.show()
```

Der `classification_report` liefert Accuracy, Precision, Recall und F1‑Score für jede Klasse. Die Heatmap der Confusion Matrix macht Fehlklassifikationen visuell sichtbar.

## Lösung zu Aufgabe 3 – K‑Fold Cross‑Validation

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
import numpy as np

# Iris‑Daten laden
iris = load_iris()
X, y = iris.data, iris.target

# 1./2. Cross‑Validation für Entscheidungsbaum
tree = DecisionTreeClassifier(random_state=42)
cv_scores_tree = cross_val_score(tree, X, y, cv=5, scoring='accuracy')
print("Entscheidungsbaum Accuracy je Fold:", cv_scores_tree)
print("Durchschnittliche Accuracy (Tree):", np.mean(cv_scores_tree))

# 3. Cross‑Validation für SVC (linearer Kernel)
svc = SVC(kernel='linear', C=1.0)
cv_scores_svc = cross_val_score(svc, X, y, cv=5, scoring='accuracy')
print("SVC Accuracy je Fold:", cv_scores_svc)
print("Durchschnittliche Accuracy (SVC):", np.mean(cv_scores_svc))
```

Bei der Kreuzvalidierung wird das Modell mehrfach mit unterschiedlichen Aufteilungen trainiert und getestet. Die durchschnittliche Accuracy ermöglicht einen robusteren Vergleich zwischen dem Entscheidungsbaum und dem linearen SVC; oft schneidet der SVC auf dem Iris‑Datensatz etwas besser ab, weil er eine lineare Entscheidungsgrenze sucht und weniger zum Überfitten neigt.
