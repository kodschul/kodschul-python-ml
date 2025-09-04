# Lösungen zu Lab 13.2 – Random Forests: Ensembles, Feature-Bedeutung

## Lösung zu Aufgabe 1 – Random Forest auf einem Klassifikationsdatensatz

```python
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# 1. Datensatz erzeugen
X, y = make_classification(n_samples=500, n_features=10, n_informative=5, n_redundant=2, random_state=0)

# 2. Train/Test-Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# 3. Random Forest trainieren
rf = RandomForestClassifier(n_estimators=100, random_state=0)
rf.fit(X_train, y_train)

# 4. Decision Tree trainieren
tree = DecisionTreeClassifier(random_state=0)
tree.fit(X_train, y_train)

# Accuracy berechnen
rf_train_acc = accuracy_score(y_train, rf.predict(X_train))
rf_test_acc = accuracy_score(y_test, rf.predict(X_test))

tree_train_acc = accuracy_score(y_train, tree.predict(X_train))
tree_test_acc = accuracy_score(y_test, tree.predict(X_test))

print(f'Random Forest – Training Accuracy: {rf_train_acc:.3f}, Test Accuracy: {rf_test_acc:.3f}')
print(f'Decision Tree – Training Accuracy: {tree_train_acc:.3f}, Test Accuracy: {tree_test_acc:.3f}')
```

Der einzelne Entscheidungsbaum passt sich stark an die Trainingsdaten an und zeigt eine größere Diskrepanz zwischen Training- und Test-Accuracy (Overfitting). Der Random Forest mittelt über viele Bäume und generalisiert daher besser.

## Lösung zu Aufgabe 2 – Feature Importance analysieren

```python
import pandas as pd
import matplotlib.pyplot as plt

# 1. Feature Importances abrufen
importances = rf.feature_importances_
feature_names = [f'Feature {i}' for i in range(X.shape[1])]
imp_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
imp_df_sorted = imp_df.sort_values('Importance', ascending=False)

# 2. Die fünf wichtigsten Features
top5 = imp_df_sorted.head(5)

# 3. Balkendiagramm
plt.barh(top5['Feature'][::-1], top5['Importance'][::-1])
plt.xlabel('Importance')
plt.title('Top 5 Feature Importances im Random Forest')
plt.show()

print(top5)
```

Hohe Importanzwerte bedeuten, dass das entsprechende Feature häufig bei Splits verwendet wird und stark zur Reduktion der Impurity beiträgt. Es gilt jedoch zu beachten, dass Importanzen nur relative Wichtigkeiten darstellen und nicht die Kausalität ersetzen.

## Lösung zu Aufgabe 3 – Ensemble-Vergleich: Bagging vs. Boosting

```python
from sklearn.ensemble import BaggingClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
import numpy as np

# 1. Basismodell für Bagging: Decision Tree
bag = BaggingClassifier(base_estimator=DecisionTreeClassifier(), n_estimators=50, random_state=0)
gboost = GradientBoostingClassifier(random_state=0)

# 2. 5-fache Kreuzvalidierung
models = {
    'RandomForest': rf,
    'Bagging': bag,
    'GradientBoosting': gboost
}

for name, model in models.items():
    scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    print(f'{name}: Mittelwert Accuracy = {scores.mean():.3f} (+/- {scores.std():.3f})')
```

Random Forests (Bagging mit Feature‑Randomisierung) reduzieren vor allem die Varianz und eignen sich gut für hochdimensionale Daten. Gradient Boosting verbessert die Fehler schrittweise und kann eine höhere Genauigkeit erzielen, ist jedoch empfindlicher gegenüber Hyperparametern und overfitting. Bagging mit vollständigen Bäumen liegt in seiner Leistung oft zwischen Random Forest und Boosting.
