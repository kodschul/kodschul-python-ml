# Lösungen zu Lab 13.3 – Performancevergleich & Interpretierbarkeit

## Lösung zu Aufgabe 1 – Modellvergleich

```python
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd

# 1. Datensatz erzeugen
X, y = make_classification(n_samples=1000, n_features=8, n_informative=6, random_state=42)

# 2. Train/Test-Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 3. Modelle trainieren
models = {
    'DecisionTree': DecisionTreeClassifier(random_state=42),
    'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
    'GradientBoosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
}

results = []
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    results.append({
        'Modell': name,
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1-Score': f1_score(y_test, y_pred)
    })

res_df = pd.DataFrame(results)
print(res_df)
```

Der Random Forest und Gradient Boosting erzielen in der Regel höhere Werte bei Accuracy und F1‑Score als ein einzelner Entscheidungsbaum, da Ensembles besser generalisieren. Der Entscheidungsbaum kann jedoch leichter interpretiert werden.

## Lösung zu Aufgabe 2 – Interpretierbarkeit eines Entscheidungsbaums

```python
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree

# 1. Baum mit begrenzter Tiefe trainieren
tree = DecisionTreeClassifier(max_depth=3, random_state=42)
tree.fit(X_train, y_train)

# 2. Baum visualisieren
plt.figure(figsize=(12, 6))
plot_tree(tree, feature_names=[f'Feature{i}' for i in range(X.shape[1])],
          class_names=['Klasse0','Klasse1'], filled=True)
plt.title('Entscheidungsbaum (max_depth=3)')
plt.show()

# 3. Beispielpfade analysieren
# Wir können die Regeln anhand des Bildes interpretieren, z. B.:
# - Wenn Feature3 <= -0.2 und Feature5 <= 0.7, dann Klasse 0
# - Wenn Feature3 > -0.2 und Feature1 <= 1.0, dann Klasse 1
```

Ein flacher Baum mit wenigen Ebenen lässt sich leicht manuell nachvollziehen: Jede Bedingung teilt die Daten entlang einer einzigen Schwelle, und man kann die Bedeutung der Merkmale schnell erfassen. Ein tiefer Baum hingegen enthält viele Verzweigungen und wird dadurch schwer verständlich.

## Lösung zu Aufgabe 3 – Permutation Importance

```python
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt

# 1. Permutation Importance auf dem Random-Forest-Modell berechnen
result = permutation_importance(models['RandomForest'], X_test, y_test,
                                n_repeats=10, random_state=0, scoring='accuracy')

# 2. Wichtigkeiten sortieren
sorted_idx = result.importances_mean.argsort()[::-1]
features = [f'Feature{i}' for i in range(X.shape[1])]
importances = result.importances_mean[sorted_idx]

# 3. Grafische Darstellung
plt.barh([features[i] for i in sorted_idx[::-1]], importances[::-1])
plt.xlabel('Permutation Importance (Accuracy drop)')
plt.title('Permutation Importance – Random Forest')
plt.show()

print('Permutation Importances (Accuracy drop):')
for idx in sorted_idx:
    print(f'{features[idx]}: {result.importances_mean[idx]:.4f}')

# Vergleich mit eingebauter Feature Importance
print('
Eingebaute Feature Importances:')
for idx in sorted_idx:
    print(f'{features[idx]}: {models["RandomForest"].feature_importances_[idx]:.4f}')
```

Während `feature_importances_` die Splits in den Bäumen auswertet, misst die Permutation Importance den tatsächlichen Einfluss eines Features auf die Modellleistung, indem es die Werte zufällig permutiert und die Verringerung der Accuracy beobachtet. Dadurch werden auch Wechselwirkungen und Maskierungseffekte besser erkannt, was sie robuster macht als die eingebauten Importanzen.
