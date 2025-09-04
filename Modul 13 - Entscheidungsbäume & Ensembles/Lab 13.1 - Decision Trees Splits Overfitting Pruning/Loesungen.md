# Lösungen zu Lab 13.1 – Decision Trees: Splits, Overfitting, Pruning

## Lösung zu Aufgabe 1 – Entscheidungsbaum auf synthetischen Daten

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# 1. Daten erzeugen
X, y = make_moons(n_samples=300, noise=0.25, random_state=42)

# 2. Split in Train/Test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)

# 3. Entscheidungsbaum ohne Begrenzung
tree = DecisionTreeClassifier(random_state=42)
tree.fit(X_train, y_train)

# 4. Entscheidungsgrenze visualisieren
def plot_decision_boundary(model, X, y, title):
    x_min, x_max = X[:,0].min() - 0.5, X[:,0].max() + 0.5
    y_min, y_max = X[:,1].min() - 0.5, X[:,1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                         np.linspace(y_min, y_max, 300))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
    plt.scatter(X[:,0], X[:,1], c=y, edgecolor='k', cmap='coolwarm')
    plt.title(title)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()

plot_decision_boundary(tree, X_train, y_train, 'Entscheidungsbaum – Trainingsdaten')

# 5. Accuracy berechnen
train_acc = accuracy_score(y_train, tree.predict(X_train))
test_acc = accuracy_score(y_test, tree.predict(X_test))
print(f'Training Accuracy: {train_acc:.3f}')
print(f'Test Accuracy: {test_acc:.3f}')
```

Die Genauigkeit auf den Trainingsdaten ist oft deutlich höher als auf den Testdaten – ein Hinweis auf Overfitting. Das Modell hat die Trainingsdaten nahezu auswendig gelernt und generalisiert schlechter auf neue Daten.

## Lösung zu Aufgabe 2 – Einfluss der Baumtiefe

```python
from sklearn.metrics import accuracy_score

depths = [1, 3, 5, 7, 9]
train_scores = []
test_scores = []

for d in depths:
    tree = DecisionTreeClassifier(max_depth=d, random_state=42)
    tree.fit(X_train, y_train)
    train_scores.append(accuracy_score(y_train, tree.predict(X_train)))
    test_scores.append(accuracy_score(y_test, tree.predict(X_test)))

# Ergebnisse plotten
plt.plot(depths, train_scores, marker='o', label='Trainings-Accuracy')
plt.plot(depths, test_scores, marker='o', label='Test-Accuracy')
plt.xlabel('max_depth')
plt.ylabel('Accuracy')
plt.title('Einfluss der Baumtiefe')
plt.legend()
plt.show()

for d, tr, te in zip(depths, train_scores, test_scores):
    print(f'max_depth={d}: Training {tr:.3f}, Test {te:.3f}')
```

Bei kleiner Tiefe (1 oder 3) ist das Modell zu simpel und unterfitten; bei großer Tiefe (z. B. 9) ist die Trainings-Accuracy nahezu perfekt, aber die Test-Accuracy sinkt wieder – ein Zeichen für Overfitting. Eine mittlere Tiefe bringt meist die beste Balance.

## Lösung zu Aufgabe 3 – Kostenkomplexitätspruning

```python
# 1. Ursprünglichen Baum trainieren
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# 2. ccp_alpha-Werte bestimmen
path = clf.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas = path.ccp_alphas

# Um die Anzahl zu begrenzen, wählen wir die ersten 5 von ihnen (ohne 0)
ccp_alphas = ccp_alphas[1:6]

test_accs = []
for alpha in ccp_alphas:
    pruned_tree = DecisionTreeClassifier(random_state=42, ccp_alpha=alpha)
    pruned_tree.fit(X_train, y_train)
    test_accs.append(accuracy_score(y_test, pruned_tree.predict(X_test)))

for alpha, acc in zip(ccp_alphas, test_accs):
    print(f'ccp_alpha={alpha:.5f}: Test Accuracy={acc:.3f}')
```

Durch das Beschneiden (Pruning) werden weniger Knoten im Baum verwendet, was Overfitting reduziert. Ein mittlerer `ccp_alpha`-Wert führt in der Regel zu einer guten Balance zwischen Komplexität und Generalisierung; sehr hohe Werte führen zu einem zu einfachen Modell (Underfitting).
