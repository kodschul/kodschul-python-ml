# Lösungen zu Lab 12.2 – Support Vector Machines: Kernel‑Idee & Margin

## Lösung zu Aufgabe 1 – Lineare SVM und Margin

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# 1. Datensatz erzeugen
X, y = make_blobs(n_samples=200, centers=2, cluster_std=1.0, random_state=1)

# 2. Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# 3. Lineare SVM trainieren
svc = SVC(kernel='linear')
svc.fit(X_train, y_train)

# 4. Visualisierung
def plot_svm_decision_boundary(model, X, y):
    plt.figure(figsize=(6,4))
    # Plot Daten
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', s=30, edgecolors='k')

    # Entscheidungslinie und Margins
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # Meshgrid
    xx = np.linspace(xlim[0], xlim[1], 30)
    yy = np.linspace(ylim[0], ylim[1], 30)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    Z = model.decision_function(xy).reshape(XX.shape)

    # Plot decision boundary und margins
    ax.contour(XX, YY, Z, levels=[-1, 0, 1], linestyles=['--','-','--'],
               colors='k')
    # Support Vectors
    ax.scatter(model.support_vectors_[:, 0],
               model.support_vectors_[:, 1],
               s=100, linewidth=1, facecolors='none', edgecolors='k')
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.set_title("Lineare SVM: Entscheidungsgerade und Margin")

plot_svm_decision_boundary(svc, X_train, y_train)
plt.show()
```

Die lineare SVM trennt die beiden Klassen mit einer Geraden. Die gestrichelten Linien stellen den Rand (Margin) dar; die Punkte auf diesen Linien sind die Stützvektoren, welche die Lage der Trennlinie bestimmen.

## Lösung zu Aufgabe 2 – RBF‑Kernel

```python
from sklearn.datasets import make_circles
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import numpy as np

# 1. Nichtlinear trennbarer Datensatz
X, y = make_circles(n_samples=400, factor=0.5, noise=0.1, random_state=0)

# 2. Modelle mit linearem und RBF‑Kernel
svc_linear = SVC(kernel='linear').fit(X, y)
svc_rbf = SVC(kernel='rbf', gamma='scale').fit(X, y)

def plot_decision_surface(model, X, y, title):
    x_min, x_max = X[:,0].min() - 1, X[:,0].max() + 1
    y_min, y_max = X[:,1].min() - 1, X[:,1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 400),
                         np.linspace(y_min, y_max, 400))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
    plt.scatter(X[:,0], X[:,1], c=y, cmap='coolwarm', edgecolors='k', s=20)
    plt.title(title)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")

plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plot_decision_surface(svc_linear, X, y, "Linearer Kernel")
plt.subplot(1,2,2)
plot_decision_surface(svc_rbf, X, y, "RBF‑Kernel")
plt.tight_layout()
plt.show()
```

Der lineare Kernel kann die verschachtelten Kreise nicht trennen – er erzeugt eine gerade Trennlinie. Der RBF‑Kernel projiziert die Daten in eine höhere Dimension und findet eine ringförmige Trennfläche, die die zwei Klassen korrekt trennt.

## Lösung zu Aufgabe 3 – Hyperparameter‑Tuning

```python
from sklearn.model_selection import GridSearchCV

# 1. Datensatz aus Aufgabe 2 nutzen: X, y
# Parameter‑Grid
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [0.01, 0.1, 1, 10]
}

# 3. GridSearchCV mit RBF‑Kernel
svc = SVC(kernel='rbf')
grid = GridSearchCV(svc, param_grid, cv=5, scoring='accuracy')
grid.fit(X, y)

print("Beste Parameter:", grid.best_params_)
print(f"Mittlere Accuracy: {grid.best_score_:.3f}")
```

Der Parameter **C** steuert die Fehlertoleranz: Ein kleines C erlaubt mehr Fehlklassifikationen, führt zu einer größeren Margin und glatteren Entscheidungsgrenzen (größerer Bias, kleinere Varianz). Ein großes C erzwingt eine möglichst fehlerfreie Klassifikation der Trainingsdaten, was zu engeren Margins und potenziellem Overfitting führt.  
Der Parameter **gamma** bestimmt die Reichweite des RBF‑Kernels: Ein kleines `gamma` führt zu einer glatten, globalen Trennfläche, während ein großes `gamma` die Entscheidungsgrenze sehr lokal macht (Modelle reagieren stark auf einzelne Punkte). Das optimale Paar aus `C` und `gamma` liefert die beste Balance zwischen Bias und Varianz.
