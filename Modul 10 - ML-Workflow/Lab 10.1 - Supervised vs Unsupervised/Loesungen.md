# Lösungen zu Lab 10.1 – Supervised vs. Unsupervised Learning

## Lösung zu Aufgabe 1 – KMeans‐Clustering

```python
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans

# 1. Iris‑Datensatz laden und in DataFrame konvertieren
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)

# 2. KMeans‑Clustering mit k=3
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X)

# 3. Visualisierung
fig, ax = plt.subplots()
scatter = ax.scatter(X.iloc[:, 0], X.iloc[:, 1], c=clusters)
ax.set_xlabel("Sepal length (cm)")
ax.set_ylabel("Sepal width (cm)")
ax.set_title("KMeans‑Cluster auf Iris‑Merkmalen")
plt.show()

# 4. Vergleich mit echten Arten
labels = iris.target  # echte Klassen
ct = pd.crosstab(clusters, labels,
                 rownames=["Cluster"], colnames=["Art"])
print(ct)
```

Der Crosstab zeigt, wie die Cluster mit den tatsächlichen Arten zusammenhängen. Obwohl KMeans die Arten nicht kennt, erkennt es in den Daten klare Gruppen.

## Lösung zu Aufgabe 2 – Entscheidungsbaumklassifikation

```python
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# 1. Datensatz laden
iris = load_iris()
X = iris.data
y = iris.target

# 2. Train/Test‑Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# 3. Entscheidungsbaum trainieren
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# 4. Vorhersagen und Evaluation
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print(f"Accuracy: {acc:.3f}")
print("Confusion Matrix:")
print(cm)
```

Der Entscheidungsbaum erreicht in der Regel eine hohe Genauigkeit auf dem Iris‑Datensatz. Die Confusion Matrix zeigt, wie viele Exemplare jeder Art korrekt bzw. falsch klassifiziert wurden.

## Lösung zu Aufgabe 3 – Szenarien klassifizieren

1. **Unüberwachtes Lernen**. Es gibt keine vorgegebenen Labels; das Unternehmen möchte die Kunden anhand ihrer Merkmale in Gruppen clustern, was typisch für unsupervised learning ist (z. B. k‑Means oder hierarchisches Clustering).

2. **Überwachtes Lernen (Regression)**. Der Umsatz ist eine kontinuierliche Zielvariable, und es existieren historische Daten mit bekannten Umsätzen. Ein Modell wie eine lineare Regression kann diese Werte vorhersagen.

3. **Unüberwachtes Lernen (Anomalieerkennung)**. Es gibt keine expliziten Labels für „normal“ oder „Fehler“. Stattdessen sollen ungewöhnliche Muster erkannt werden. Dies erfolgt häufig mithilfe unüberwachter Methoden wie Isolation Forest oder Autoencoder.
