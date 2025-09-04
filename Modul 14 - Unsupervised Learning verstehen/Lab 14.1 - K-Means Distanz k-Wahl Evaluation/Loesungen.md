# Lösungen zu Lab 14.1 – K-Means Clustering: Distanzmaße, k-Wahl, Evaluation

## Lösung zu Aufgabe 1 – K-Means auf synthetischen Daten

```python
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import numpy as np

# 1. Datensatz erzeugen
X, y_true = make_blobs(n_samples=400, centers=3, cluster_std=0.60, random_state=0)

# 2. K-Means mit k=3
kmeans = KMeans(n_clusters=3, random_state=0)
labels = kmeans.fit_predict(X)

# 3. Visualisierung
plt.scatter(X[:,0], X[:,1], c=labels, s=30, cmap='viridis')
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], c='red', s=200, marker='X')
plt.title('K-Means Clustering (k=3)')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()

# 4. Inertia und Silhouette
inertia = kmeans.inertia_
sil_score = silhouette_score(X, labels)
print(f'Inertia: {inertia:.2f}')
print(f'Silhouette-Score: {sil_score:.3f}')
```

Die Inertia ist die Summe der quadratischen Abstände der Punkte zu ihren Clusterzentren – je kleiner, desto kompakter sind die Cluster. Der Silhouette-Score misst Trennschärfe und Kompaktheit (zwischen -1 und 1).

## Lösung zu Aufgabe 2 – k wählen mit Elbow und Silhouette

```python
inertias = []
sil_scores = []
k_values = range(1, 11)

for k in k_values:
    km = KMeans(n_clusters=k, random_state=0)
    labels = km.fit_predict(X)
    inertias.append(km.inertia_)
    if k > 1:
        sil_scores.append(silhouette_score(X, labels))
    else:
        sil_scores.append(np.nan)

# Elbow-Diagramm
plt.plot(k_values, inertias, marker='o')
plt.xlabel('k')
plt.ylabel('Inertia')
plt.title('Elbow-Methode zur Wahl von k')
plt.show()

# Silhouette-Diagramm (ab k=2)
plt.plot(list(k_values)[1:], sil_scores[1:], marker='o')
plt.xlabel('k')
plt.ylabel('Silhouette-Score')
plt.title('Silhouette-Score vs. k')
plt.show()

for k, inertia, sil in zip(k_values, inertias, sil_scores):
    print(f'k={k}: Inertia={inertia:.2f}, Silhouette={sil:.3f}')
```

Der Knick im Elbow-Diagramm tritt typischerweise bei k=3 auf – ab diesem Punkt sinkt die Inertia weniger stark. Der Silhouette-Score ist bei k=3 ebenfalls hoch. Beide Methoden deuten darauf hin, dass drei Cluster passend sind.

## Lösung zu Aufgabe 3 – Skalierung und Distanzmaß

```python
from sklearn.preprocessing import StandardScaler

# 1. Datensatz mit verschiedenen Skalen erzeugen
import numpy as np
np.random.seed(0)
n = 300
feature1 = np.random.rand(n) * 1000   # Werte zwischen 0 und 1000
feature2 = np.random.rand(n)          # Werte zwischen 0 und 1
# Erstellen Sie künstliche Cluster
clusters = np.random.choice([0,1,2], size=n, p=[0.3,0.4,0.3])
# Verschieben Features je nach Cluster
feature1 += clusters * 300
feature2 += clusters * 0.5
X2 = np.column_stack((feature1, feature2))

# 2. K-Means ohne Skalierung
km_unscaled = KMeans(n_clusters=3, random_state=0)
labels_unscaled = km_unscaled.fit_predict(X2)
inertia_unscaled = km_unscaled.inertia_
sil_unscaled = silhouette_score(X2, labels_unscaled)

# 3. K-Means nach Skalierung
scaler = StandardScaler()
X2_scaled = scaler.fit_transform(X2)
km_scaled = KMeans(n_clusters=3, random_state=0)
labels_scaled = km_scaled.fit_predict(X2_scaled)
inertia_scaled = km_scaled.inertia_
sil_scaled = silhouette_score(X2_scaled, labels_scaled)

print('Ohne Skalierung: Inertia={:.2f}, Silhouette={:.3f}'.format(inertia_unscaled, sil_unscaled))
print('Mit Skalierung : Inertia={:.2f}, Silhouette={:.3f}'.format(inertia_scaled, sil_scaled))
```

Beim ersten Versuch dominiert das Feature mit der größeren Skala (0–1000) die Distanzberechnung; die Cluster basieren vor allem auf diesem Merkmal. Nach Standardisierung tragen beide Features gleichermaßen zur Distanz bei, was zu sinnvolleren Clusterungen führt. Feature-Skalierung ist daher entscheidend für Distanz-basierte Algorithmen wie K-Means.
