# Lösungen zu Lab 14.2 – Anwendungsfälle: Segmentierung und Vorstrukturierung

## Lösung zu Aufgabe 1 – Kundensegmentierung

```python
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# 1. Künstlichen Datensatz erzeugen
np.random.seed(42)
n = 200
data = pd.DataFrame({
    'Alter': np.random.randint(18, 70, size=n),
    'Einkommen': np.random.randint(20000, 120000, size=n),
    'Ausgaben_Luxus': np.random.randint(0, 5000, size=n),
    'Ausgaben_Notwendiges': np.random.randint(500, 3000, size=n),
    'Online_Aktivität': np.random.rand(n)
})

# 2. K-Means Clustering
kmeans = KMeans(n_clusters=3, random_state=42)
data['Cluster'] = kmeans.fit_predict(data)

# 3. Visualisierung (Einkommen vs. Luxusausgaben)
plt.scatter(data['Einkommen'], data['Ausgaben_Luxus'], c=data['Cluster'], cmap='viridis')
plt.xlabel('Einkommen')
plt.ylabel('Ausgaben für Luxus')
plt.title('Kundensegmente')
plt.show()

# 4. Durchschnittswerte pro Cluster
cluster_summary = data.groupby('Cluster').mean()
print(cluster_summary)
```

Die Gruppen können z. B. als „geringes Einkommen / geringe Ausgaben“, „mittleres Einkommen / moderate Ausgaben“ und „hohes Einkommen / hohe Luxusausgaben“ interpretiert werden. Solche Segmente unterstützen zielgerichtetes Marketing.

## Lösung zu Aufgabe 2 – Cluster als Feature für eine Klassifikation

```python
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
import numpy as np
import pandas as pd

# 1. Iris-Daten laden
iris = load_iris()
X = iris.data
y = iris.target

# 2. K-Means und Clusterlabels erzeugen
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X)

# 3. Modell ohne Clusterfeature
knn = KNeighborsClassifier(n_neighbors=5)
scores_no_cluster = cross_val_score(knn, X, y, cv=5, scoring='f1_macro')

# 4. Modell mit Clusterfeature
X_aug = np.column_stack((X, clusters))
scores_with_cluster = cross_val_score(knn, X_aug, y, cv=5, scoring='f1_macro')

print('F1-Score ohne Clusterfeature: {:.3f} ± {:.3f}'.format(scores_no_cluster.mean(), scores_no_cluster.std()))
print('F1-Score mit Clusterfeature:  {:.3f} ± {:.3f}'.format(scores_with_cluster.mean(), scores_with_cluster.std()))
```

In diesem Beispiel verbessert das zusätzliche Clusterfeature die Leistung nur minimal, da K-Means die gleichen Gruppen erkennt wie die Zielvariablen selbst. In anderen Datensätzen kann ein derartiges Feature zusätzliche Struktur einbringen und die Vorhersage verbessern.

## Lösung zu Aufgabe 3 – Dimensionality Reduction + Clustering

```python
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
import matplotlib.pyplot as plt
import numpy as np

# 1. Datensatz laden
digits = load_digits()
X = digits.data
y = digits.target

# 2. PCA-Reduktion auf zwei Komponenten
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X)

# 3. K-Means Clustering mit k=10
kmeans = KMeans(n_clusters=10, random_state=42)
clusters = kmeans.fit_predict(X_pca)

# 4. Visualisierung
plt.figure(figsize=(8,6))
plt.scatter(X_pca[:,0], X_pca[:,1], c=clusters, cmap='tab10', s=15)
plt.xlabel('PCA Komponente 1')
plt.ylabel('PCA Komponente 2')
plt.title('K-Means Clustering auf PCA-reduzierten Digits')
plt.show()

# Adjusted Rand Score
ars = adjusted_rand_score(y, clusters)
print(f'Adjusted Rand Score: {ars:.3f}')
```

Der Adjusted Rand Score (ARS) vergleicht die Übereinstimmung zwischen den gefundenen Clustern und den tatsächlichen Ziffern. Ein Wert von 1 bedeutet perfekte Übereinstimmung, 0 entspricht Zufall. In diesem Fall liegt der ARS typischerweise im Bereich 0.3–0.5, da K-Means auf den PCA-Komponenten nicht alle Feinheiten der handgeschriebenen Ziffern erfasst.
