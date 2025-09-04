"""
PCA + KMeans Pipeline Example

This script demonstrates how to reduce dimensionality using PCA and then apply KMeans
clustering on the reduced data. It uses scikit-learn's Pipeline and prints silhouette
scores for different numbers of clusters.

Dependencies:
    - numpy
    - scikit-learn

Usage:
    python pca_kmeans_pipeline.py
"""
import numpy as np
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Load digits dataset (handwritten digits)
X, y = load_digits(return_X_y=True)

# Create a pipeline with PCA and KMeans
pipeline = Pipeline([
    ('pca', PCA(n_components=20, random_state=0)),
    ('kmeans', KMeans(n_clusters=10, random_state=0)),
])

# Fit pipeline and evaluate silhouette score
pipeline.fit(X)
y_pred = pipeline.named_steps['kmeans'].labels_
sil_score = silhouette_score(X, y_pred)
print(f"Silhouette Score with 10 clusters: {sil_score:.3f}")
