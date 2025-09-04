"""
Clustering Evaluation Metrics

This script illustrates how to evaluate the quality of clusters using metrics such as
Silhouette Score and Davies-Bouldin Index. It performs KMeans clustering on a synthetic
blobs dataset for a range of cluster numbers and prints the metrics.

Dependencies:
    - numpy
    - scikit-learn

Usage:
    python clustering_evaluation.py
"""
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score

# Generate data
X, _ = make_blobs(n_samples=600, centers=4, cluster_std=0.5, random_state=42)

# Evaluate metrics for different k
for k in range(2, 8):
    kmeans = KMeans(n_clusters=k, random_state=0)
    labels = kmeans.fit_predict(X)
    sil_score = silhouette_score(X, labels)
    db_score = davies_bouldin_score(X, labels)
    print(f"k = {k}: Silhouette Score = {sil_score:.3f}, Davies-Bouldin Index = {db_score:.3f}")
