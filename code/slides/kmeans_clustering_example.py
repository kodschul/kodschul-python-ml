"""
K-Means Clustering Example with Elbow and Silhouette Methods

This script demonstrates K-Means clustering on a synthetic 2D dataset. It uses
the elbow method and silhouette score to help decide the optimal number of
clusters.

Dependencies:
    - numpy
    - matplotlib
    - scikit-learn

Usage:
    python kmeans_clustering_example.py
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Generate synthetic 2D data
def generate_data(n_samples=500, random_state=42):
    X, _ = make_blobs(n_samples=n_samples, centers=4, cluster_std=0.60, random_state=random_state)
    return X

# Plot elbow method and silhouette scores
def evaluate_kmeans(X, max_k=10):
    inertias = []
    silhouettes = []
    K_range = range(2, max_k + 1)
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=0)
        labels = kmeans.fit_predict(X)
        inertias.append(kmeans.inertia_)
        silhouettes.append(silhouette_score(X, labels))
    return K_range, inertias, silhouettes

# Plot results
def plot_evaluation(K_range, inertias, silhouettes):
    fig, ax1 = plt.subplots(figsize=(8, 4))
    color = 'tab:blue'
    ax1.set_xlabel('Number of clusters (k)')
    ax1.set_ylabel('Inertia', color=color)
    ax1.plot(K_range, inertias, marker='o', color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color = 'tab:orange'
    ax2.set_ylabel('Silhouette Score', color=color)
    ax2.plot(K_range, silhouettes, marker='s', color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title('Elbow Method and Silhouette Score for K-Means')
    fig.tight_layout()
    plt.show()

if __name__ == '__main__':
    X = generate_data()
    K_range, inertias, silhouettes = evaluate_kmeans(X)
    plot_evaluation(K_range, inertias, silhouettes)
