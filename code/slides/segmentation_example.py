"""
Customer Segmentation with K-Means

This script demonstrates how to perform customer segmentation using K-Means clustering on a small
customer dataset. The dataset includes age, income, and spending score. The resulting segments
can help businesses tailor marketing strategies.

Dependencies:
    - pandas
    - scikit-learn
    - matplotlib

Usage:
    python segmentation_example.py
"""
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Load data from CSV
def load_customer_data(path='data/customers_small.csv'):
    return pd.read_csv(path)

# Perform K-Means segmentation
def segment_customers(df, n_clusters=3):
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    df['Segment'] = kmeans.fit_predict(df[['Age', 'Income', 'SpendingScore']])
    return kmeans, df

# Plot segments

def plot_segments(df, kmeans):
    plt.figure(figsize=(8, 4))
    for segment in df['Segment'].unique():
        cluster_data = df[df['Segment'] == segment]
        plt.scatter(cluster_data['Age'], cluster_data['SpendingScore'], label=f'Segment {segment}')
    centers = kmeans.cluster_centers_
    plt.scatter(centers[:, 0], centers[:, 2], c='black', marker='x', s=100, label='Centroids')
    plt.xlabel('Age')
    plt.ylabel('Spending Score')
    plt.title('Customer Segmentation')
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    df = load_customer_data()
    kmeans, df_segmented = segment_customers(df, n_clusters=3)
    plot_segments(df_segmented, kmeans)
