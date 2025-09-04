"""
Product Purchase Data Segmentation

This example demonstrates how to segment customer purchase data using KMeans. It assumes
a dataset of purchase frequency, average spending, and number of products bought.

Dependencies:
    - pandas
    - scikit-learn

Usage:
    python segmentation_purchase_data.py
"""
import pandas as pd
from sklearn.cluster import KMeans

# Create a simple dataset manually
def create_purchase_data():
    data = {
        'Frequency': [5, 2, 8, 1, 7, 3, 6, 4],
        'AverageSpend': [100, 50, 150, 40, 140, 60, 120, 80],
        'ProductCount': [2, 1, 3, 1, 4, 2, 3, 2],
    }
    return pd.DataFrame(data)

# Perform segmentation
def segment_purchases(df, n_clusters=3):
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    df['Segment'] = kmeans.fit_predict(df)
    return kmeans, df


if __name__ == '__main__':
    df = create_purchase_data()
    kmeans, df_seg = segment_purchases(df, n_clusters=3)
    print(df_seg)
