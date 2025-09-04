"""
Decision Tree Classification Example

This script demonstrates how to train a simple decision tree classifier on a synthetic
2D dataset and visualize the decision boundary and the tree structure. It uses
scikit-learn's `DecisionTreeClassifier` and matplotlib for visualization.

Usage:
    python decision_tree_example.py

Dependencies:
    - numpy
    - matplotlib
    - scikit-learn

Note: Running this script will generate two plots: one showing the decision boundary
and another displaying the tree structure. These plots will appear sequentially.
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split

# Generate a synthetic 2D classification dataset
def generate_data(n_samples=200, random_state=42):
    X, y = make_classification(
        n_samples=n_samples,
        n_features=2,
        n_informative=2,
        n_redundant=0,
        n_clusters_per_class=1,
        class_sep=1.5,
        random_state=random_state,
    )
    return X, y

# Train a decision tree and visualize its decision boundary

def train_and_plot(X, y, max_depth=None):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    clf = DecisionTreeClassifier(max_depth=max_depth, random_state=0)
    clf.fit(X_train, y_train)
    
    # Plot decision boundary
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    plt.figure(figsize=(8, 4))
    plt.contourf(xx, yy, Z, alpha=0.3)
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, marker='o', edgecolor='k', label='Train')
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, marker='s', edgecolor='k', label='Test')
    plt.title('Decision Tree Decision Boundary')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Plot tree structure
    plt.figure(figsize=(10, 6))
    plot_tree(clf, filled=True, feature_names=['Feature 1', 'Feature 2'], class_names=['0', '1'])
    plt.title('Decision Tree Structure')
    plt.show()

    return clf


if __name__ == '__main__':
    X, y = generate_data()
    clf = train_and_plot(X, y, max_depth=3)
