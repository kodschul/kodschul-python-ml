"""
Random Forest Example: Classification with Feature Importance

This script trains a Random Forest classifier on a synthetic dataset, evaluates its
accuracy on a test set, and plots feature importances. The example illustrates how
ensemble methods can improve prediction performance over a single decision tree.

Usage:
    python random_forest_example.py

Dependencies:
    - numpy
    - matplotlib
    - scikit-learn
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Generate a synthetic classification dataset
def generate_data(n_samples=500, random_state=42):
    X, y = make_classification(
        n_samples=n_samples,
        n_features=5,
        n_informative=5,
        n_redundant=0,
        n_clusters_per_class=1,
        class_sep=1.5,
        random_state=random_state,
    )
    return X, y

# Train a Random Forest and evaluate accuracy
def train_random_forest(X, y, n_estimators=100, max_depth=5):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=0)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {acc:.3f}")
    return rf

# Plot feature importances
def plot_feature_importance(model, feature_names=None):
    importances = model.feature_importances_
    if feature_names is None:
        feature_names = [f"Feature {i}" for i in range(len(importances))]
    indices = np.argsort(importances)[::-1]
    plt.figure(figsize=(8, 4))
    plt.bar(range(len(importances)), importances[indices], align='center')
    plt.xticks(range(len(importances)), [feature_names[i] for i in indices])
    plt.ylabel('Importance')
    plt.title('Random Forest Feature Importances')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    X, y = generate_data()
    rf_model = train_random_forest(X, y)
    plot_feature_importance(rf_model)
