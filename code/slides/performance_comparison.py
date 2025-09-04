"""
Performance Comparison: Decision Tree vs. Random Forest vs. Gradient Boosting

This script compares the performance of three popular ensemble and tree-based models:
- DecisionTreeClassifier
- RandomForestClassifier
- GradientBoostingClassifier

It reports accuracy on a held-out test set and shows a bar chart of the results.

Dependencies:
    - scikit-learn
    - matplotlib

Usage:
    python performance_comparison.py
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# Generate a dataset
def generate_data(n_samples=500, random_state=42):
    X, y = make_classification(
        n_samples=n_samples,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        random_state=random_state,
    )
    return X, y

# Train models and return accuracies
def compare_models(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    models = {
        'Decision Tree': DecisionTreeClassifier(max_depth=None, random_state=0),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=0),
        'Gradient Boosting': GradientBoostingClassifier(random_state=0),
    }
    accuracies = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracies[name] = accuracy_score(y_test, y_pred)
        print(f"{name} Test Accuracy: {accuracies[name]:.3f}")
    return accuracies

# Plot a bar chart of accuracies
def plot_accuracies(accuracies):
    labels = list(accuracies.keys())
    scores = list(accuracies.values())
    plt.figure(figsize=(8, 4))
    plt.bar(labels, scores)
    plt.ylim(0, 1)
    plt.ylabel('Accuracy')
    plt.title('Model Performance Comparison')
    for i, score in enumerate(scores):
        plt.text(i, score + 0.02, f"{score:.2f}", ha='center')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    X, y = generate_data()
    acc = compare_models(X, y)
    plot_accuracies(acc)
