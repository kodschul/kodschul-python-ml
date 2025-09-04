"""
Introduction to Deep Learning with scikit-learn MLPClassifier

This script demonstrates using a Multi-Layer Perceptron (MLP) for classification using scikit-learn.
It trains an MLP on a synthetic dataset and compares its performance to logistic regression.

Dependencies:
    - numpy
    - scikit-learn
    - matplotlib

Usage:
    python deep_learning_example.py
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Generate synthetic dataset
def generate_data(n_samples=600, random_state=42):
    X, y = make_classification(
        n_samples=n_samples,
        n_features=10,
        n_informative=8,
        n_redundant=2,
        n_classes=2,
        random_state=random_state,
    )
    return X, y

# Train MLP and Logistic Regression

def train_models(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    mlp = MLPClassifier(hidden_layer_sizes=(50, 50), max_iter=100, random_state=0)
    logreg = LogisticRegression(max_iter=1000)
    mlp.fit(X_train, y_train)
    logreg.fit(X_train, y_train)
    y_pred_mlp = mlp.predict(X_test)
    y_pred_logreg = logreg.predict(X_test)
    acc_mlp = accuracy_score(y_test, y_pred_mlp)
    acc_logreg = accuracy_score(y_test, y_pred_logreg)
    print(f"MLP Test Accuracy: {acc_mlp:.3f}")
    print(f"Logistic Regression Test Accuracy: {acc_logreg:.3f}")
    return mlp, logreg

if __name__ == '__main__':
    X, y = generate_data()
    train_models(X, y)
