"""
MLP Hyperparameter Tuning Example

This script demonstrates hyperparameter tuning for an MLPClassifier using GridSearchCV. It
searches over the number of hidden layer neurons and activation functions.

Dependencies:
    - numpy
    - scikit-learn

Usage:
    python mlp_hyperparam_tuning.py
"""
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# Generate data
X, y = make_classification(n_samples=500, n_features=20, n_informative=15, n_redundant=5, random_state=42)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Define parameter grid
param_grid = {
    'hidden_layer_sizes': [(50,), (100,), (50, 50)],
    'activation': ['tanh', 'relu'],
}

# Initialize model
mlp = MLPClassifier(max_iter=200, random_state=0)

# Grid search
cv = GridSearchCV(mlp, param_grid, cv=3, n_jobs=-1)
cv.fit(X_train, y_train)

# Evaluate
y_pred = cv.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {acc:.3f}")
print("Best parameters:", cv.best_params_)
