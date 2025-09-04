"""
Hyperparameter Tuning for Random Forest using GridSearchCV

This script demonstrates how to perform hyperparameter tuning on a Random Forest classifier
using scikit-learn's GridSearchCV. It searches over a parameter grid and outputs the
best parameters and corresponding accuracy.

Dependencies:
    - numpy
    - scikit-learn

Usage:
    python random_forest_gridsearch.py
"""
import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score

# Create synthetic dataset
X, y = make_classification(n_samples=500, n_features=10, n_informative=8, n_redundant=2, random_state=42)

# Define parameter grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5],
}

# Initialize Random Forest
rf = RandomForestClassifier(random_state=0)

# Setup grid search
cv = GridSearchCV(rf, param_grid, cv=3, n_jobs=-1)
cv.fit(X, y)

print("Best Parameters:", cv.best_params_)
print(f"Best CV Accuracy: {cv.best_score_:.3f}")
