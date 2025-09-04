"""
Logistic Regression with Cross-Validation

This script demonstrates how to use LogisticRegressionCV to find the best regularization
parameter C using cross-validation on a synthetic dataset.

Dependencies:
    - numpy
    - scikit-learn

Usage:
    python logistic_regression_cv.py
"""
import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Generate dataset
X, y = make_classification(n_samples=500, n_features=10, n_informative=6, n_redundant=4, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Train Logistic Regression with CV
logreg_cv = LogisticRegressionCV(cv=5, max_iter=1000)
logreg_cv.fit(X_train, y_train)

# Evaluate
y_pred = logreg_cv.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {acc:.3f}")
print(f"Best C values: {logreg_cv.C_}")
