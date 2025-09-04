"""
Class Imbalance Example: Handling Imbalanced Data with class_weight

This script shows how to address class imbalance in classification using the `class_weight`
parameter in scikit-learn's LogisticRegression. It generates an imbalanced dataset and
compares performance with and without class weighting.

Dependencies:
    - numpy
    - scikit-learn

Usage:
    python class_imbalance_example.py
"""
import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Generate imbalanced data
X, y = make_classification(
    n_samples=1000, n_features=20, n_informative=2, n_redundant=10,
    weights=[0.9, 0.1], random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Logistic Regression without class_weight
clf_no_weight = LogisticRegression(max_iter=1000)
clf_no_weight.fit(X_train, y_train)
print("Without class weight:")
print(classification_report(y_test, clf_no_weight.predict(X_test)))

# Logistic Regression with class_weight='balanced'
clf_weighted = LogisticRegression(max_iter=1000, class_weight='balanced')
clf_weighted.fit(X_train, y_train)
print("With class_weight='balanced':")
print(classification_report(y_test, clf_weighted.predict(X_test)))
