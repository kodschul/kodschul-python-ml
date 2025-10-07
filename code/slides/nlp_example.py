"""
Natural Language Processing Example: Text Classification

This script demonstrates basic NLP steps: tokenization, vectorization with TF-IDF, and
training a Naive Bayes classifier on a small text dataset. It prints out the accuracy
and shows how to transform new text into feature vectors.

Dependencies:
    - pandas
    - scikit-learn

Usage:
    python nlp_example.py
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Load small text dataset
def load_texts(path='../data/tiny_texts.csv'):
    return pd.read_csv(path)

# Train classifier
def train_text_classifier(df):
    X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.3, random_state=0)
    vectorizer = TfidfVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    clf = MultinomialNB()
    clf.fit(X_train_vec, y_train)
    y_pred = clf.predict(X_test_vec)
    acc = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {acc:.3f}")
    return clf, vectorizer

# Predict new texts
def predict_new_texts(clf, vectorizer, texts):
    X_new_vec = vectorizer.transform(texts)
    return clf.predict(X_new_vec)

if __name__ == '__main__':
    df = load_texts()
    clf, vectorizer = train_text_classifier(df)
    sample_texts = ["Machine learning is amazing", "I love reading books"]
    preds = predict_new_texts(clf, vectorizer, sample_texts)
    for text, pred in zip(sample_texts, preds):
        print(f'\"{text}\" -> {pred}')
