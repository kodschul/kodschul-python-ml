# Lösungen zu Lab 15.1 – Natural Language Processing: Text zu Features

## Lösung zu Aufgabe 1 – Tokenisierung und Vorverarbeitung

```python
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string

# Falls notwendig, Ressourcen laden
nltk.download('punkt')
nltk.download('stopwords')

dokumente = [
    "Das Wetter ist heute schön und sonnig.",
    "Morgen wird es regnen, bring einen Regenschirm mit.",
    "Der Hund spielt mit dem Ball im Park.",
    "Python ist eine vielseitige Programmiersprache.",
    "Maschinelles Lernen ermöglicht spannende Anwendungen.",
    "Ich habe einen Kuchen gebacken und er schmeckt lecker."
]

stop_words = set(stopwords.words('german'))
punct = set(string.punctuation)

def preprocess(sent):
    tokens = word_tokenize(sent)
    # in Kleinbuchstaben, Stopwörter und Interpunktion entfernen
    cleaned = [w.lower() for w in tokens if w.lower() not in stop_words and w not in punct]
    return cleaned

cleaned_docs = [preprocess(doc) for doc in dokumente]
print(cleaned_docs)
```

Die Listen enthalten nun nur noch bedeutungstragende Wörter in Kleinbuchstaben.

## Lösung zu Aufgabe 2 – Bag-of-Words und TF‑IDF

```python
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import pandas as pd

# Für CountVectorizer müssen Strings übergeben werden
docs_joined = [' '.join(doc) for doc in cleaned_docs]

# 1. Bag-of-Words
count_vec = CountVectorizer()
bow = count_vec.fit_transform(docs_joined)
bow_df = pd.DataFrame(bow.toarray(), columns=count_vec.get_feature_names_out())
print('Bag-of-Words Vokabular:', count_vec.get_feature_names_out())
print(bow_df)

# 2. TF-IDF
tfidf_vec = TfidfVectorizer()
tfidf = tfidf_vec.fit_transform(docs_joined)
tfidf_df = pd.DataFrame(tfidf.toarray(), columns=tfidf_vec.get_feature_names_out())
print(tfidf_df.head())
```

Der Bag‑of‑Words Ansatz zählt, wie oft jedes Wort vorkommt. TF‑IDF gewichtet seltene Wörter höher und häufige Wörter niedriger, wodurch oft wichtige Begriffe stärker hervortreten.

## Lösung zu Aufgabe 3 – Textklassifikation

```python
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

labels = [0, 0, 0, 1, 1, 0]  # Technik = 1, sonst = 0

# Text in Vektoren umwandeln (TF-IDF)
X = tfidf
y = labels

# Train/Test-Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42)

# Modell trainieren
clf = MultinomialNB()
clf.fit(X_train, y_train)

# Vorhersagen und Bewertung
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
print(f'Accuracy: {acc:.3f}, F1-Score: {f1:.3f}')
print('Vorhersagen:', y_pred)
```

Der Naive-Bayes-Klassifikator eignet sich gut für Textdaten. Fehlklassifikationen treten beispielsweise auf, wenn ein Satz sowohl technische als auch allgemeine Wörter enthält und der Modellkontext zu klein ist.
