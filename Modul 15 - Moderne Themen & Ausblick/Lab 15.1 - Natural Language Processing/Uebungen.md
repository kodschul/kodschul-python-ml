# Lab 15.1 – Natural Language Processing: Text zu Features

In diesem Lab verarbeiten Sie Text in deutsche Sätze, tokenisieren und vektorisieren ihn und trainieren anschließend ein einfaches Klassifikationsmodell.

## Aufgabe 1 – Tokenisierung und Vorverarbeitung

1. Gegeben ist eine Liste deutscher Sätze:  

   ```python
   dokumente = [
       "Das Wetter ist heute schön und sonnig.",
       "Morgen wird es regnen, bring einen Regenschirm mit.",
       "Der Hund spielt mit dem Ball im Park.",
       "Python ist eine vielseitige Programmiersprache.",
       "Maschinelles Lernen ermöglicht spannende Anwendungen.",
       "Ich habe einen Kuchen gebacken und er schmeckt lecker."
   ]
   ```
2. Verwenden Sie `nltk.tokenize.word_tokenize`, um jeden Satz zu tokenisieren.
3. Entfernen Sie Stopwörter (z. B. mit der deutschen Stopwort-Liste aus `nltk.corpus.stopwords`) und Interpunktionszeichen.
4. Wandeln Sie alle Wörter in Kleinbuchstaben um.

## Aufgabe 2 – Bag-of-Words und TF‑IDF

1. Nutzen Sie die vorverarbeiteten Dokumente aus Aufgabe 1.
2. Erzeugen Sie eine Bag-of-Words-Darstellung mit `CountVectorizer` aus `sklearn.feature_extraction.text`.
3. Erzeugen Sie außerdem eine TF‑IDF-Darstellung mit `TfidfVectorizer`.
4. Geben Sie die Wortliste (`vocabulary_`) und die resultierenden Matrizen (als DataFrame) aus. Was unterscheidet TF‑IDF von der reinen Häufigkeit?

## Aufgabe 3 – Textklassifikation

1. Erweitern Sie die Liste aus Aufgabe 1 um eine Zielvariable, z. B. `label = [0, 0, 0, 1, 1, 0]`, die angibt, ob ein Satz sich mit Technik (1) oder etwas anderem (0) beschäftigt.
2. Teilen Sie die Daten in Training und Test (z. B. 4 : 2).
3. Trainieren Sie einen `MultinomialNB`-Klassifikator (Naive Bayes) auf der TF‑IDF-Darstellung der Trainingsdaten.
4. Bewerten Sie die Accuracy und den F1‑Score auf dem Testdatensatz. Welche Fehlklassifikationen treten auf?
