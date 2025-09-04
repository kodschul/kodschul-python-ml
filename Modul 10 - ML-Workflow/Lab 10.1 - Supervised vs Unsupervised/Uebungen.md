# Lab 10.1 – Supervised vs. Unsupervised Learning

In diesem Lab beschäftigen Sie sich mit der Unterscheidung zwischen überwachtem und unüberwachtem Lernen. Verwenden Sie für alle Aufgaben Python (z. B. in einem Notebook) sowie die Bibliotheken `pandas`, `matplotlib` und `scikit‑learn`.

## Aufgabe 1 – KMeans‐Clustering ohne Labels

1. Laden Sie den Iris‑Datensatz aus `sklearn.datasets` und speichern Sie ihn in einem `pandas.DataFrame`. Entfernen Sie die Spalten mit den Art‑Labels („target“).
2. Führen Sie ein KMeans‑Clustering mit `n_clusters=3` auf den Merkmalen durch.
3. Visualisieren Sie die Cluster, indem Sie zwei der vier Merkmale (z. B. `sepal_length` und `sepal_width`) in einem Streudiagramm darstellen und die Punkte nach Cluster färben.
4. Vergleichen Sie die KMeans‑Cluster mit den echten Arten (die Sie kurzzeitig wieder einfügen können). Erstellen Sie dazu eine Kreuztabelle (`pd.crosstab`) von Cluster vs. Art.

## Aufgabe 2 – Überwachtes Lernen mit einem Entscheidungsbaum

1. Laden Sie den Iris‑Datensatz erneut, diesmal mit den zugehörigen Labels.
2. Teilen Sie die Daten mit `train_test_split` (80 % Training, 20 % Test; `random_state=42`) in Trainings‑ und Testdatensatz.
3. Trainieren Sie einen `DecisionTreeClassifier` auf dem Trainingsteil und treffen Sie Vorhersagen auf dem Testteil.
4. Berechnen Sie die Accuracy sowie eine Confusion Matrix für die Testvorhersagen.

## Aufgabe 3 – Szenarien klassifizieren

Ordnen Sie die folgenden Problemstellungen zu: Handelt es sich um **überwachtes Lernen** oder **unüberwachtes Lernen**? Begründen Sie kurz Ihre Entscheidung.

1. Ein Versicherungsunternehmen möchte Neukunden in homogene Segmente einteilen, ohne vorher zu wissen, welche Gruppen existieren.
2. Ein Online‑Shop will vorhersagen, wie hoch der Umsatz eines Kunden im nächsten Monat sein wird, basierend auf seinen bisherigen Einkäufen.
3. Ein Krankenhaus möchte aus Patientendaten automatisch Alarm schlagen, wenn ungewöhnliche Muster auftreten, die auf einen möglichen Fehler im Messsystem hinweisen.
