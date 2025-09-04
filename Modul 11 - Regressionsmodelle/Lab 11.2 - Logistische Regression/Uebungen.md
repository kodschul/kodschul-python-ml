# Lab 11.2 – Logistische Regression: Klassifikation und Entscheidungsgrenzen

Die logistische Regression ist ein wichtiges Modell zur Vorhersage binärer Ereignisse. In diesem Lab lernen Sie, ein logistische‑Regression‑Modell zu trainieren, die Entscheidungsgrenze zu visualisieren und Klassifikationsmetriken zu interpretieren.

## Aufgabe 1 – Datensatz erzeugen und Modelltraining

1. Verwenden Sie `make_classification` aus `sklearn.datasets`, um einen zweidimensionalen binären Datensatz mit 200 Beobachtungen zu erzeugen (Parameter: `n_features=2`, `n_informative=2`, `n_redundant=0`, `random_state=0`). Speichern Sie die Daten in `X`, die Labels in `y`.
2. Teilen Sie den Datensatz mit `train_test_split` (70 % Training, 30 % Test, `random_state=0`).
3. Trainieren Sie ein `LogisticRegression`‑Modell und geben Sie die gelernten Koeffizienten und den Achsenabschnitt aus.
4. Bewerten Sie das Modell auf den Testdaten: Berechnen Sie Accuracy, Precision, Recall und F1‑Score.

## Aufgabe 2 – Visualisierung der Entscheidungsgrenze

1. Erstellen Sie ein Streudiagramm der beiden Trainingsfeatures, wobei Sie die Klassen unterschiedlich einfärben.
2. Erstellen Sie anschließend eine feinmaschige Gittermatrix aus Werten für die beiden Features (z. B. mittels `np.meshgrid`) und lassen Sie das trainierte Modell die Wahrscheinlichkeit für Klasse 1 auf diesem Gitter vorhersagen.
3. Zeichnen Sie die Entscheidungsgrenze (Konturlinie für Wahrscheinlichkeit 0.5) sowie die farbig eingefärbte Entscheidungsfläche (Probability Surface). Zeichnen Sie die Trainingspunkte darüber.

## Aufgabe 3 – Schwellenwertanpassung und ROC‑Kurve

1. Erzeugen Sie mit Ihrem trainierten Modell die vorhergesagten Wahrscheinlichkeiten für die Testdaten (`predict_proba`).
2. Setzen Sie einen alternativen Schwellenwert von 0.7, um die Wahrscheinlichkeiten in Klassenlabels umzuwandeln. Berechnen Sie erneut Accuracy, Precision, Recall und F1‑Score und vergleichen Sie diese mit dem Standard‑Schwellenwert von 0.5.
3. Zeichnen Sie die ROC‑Kurve und berechnen Sie die **AUC** (Area Under the Curve). Verwenden Sie `roc_curve` und `roc_auc_score` aus `sklearn.metrics`. Interpretieren Sie, was eine höhere AUC bedeutet.
