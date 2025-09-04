# Lab 15.2 – Einführung in Deep Learning: einfache dichte Netze

Obwohl wir hier keine Deep-Learning-Bibliothek wie Keras zur Verfügung haben, können wir neuronale Netze mit dem `MLPClassifier` aus `scikit-learn` erkunden. Dieses Modell ist ein Feedforward-Netz mit einem oder mehreren vollständig verbundenen (dichten) Schichten.

## Aufgabe 1 – MLP-Klassifikation auf synthetischen Daten

1. Erzeugen Sie mit `make_classification` einen Datensatz mit 600 Beobachtungen, 20 Features und 2 Klassen (`n_informative=15`, `random_state=0`).
2. Teilen Sie die Daten in Training und Test (70/30).
3. Trainieren Sie einen `MLPClassifier` mit einer versteckten Schicht von 10 Neuronen (`hidden_layer_sizes=(10,)`), der Aktivierungsfunktion `'relu'` und dem Optimierer `'adam'`.
4. Zeichnen Sie den Lernverlauf (Loss-Kurve) über die Trainingsiterationen hinweg (verwenden Sie `clf.loss_curve_`).
5. Berechnen Sie Accuracy und F1‑Score auf dem Testdatensatz.

## Aufgabe 2 – Vergleich mit Logistischer Regression

1. Trainieren Sie einen `LogisticRegression`-Classifier auf demselben Datensatz aus Aufgabe 1.
2. Vergleichen Sie die Test-Accuracy und F1‑Scores des MLP und der logistischen Regression.
3. Diskutieren Sie, warum das MLP bei komplexeren Datensätzen Vorteile bieten kann.

## Aufgabe 3 – MLP-Regression

1. Erzeugen Sie mit `make_regression` einen Datensatz mit 500 Beobachtungen, 5 Features (`noise=15`, `random_state=1`).
2. Trainieren Sie einen `MLPRegressor` mit zwei versteckten Schichten, z. B. `(50, 20)`.
3. Berechnen Sie den R‑Quadrat-Wert und den RMSE auf einem Testsplit (30 % Test).
4. Vergleichen Sie die Leistung des MLP mit einer klassischen `LinearRegression` auf demselben Datensatz.
