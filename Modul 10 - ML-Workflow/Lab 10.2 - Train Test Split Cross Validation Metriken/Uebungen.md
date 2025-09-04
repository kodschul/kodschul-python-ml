# Lab 10.2 – Train/Test‑Split, Cross‑Validation und Metriken

In diesem Lab vertiefen Sie die Konzepte der Datenaufteilung, der Kreuzvalidierung und der Leistungskennzahlen für Regression und Klassifikation. Sie arbeiten mit künstlich erzeugten Beispieldaten und Standard‑Datasets aus `sklearn`.

## Aufgabe 1 – Train/Test‑Split bei einer Regressionsaufgabe

1. Erzeugen Sie mit `make_regression` aus `sklearn.datasets` einen Regressionsdatensatz mit 200 Beobachtungen, 1 Feature und etwas Rauschen (`noise=10.0`).
2. Teilen Sie die Daten in einen Trainings‑ und Testdatensatz (80 % Training, 20 % Test, `random_state=42`).
3. Trainieren Sie ein `LinearRegression`‑Modell auf dem Trainingsteil.
4. Bewerten Sie das Modell auf dem Testdatensatz mittels **R‑Quadrat** (Bestimmtheitsmaß) und **mittlerer quadratischer Fehler** (MSE).

## Aufgabe 2 – Klassifikationsmetriken und Confusion Matrix

1. Laden Sie den Iris‑Datensatz und teilen Sie ihn in Training und Test (80/20, `random_state=42`, mit `stratify=y`).
2. Trainieren Sie einen `KNeighborsClassifier` mit `n_neighbors=5`.
3. Erzeugen Sie Vorhersagen auf dem Testdatensatz und berechnen Sie die Metriken **Accuracy**, **Precision**, **Recall** und **F1‑Score** (verwenden Sie z. B. `classification_report` aus `sklearn.metrics`).
4. Stellen Sie die Confusion Matrix grafisch als Heatmap dar (verwenden Sie `seaborn.heatmap` oder `matplotlib`).

## Aufgabe 3 – K‑Fold Cross‑Validation

1. Verwenden Sie erneut den Iris‑Datensatz (ohne Testsplit) und führen Sie eine 5‑fache Kreuzvalidierung (`KFold`) für einen `DecisionTreeClassifier` durch.
2. Ermitteln Sie mit `cross_val_score` die Accuracy‑Werte in jedem Fold und den Durchschnitt.
3. Wiederholen Sie dasselbe für einen `SVC` (Support‑Vector‑Classifier) mit linearem Kernel (`kernel='linear'`) und vergleichen Sie die durchschnittliche Accuracy der beiden Modelle.
