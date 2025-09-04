# Lab 11.3 – Hyperparameter‑Tuning (Grid‑ und Randomized Search)

In diesem Lab lernen Sie, wie man die Hyperparameter von Modellen mithilfe von Raster‑ (Grid‑) und Zufalls‑ (Randomized‑) Suche optimiert. Verwenden Sie die Werkzeuge `GridSearchCV` und `RandomizedSearchCV` aus `sklearn.model_selection`.

## Aufgabe 1 – KNN‑Klassifikation optimieren

1. Laden Sie den Iris‑Datensatz und teilen Sie ihn in Merkmale `X` und Ziel `y`.
2. Definieren Sie einen KNN‑Klassifikator (`KNeighborsClassifier`) als Basismodell.
3. Erstellen Sie einen Parameter‑Grid für die Anzahl der Nachbarn: `{'n_neighbors': [1,3,5,7,9,11]}`.
4. Verwenden Sie `GridSearchCV` mit 5‑facher Kreuzvalidierung (`cv=5`), um das beste `n_neighbors` zu bestimmen. Geben Sie den besten Parameter und die zugehörige mittlere Accuracy aus.

## Aufgabe 2 – Ridge‑Regression mit Grid Search

1. Erzeugen Sie einen Regressionsdatensatz mit `make_regression` (200 Beobachtungen, 5 Features, `noise=15`, `random_state=0`).
2. Definieren Sie ein Ridge‑Regressionsmodell (`Ridge`).
3. Erstellen Sie einen Parameter‑Grid für `alpha`, z. B. `[0.01, 0.1, 1.0, 10.0, 100.0]`.
4. Führen Sie `GridSearchCV` mit 5‑facher Kreuzvalidierung durch, wobei als Scoring der negative mittlere quadratische Fehler (`neg_mean_squared_error`) verwendet wird. Welcher Wert für `alpha` liefert den geringsten MSE?

## Aufgabe 3 – Randomized Search für einen Random Forest

1. Laden Sie den Datensatz `load_breast_cancer` aus `sklearn.datasets`.
2. Definieren Sie einen `RandomForestClassifier`.
3. Legen Sie einen Parameter‑Raum fest:  
   - `n_estimators`: [50, 100, 200]  
   - `max_depth`: [None, 5, 10, 20]  
   - `max_features`: ['sqrt', 'log2', None]
4. Verwenden Sie `RandomizedSearchCV` mit 20 Iterationen und 5‑facher Kreuzvalidierung (`cv=5`), um den besten Parameter‑Satz zu finden. Als Scoring kann `accuracy` verwendet werden.
5. Geben Sie die besten Parameter und die dazugehörige mittlere Accuracy aus.
