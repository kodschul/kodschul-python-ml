# Lab 13.2 – Random Forests: Ensembles, Feature-Bedeutung

In diesem Lab lernen Sie, wie Ensemble-Methoden wie der Random Forest funktionieren und welche Vorteile sie gegenüber einzelnen Entscheidungsbäumen haben. Außerdem betrachten Sie die Wichtigkeit von Features.

## Aufgabe 1 – Random Forest auf einem Klassifikationsdatensatz

1. Erzeugen Sie mit `make_classification` einen Datensatz mit 500 Beobachtungen, 10 Features, wovon 5 informativ sind (`n_informative=5`, `n_redundant=2`, `random_state=0`).
2. Teilen Sie die Daten in Training (70 %) und Test (30 %).
3. Trainieren Sie einen `RandomForestClassifier` mit 100 Bäumen (`n_estimators=100`) und Standardparametern (`random_state=0`).
4. Berechnen Sie Accuracy auf Training und Test. Vergleichen Sie mit einem einzelnen `DecisionTreeClassifier` (z. B. `max_depth=None`) und besprechen Sie den Unterschied im Overfitting.

## Aufgabe 2 – Feature Importance analysieren

1. Nutzen Sie den in Aufgabe 1 trainierten Random Forest.
2. Lassen Sie sich die `feature_importances_` ausgeben und sortieren Sie die Features nach Wichtigkeit.
3. Visualisieren Sie die wichtigsten fünf Features in einem horizontalen Balkendiagramm.
4. Was sagen hohe Wichtigkeitswerte über den Einfluss eines Features auf das Modell aus?

## Aufgabe 3 – Ensemble-Vergleich: Bagging vs. Boosting

1. Verwenden Sie denselben Datensatz wie in Aufgabe 1.
2. Trainieren Sie einen `BaggingClassifier` mit Entscheidungsbäumen als Basismodelle (z. B. 50 Bäume).
3. Trainieren Sie außerdem einen `GradientBoostingClassifier` mit Standardparametern.
4. Vergleichen Sie die Accuracy der drei Modelle (Random Forest, Bagging, Gradient Boosting) mittels 5‑facher Kreuzvalidierung (`cross_val_score`). Welches Modell erzielt die beste mittlere Accuracy? Nennen Sie mögliche Gründe.
