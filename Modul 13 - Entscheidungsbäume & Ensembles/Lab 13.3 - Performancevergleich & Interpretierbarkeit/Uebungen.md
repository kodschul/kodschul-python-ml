# Lab 13.3 – Performancevergleich & Interpretierbarkeit

In diesem Lab vergleichen Sie die Performance von Entscheidungsbäumen und Ensemble-Modellen und untersuchen deren Interpretierbarkeit.

## Aufgabe 1 – Modellvergleich auf einem Datensatz

1. Erzeugen Sie mit `make_classification` einen Datensatz mit 1000 Beobachtungen, 8 Features (`n_informative=6`, `random_state=42`).
2. Teilen Sie die Daten in Training und Test (70/30).
3. Trainieren Sie drei Modelle: `DecisionTreeClassifier` (ohne max_depth-Beschränkung), `RandomForestClassifier` (100 Bäume) und `GradientBoostingClassifier` (100 Iterationen).
4. Berechnen Sie für jedes Modell Accuracy, Precision, Recall und F1‑Score auf dem Testdatensatz. Halten Sie die Werte in einer Tabelle fest.
5. Diskutieren Sie die Ergebnisse: Welches Modell erzielt die beste Performance? Welches überfitten am stärksten?

## Aufgabe 2 – Interpretierbarkeit eines Entscheidungsbaums

1. Trainieren Sie einen `DecisionTreeClassifier` mit `max_depth=3` auf dem Datensatz aus Aufgabe 1.
2. Visualisieren Sie den Baum mit `plot_tree` aus `sklearn.tree` und speichern Sie den Plot als Bilddatei.
3. Analysieren Sie die Pfade im Baum: Welche Merkmale und Schwellenwerte werden verwendet? Wie lauten die Blätter (Vorhersagen) für zwei unterschiedliche Pfade?
4. Diskutieren Sie, warum ein flacher Baum leichter zu interpretieren ist als ein tiefer Baum.

## Aufgabe 3 – Permutation Importance

1. Nutzen Sie das Random‑Forest‑Modell aus Aufgabe 1.
2. Berechnen Sie die Permutation Importance mit `permutation_importance` (Scoring: 'accuracy', `n_repeats=10`, `random_state=0`) auf den Testdaten.
3. Zeigen Sie die fünf wichtigsten Features in einer Grafik und vergleichen Sie sie mit den `feature_importances_` aus dem Modell selbst.
4. Warum kann Permutation Importance ein robusteres Bild der Merkmalsbedeutung liefern als die integrierte Feature Importance?
