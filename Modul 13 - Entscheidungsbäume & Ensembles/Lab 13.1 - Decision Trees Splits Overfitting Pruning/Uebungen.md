# Lab 13.1 – Decision Trees: Splits, Overfitting, Pruning

In diesem Lab erkunden Sie Entscheidungsbäume, ihre Aufteilungskriterien, Overfitting und Verfahren zur Begrenzung der Baumtiefe. Verwenden Sie `pandas`, `scikit-learn` und `matplotlib`.

## Aufgabe 1 – Entscheidungsbaum auf einem synthetischen Datensatz

1. Erzeugen Sie mit `make_moons` aus `sklearn.datasets` einen Datensatz mit zwei Merkmalen und 300 Beobachtungen (`noise=0.25`, `random_state=42`).
2. Teilen Sie die Daten in Trainings- und Testdaten (70/30).
3. Trainieren Sie einen `DecisionTreeClassifier` ohne Begrenzung der Tiefe (`max_depth=None`).
4. Visualisieren Sie die Entscheidungsgrenze im 2D-Raum. Verwenden Sie dazu ein Gitter aus Punkten und `predict` für jedes Gitterelement, um die Klassenbereiche zu kennzeichnen.
5. Berechnen Sie Accuracy auf Trainings- und Testdaten. Beobachten Sie, ob Overfitting vorliegt.

## Aufgabe 2 – Einfluss der Baumtiefe

1. Verwenden Sie denselben Datensatz wie in Aufgabe 1.
2. Trainieren Sie mehrere Bäume mit unterschiedlichen `max_depth`-Werten, z. B. 1, 3, 5, 7, 9.
3. Für jeden Baum berechnen Sie die Accuracy für Training und Test.
4. Stellen Sie die Ergebnisse in einem Liniendiagramm dar (x-Achse: `max_depth`, y-Achse: Accuracy für Training und Test). Diskutieren Sie, bei welchen Tiefen der Baum zu einfach oder zu komplex ist.

## Aufgabe 3 – Kostenkomplexitätspruning

1. Nutzen Sie den Datensatz aus Aufgabe 1.
2. Verwenden Sie die Methode `cost_complexity_pruning_path` des trainierten Baums, um mögliche `ccp_alpha`-Werte zu erhalten.
3. Trainieren Sie für eine Auswahl von `ccp_alpha`-Werten (z. B. die ersten 5 unterschiedlichen Werte) jeweils einen neuen Baum und berechnen Sie die Accuracy auf den Testdaten.
4. Welcher `ccp_alpha` führt zu einem guten Kompromiss zwischen Modellkomplexität und Generalisierungsleistung?
