# Lab 12.1 – K‑Nearest Neighbors: Bias/Varianz und Skalierungseffekte

Der K‑Nearest‑Neighbors‑Algorithmus (KNN) ist sensitiv gegenüber der Wahl des Parameters *k* sowie der Skalierung der Features. In diesem Lab untersuchen Sie diese Effekte.

## Aufgabe 1 – Bias‑Varianz‑Tradeoff anhand verschiedener *k*

1. Laden Sie den Iris‑Datensatz und teilen Sie ihn in Training und Test (80/20, `random_state=42`, mit `stratify=y`).
2. Trainieren Sie mehrere `KNeighborsClassifier`‑Modelle mit unterschiedlichen `k`‑Werten (z. B. 1, 3, 5, 7, 9, 11, 15).
3. Für jedes `k` berechnen Sie die Accuracy auf dem Trainingsdatensatz und auf dem Testdatensatz.
4. Stellen Sie die Ergebnisse in einem Liniendiagramm dar (x‑Achse: `k`, y‑Achse: Accuracy). Diskutieren Sie, bei welchen `k`‑Werten Overfitting (hohe Trainings‑Accuracy, niedrige Test‑Accuracy) bzw. Underfitting (beides niedrig) auftritt.

## Aufgabe 2 – Skalierungseffekte bei KNN

1. Erzeugen Sie einen kleinen Datensatz mit zwei Merkmalen auf sehr unterschiedlichen Skalen, z. B.:

   ```text
   Länge_cm   Gewicht_kg   Klasse
   150        50           0
   160        55           0
   170        65           0
   180        80           1
   190        90           1
   200        100          1
   ```

2. Teilen Sie den Datensatz in Training und Test (z. B. 4 Trainings‑ und 2 Testbeispiele).
3. Trainieren Sie einen `KNeighborsClassifier` mit `k=3` ohne vorherige Skalierung. Notieren Sie die Test‑Accuracy.
4. Skalieren Sie die Features mit `StandardScaler` (Mittelwert 0, Standardabweichung 1) und trainieren Sie denselben KNN erneut. Vergleichen Sie die neue Accuracy mit der unskalierten Version.
5. Erklären Sie, warum die Skalierung einen Einfluss hat.

## Aufgabe 3 – KNN‑Regression

1. Erzeugen Sie mit `make_regression` einen Datensatz mit 100 Beobachtungen, 1 Feature (`noise=20`, `random_state=1`).
2. Teilen Sie die Daten in Training und Test (70/30).
3. Trainieren Sie einen `KNeighborsRegressor` für verschiedene `k`‑Werte (z. B. 1, 3, 5, 7, 9).
4. Berechnen Sie den RMSE auf dem Testdatensatz für jeden `k`. In welchem Bereich liegt der beste Kompromiss zwischen Bias und Varianz?
