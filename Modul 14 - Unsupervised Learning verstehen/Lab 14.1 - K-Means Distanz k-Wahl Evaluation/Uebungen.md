# Lab 14.1 – K-Means Clustering: Distanzmaße, k-Wahl, Evaluation

Dieses Lab vermittelt praktische Erfahrung mit K‑Means‑Clustering, der Wahl von k und der Bewertung von Clustern.

## Aufgabe 1 – K-Means auf synthetischen Daten

1. Erzeugen Sie mit `make_blobs` einen Datensatz mit 400 Beobachtungen, 2 Merkmalen und drei klar getrennten Gruppen (`centers=3`, `cluster_std=0.60`, `random_state=0`).
2. Führen Sie K-Means-Clustering mit `n_clusters=3` durch.
3. Visualisieren Sie die Datenpunkte und die Clusterzentren.
4. Berechnen Sie den Inertia-Wert (`inertia_`) und den Silhouette-Score (`silhouette_score`).

## Aufgabe 2 – k wählen mit Elbow und Silhouette

1. Verwenden Sie den Datensatz aus Aufgabe 1.
2. Berechnen Sie für k = 1 bis 10 den Inertia-Wert und zeichnen Sie das Elbow-Diagramm (k gegen Inertia).
3. Berechnen Sie für k = 2 bis 10 den durchschnittlichen Silhouette-Score und zeichnen Sie ihn (k gegen Silhouette).
4. Welche k-Werte scheinen laut beiden Methoden sinnvoll? Begründen Sie.

## Aufgabe 3 – Skalierung und Distanzmaß

1. Erzeugen Sie einen Datensatz mit zwei Merkmalen auf unterschiedlichen Skalen (z. B. `[0, 1000]` und `[0, 1]`) und drei Clustergruppen (k = 3).
2. Führen Sie K-Means einmal ohne Skalierung und einmal nach Standardisierung der Features (`StandardScaler`) durch.
3. Vergleichen Sie die Clusterzuweisungen sowie Inertia- und Silhouette-Werte.
4. Warum ist Feature-Skalierung wichtig für Distanz-basierte Methoden wie K-Means?
