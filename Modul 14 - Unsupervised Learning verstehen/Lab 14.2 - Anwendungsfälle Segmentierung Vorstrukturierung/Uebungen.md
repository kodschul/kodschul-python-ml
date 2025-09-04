# Lab 14.2 – Anwendungsfälle: Segmentierung und Vorstrukturierung

In diesem Lab wenden Sie unüberwachtes Lernen praktisch an, um Daten zu segmentieren und als Vorstufe für überwachte Modelle zu nutzen.

## Aufgabe 1 – Kundensegmentierung

1. Erstellen Sie einen künstlichen Datensatz mit `pandas`, der fünf Merkmale pro Kunde enthält: `Alter`, `Einkommen`, `Ausgaben_Luxus`, `Ausgaben_Notwendiges`, `Online_Aktivität`. Erzeugen Sie 200 zufällige Zeilen mit realistischen Werten (verwenden Sie z. B. `numpy.random`).
2. Führen Sie K-Means-Clustering mit k = 3 durch und fügen Sie das Clusterlabel dem DataFrame hinzu.
3. Visualisieren Sie die Kunden in zwei geeigneten Dimensionen (z. B. Einkommen vs. Ausgaben_Luxus) und färben Sie sie nach Cluster.
4. Charakterisieren Sie die Cluster: Berechnen Sie für jedes Cluster den Durchschnitt der Merkmale und interpretieren Sie die Segmente (z. B. „High Income / High Spending“-Gruppe).

## Aufgabe 2 – Cluster als Feature für eine Klassifikation

1. Verwenden Sie den Iris-Datensatz (Supervised Problem: Art zu erkennen).
2. Führen Sie K-Means mit k = 3 auf den kompletten Merkmalen durch und speichern Sie die Clusterlabels.
3. Trainieren Sie einen `KNeighborsClassifier` einmal nur mit den originalen Features und einmal mit den originalen Features plus dem neuen Clusterfeature.
4. Vergleichen Sie Accuracy und F1‑Score der beiden Modelle mittels 5‑facher Kreuzvalidierung. Bringt das Clusterfeature einen Vorteil?

## Aufgabe 3 – Dimensionality Reduction + Clustering

1. Laden Sie das `digits`-Dataset aus `sklearn.datasets` (Handgeschriebene Ziffern, 8×8 Pixel = 64 Features).
2. Reduzieren Sie die Dimensionalität mit `PCA` auf zwei Hauptkomponenten.
3. Führen Sie K-Means mit k = 10 (da es zehn Ziffern gibt) auf den reduzierten Daten durch.
4. Visualisieren Sie die zwei PCA-Komponenten, färben Sie die Punkte nach den Clusterlabels und vergleichen Sie diese mit den tatsächlichen Ziffern. Berechnen Sie den Adjusted Rand Score (`adjusted_rand_score`) zwischen Clustern und echten Labels.
