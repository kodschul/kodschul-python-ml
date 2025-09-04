# Lab 12.2 – Support Vector Machines (SVM): Kernel‑Idee & Margin

Support Vector Machines sind leistungsfähige Klassifikatoren, die versuchen, die Margin zwischen Klassen zu maximieren. In diesem Lab experimentieren Sie mit linearen und nichtlinearen SVMs und untersuchen den Einfluss von Hyperparametern.

## Aufgabe 1 – Lineare SVM und Margin

1. Erzeugen Sie mit `make_blobs` aus `sklearn.datasets` einen Datensatz mit zwei Klassen, die linear trennbar sind (`centers=2`, `cluster_std=1.0`, `random_state=1`).
2. Teilen Sie den Datensatz in Training und Test (80/20).
3. Trainieren Sie einen `SVC` mit linearem Kernel (`kernel='linear'`).
4. Visualisieren Sie die Trainingsdaten sowie die Entscheidungsgerade und die Margin. Sie können die Koordinaten der Stützvektoren (`svc.support_vectors_`) nutzen, um die Margin‑Linien zu zeichnen.

## Aufgabe 2 – Kernel‑Trick mit RBF

1. Erzeugen Sie einen nichtlinear trennbaren Datensatz, z. B. mit `make_circles` (`factor=0.5`, `noise=0.1`, `random_state=0`).
2. Trainieren Sie zwei SVM‑Modelle: eines mit linearem Kernel, eines mit radial basis function (RBF)‑Kernel (`kernel='rbf'`), jeweils mit Standard‑Parametern.
3. Für beide Modelle: Visualisieren Sie die Entscheidungsfläche (z. B. mit einem feinmaschigen Gitter wie in Lab 11.2 Aufgabe 2) und vergleichen Sie. Welche Variante kann die nichtlinearen Strukturen korrekt trennen?

## Aufgabe 3 – Hyperparameter‑Tuning für SVM

1. Verwenden Sie den Datensatz aus Aufgabe 2 (`make_circles`).
2. Legen Sie einen Parameter‑Grid fest für `C` (z. B. [0.1, 1, 10, 100]) und `gamma` (z. B. [0.01, 0.1, 1, 10]).
3. Führen Sie eine `GridSearchCV` mit 5‑facher Kreuzvalidierung durch (`scoring='accuracy'`), um die beste Kombination zu finden.
4. Geben Sie die besten Parameter sowie die mittlere Accuracy aus und erklären Sie kurz, wie `C` und `gamma` die Entscheidungsgrenze beeinflussen.
