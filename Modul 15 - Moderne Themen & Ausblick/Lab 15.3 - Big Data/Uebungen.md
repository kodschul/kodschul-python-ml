# Lab 15.3 – Big Data: Spark & PySpark mit Hadoop – skalierbare Datenverarbeitung

Da in dieser Umgebung keine Spark-Installation verfügbar ist, simulieren wir das Arbeiten mit großen Datenmengen mit `pandas` und lernen die grundlegenden Konzepte wie MapReduce und Parallelisierung kennen.

## Aufgabe 1 – Arbeiten mit großen Daten in Pandas

1. Erzeugen Sie einen DataFrame mit einer Million Zeilen und den Spalten `user_id` (Zufallszahl von 1 bis 10 000), `event_type` (zufällig aus ['click','view','purchase']), und `value` (Zufallszahl zwischen 0 und 100). Verwenden Sie `numpy.random`.
2. Speichern Sie den DataFrame als CSV-Datei.
3. Laden Sie die CSV-Datei anschließend in kleinen Stücken (`chunksize=100 000`) mit `pandas.read_csv` und berechnen Sie den durchschnittlichen `value` je `event_type` (Aggregation über alle Chunks hinweg).
4. Vergleichen Sie die Laufzeit dieses Ansatzes mit dem Laden der gesamten Datei auf einmal. Warum ist Chunking bei großen Datenmengen hilfreich?

## Aufgabe 2 – Parallele Verarbeitung mit Joblib

1. Erstellen Sie eine Liste von 100 000 Zahlen und schreiben Sie eine Funktion, die für jede Zahl `n` die Summe der Quadrate von 1 bis `n` berechnet.
2. Verwenden Sie `joblib.Parallel` und `joblib.delayed`, um die Berechnungen parallel in 4 Prozessen auszuführen. Messen Sie die Ausführungszeit im Vergleich zur sequentiellen Berechnung mit einer Schleife.
3. Wie viel schneller ist die parallele Variante? Was muss beachtet werden, wenn Daten zwischen Prozessen ausgetauscht werden?

## Aufgabe 3 – MapReduce-Konzept demonstrieren

1. Implementieren Sie eine einfache MapReduce-ähnliche Funktion in Python, um die Wortanzahl in einer Liste von Textzeilen zu berechnen:
   - **Map-Phase**: Zerlegen Sie jede Zeile in Wörter und erzeugen Sie `(Wort, 1)`-Paare.
   - **Shuffle/Sort-Phase**: Fassen Sie identische Wörter zusammen (z. B. mithilfe eines `defaultdict`).
   - **Reduce-Phase**: Summieren Sie die Einsen pro Wort.
2. Testen Sie Ihre Implementierung mit einer Liste von 100 Textzeilen.
3. Wie unterscheidet sich dieses Modell von einer direkten Berechnung mit `Counter` aus der Python-Standardbibliothek, und warum ist MapReduce für verteilte Systeme geeignet?
