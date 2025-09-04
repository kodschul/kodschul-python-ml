# Lösungen zu Lab 15.3 – Big Data: Spark & PySpark – skalierbare Datenverarbeitung

## Lösung zu Aufgabe 1 – Arbeiten mit großen Daten in Pandas

```python
import pandas as pd
import numpy as np
import time

# 1. DataFrame mit 1 Million Zeilen erzeugen
np.random.seed(0)
n_rows = 1_000_000
df = pd.DataFrame({
    'user_id': np.random.randint(1, 10001, size=n_rows),
    'event_type': np.random.choice(['click','view','purchase'], size=n_rows, p=[0.7, 0.25, 0.05]),
    'value': np.random.rand(n_rows) * 100
})

# 2. CSV speichern
csv_path = '/home/oai/share/big_data_events.csv'
df.to_csv(csv_path, index=False)

# 3. Aggregation mit Chunking
start_time = time.time()
chunk_size = 100_000
totals = {'click': 0, 'view': 0, 'purchase': 0}
counts = {'click': 0, 'view': 0, 'purchase': 0}

for chunk in pd.read_csv(csv_path, chunksize=chunk_size):
    totals_chunk = chunk.groupby('event_type')['value'].sum().to_dict()
    counts_chunk = chunk.groupby('event_type')['value'].count().to_dict()
    for key in totals_chunk:
        totals[key] += totals_chunk[key]
        counts[key] += counts_chunk[key]

avg_values = {evt: totals[evt]/counts[evt] for evt in totals}
duration_chunk = time.time() - start_time

print('Durchschnittlicher value je event_type (Chunking):', avg_values)
print(f'Laufzeit mit Chunking: {duration_chunk:.2f} Sekunden')

# Vergleich: Laden des gesamten DataFrames auf einmal
start_time = time.time()
df_full = pd.read_csv(csv_path)
avg_values_full = df_full.groupby('event_type')['value'].mean().to_dict()
duration_full = time.time() - start_time

print('Durchschnittlicher value je event_type (komplett):', avg_values_full)
print(f'Laufzeit ohne Chunking: {duration_full:.2f} Sekunden')
```

Das Lesen und Aggregieren in Chunks benötigt weniger Speicher, da jeweils nur ein Teil der Daten im RAM liegt. Für sehr große Dateien ermöglicht dieser Ansatz überhaupt erst die Verarbeitung, auch wenn die Laufzeit pro Zeile leicht höher ist.

## Lösung zu Aufgabe 2 – Parallele Verarbeitung mit Joblib

```python
from joblib import Parallel, delayed
import time

# 1. Funktion definieren
def sum_of_squares(n):
    return sum(i*i for i in range(1, n+1))

numbers = list(range(1, 100001))

# Sequentielle Berechnung
start = time.time()
results_seq = [sum_of_squares(n) for n in numbers]
seq_time = time.time() - start
print(f'Sequenzielle Zeit: {seq_time:.2f} s')

# 2. Parallele Berechnung mit 4 Jobs
start = time.time()
results_par = Parallel(n_jobs=4)(delayed(sum_of_squares)(n) for n in numbers)
par_time = time.time() - start
print(f'Parallele Zeit: {par_time:.2f} s')
```

Die parallele Variante ist deutlich schneller, da die Berechnungen auf mehrere Kerne verteilt werden. Allerdings müssen Daten zwischen Prozessen serialisiert werden; bei sehr großen Objekten kann dies den Vorteil reduzieren.

## Lösung zu Aufgabe 3 – MapReduce-Konzept demonstrieren

```python
from collections import defaultdict


def map_reduce_word_count(lines):
    # Map-Phase
    mapped = []
    for line in lines:
        words = line.strip().split()
        mapped.extend([(word.lower(), 1) for word in words])
    # Shuffle/Sort-Phase
    grouped = defaultdict(list)
    for word, count in mapped:
        grouped[word].append(count)
    # Reduce-Phase
    reduced = {word: sum(counts) for word, counts in grouped.items()}
    return reduced

# Beispiel
text_lines = [
    'Der Hund spielt im Park',
    'Der Park ist groß',
    'Hund und Katze spielen'
] * 33  # 99 Zeilen

word_counts = map_reduce_word_count(text_lines)
print(word_counts)

# Vergleich mit Counter
from collections import Counter
counter_counts = Counter(' '.join(text_lines).lower().split())
print(counter_counts)
```

Die MapReduce-Implementierung folgt dem Paradigma der verteilten Verarbeitung: Daten werden in unabhängige Paare zerlegt (Map), nach Schlüssel sortiert und gruppiert (Shuffle), dann aggregiert (Reduce). Anders als `Counter`, das lokal in einem Schritt arbeitet, ermöglicht MapReduce die Verteilung der Phasen auf verschiedene Knoten. Dies macht das Modell für sehr große Datenmengen und Clusterumgebungen geeignet.
