# Lab 11.1 – Lineare Regression: Annahmen, Gütemaße, Regularisierung

In diesem Lab wenden Sie die lineare Regression auf einen kleinen Datensatz an, überprüfen wichtige Modellannahmen und lernen Regularisierungsmethoden kennen.

Der folgende Datensatz zeigt die wöchentlichen **Werbeausgaben** eines Unternehmens (in Tausend Euro) sowie den erzielten **Umsatz** (ebenfalls in Tausend Euro). Laden Sie den Datensatz als DataFrame – z. B. per Kopieren der Tabelle – oder übertragen Sie ihn als Python‑Dictionary.

| Index | TV_Werbung | Radio_Werbung | Umsatz |
|------:|-----------:|--------------:|-------:|
| 0     | 230        | 37            | 22     |
| 1     | 44         | 39            | 10     |
| 2     | 17         | 45            | 9      |
| 3     | 151        | 41            | 18     |
| 4     | 180        | 10            | 15     |
| 5     | 8          | 25            | 4      |
| 6     | 199        | 3             | 18     |
| 7     | 66         | 14            | 11     |
| 8     | 152        | 23            | 19     |
| 9     | 65         | 42            | 15     |
| 10    | 150        | 33            | 20     |
| 11    | 80         | 20            | 12     |

## Aufgabe 1 – Multiple lineare Regression und Gütemaße

1. Lesen Sie den obigen Datensatz in einen `pandas.DataFrame`. Teilen Sie die Daten in Features (`TV_Werbung`, `Radio_Werbung`) und Ziel (`Umsatz`).
2. Trainieren Sie ein `LinearRegression`‑Modell und geben Sie die Koeffizienten (Steigungen) und den Achsenabschnitt aus.
3. Erzeugen Sie Vorhersagen und berechnen Sie die Gütemaße **R‑Quadrat**, **Root Mean Squared Error (RMSE)** und **Mean Absolute Error (MAE)**.
4. Erstellen Sie ein Streudiagramm der tatsächlichen Umsätze versus der vorhergesagten Umsätze.

## Aufgabe 2 – Annahmen der linearen Regression prüfen

1. Berechnen Sie die Residuen (Fehler = tatsächlicher Wert – Vorhersage).
2. Stellen Sie die Residuen gegen die vorhergesagten Werte in einem Streudiagramm dar. Achten Sie darauf, ob ein Muster erkennbar ist oder die Punkte zufällig streuen (Homoskedastizität).
3. Zeichnen Sie ein Histogramm der Residuen und optional einen QQ‑Plot, um zu prüfen, ob die Fehler annähernd normalverteilt sind.
4. Berechnen Sie die Korrelation zwischen `TV_Werbung` und `Radio_Werbung`. Was sagt dieser Wert über mögliche Multikollinearität aus?

## Aufgabe 3 – Ridge- und Lasso‑Regression

1. Teilen Sie den Datensatz in Trainings‑ und Testdaten (z. B. 70 %/30 %, `random_state=42`).
2. Trainieren Sie ein Ridge‑Regressionsmodell (`Ridge` aus `sklearn.linear_model`) mit unterschiedlichen Werten für `alpha` (z. B. 0.1, 1.0, 10.0). Berechnen Sie für jedes Modell die Test‑RMSE.
3. Wiederholen Sie Schritt 2 für ein Lasso‑Regressionsmodell (`Lasso`). Notieren Sie, bei welchem `alpha` die RMSE am niedrigsten ist.
4. Vergleichen Sie die Koeffizienten der Ridge‑ und Lasso‑Modelle mit denen der klassischen linearen Regression. Welche Features werden bei Lasso eventuell auf 0 gesetzt?
