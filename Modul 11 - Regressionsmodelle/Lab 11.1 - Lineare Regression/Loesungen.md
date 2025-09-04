# Lösungen zu Lab 11.1 – Lineare Regression: Annahmen, Gütemaße, Regularisierung

## Lösung zu Aufgabe 1 – Multiple lineare Regression und Gütemaße

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# 1. Datensatz einlesen
data = {
    "TV_Werbung": [230, 44, 17, 151, 180, 8, 199, 66, 152, 65, 150, 80],
    "Radio_Werbung": [37, 39, 45, 41, 10, 25, 3, 14, 23, 42, 33, 20],
    "Umsatz": [22, 10, 9, 18, 15, 4, 18, 11, 19, 15, 20, 12]
}
df = pd.DataFrame(data)

# Features und Ziel trennen
X = df[["TV_Werbung", "Radio_Werbung"]]
y = df["Umsatz"]

# 2. Lineare Regression trainieren
lin_reg = LinearRegression()
lin_reg.fit(X, y)

# Koeffizienten und Achsenabschnitt ausgeben
print("Koeffizienten:", lin_reg.coef_)
print("Achsenabschnitt:", lin_reg.intercept_)

# 3. Vorhersagen und Gütemaße
y_pred = lin_reg.predict(X)
r2 = r2_score(y, y_pred)
rmse = mean_squared_error(y, y_pred, squared=False)
mae = mean_absolute_error(y, y_pred)
print(f"R^2: {r2:.3f}, RMSE: {rmse:.3f}, MAE: {mae:.3f}")

# 4. Scatterplot
plt.scatter(y, y_pred)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--')
plt.xlabel("Tatsächlicher Umsatz")
plt.ylabel("Vorhergesagter Umsatz")
plt.title("Tatsächlicher vs. vorhergesagter Umsatz")
plt.show()
```

Die lineare Regression lernt eine Gerade durch die Daten. Das R‑Quadrat von rund 0.97 (je nach Daten) zeigt, dass ein großer Teil der Varianz erklärt wird.

## Lösung zu Aufgabe 2 – Annahmen prüfen

```python
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

# Residuen berechnen
residuals = y - y_pred

# 1. Residuen gegen Vorhersagen plotten
plt.scatter(y_pred, residuals)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel("Vorhergesagter Umsatz")
plt.ylabel("Residuen")
plt.title("Residuen vs. Vorhersagen")
plt.show()

# 2. Histogramm der Residuen
plt.hist(residuals, bins=5, edgecolor='black')
plt.xlabel("Residuen")
plt.ylabel("Häufigkeit")
plt.title("Histogramm der Residuen")
plt.show()

# Optional: QQ‑Plot
stats.probplot(residuals, dist="norm", plot=plt)
plt.title("QQ‑Plot der Residuen")
plt.show()

# 3. Korrelation zwischen TV und Radio
corr = df["TV_Werbung"].corr(df["Radio_Werbung"])
print(f"Korrelation zwischen TV_Werbung und Radio_Werbung: {corr:.3f}")
```

Die Residuen sollten zufällig um 0 streuen und keine systematische Struktur zeigen. Das Histogramm und der QQ‑Plot lassen erkennen, ob die Fehler ungefähr normalverteilt sind. Eine starke Korrelation (> 0.8) zwischen den Prädiktoren könnte auf Multikollinearität hinweisen. In unserem Beispiel ist die Korrelation relativ niedrig, weshalb Multikollinearität hier kein großes Problem darstellt.

## Lösung zu Aufgabe 3 – Ridge- und Lasso‑Regression

```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_squared_error

# 1. Train/Test‑Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)

# 2. Ridge‑Regression
alphas = [0.1, 1.0, 10.0]
ridge_results = {}
for a in alphas:
    ridge = Ridge(alpha=a)
    ridge.fit(X_train, y_train)
    y_pred_ridge = ridge.predict(X_test)
    rmse_ridge = mean_squared_error(y_test, y_pred_ridge, squared=False)
    ridge_results[a] = (ridge.coef_, rmse_ridge)

print("Ridge‑Ergebnisse (alpha: Koeffizienten, RMSE):")
for a, (coef, rmse_ridge) in ridge_results.items():
    print(f"alpha={a}: coeff={np.round(coef,3)}, RMSE={rmse_ridge:.3f}")

# 3. Lasso‑Regression
lasso_results = {}
for a in alphas:
    lasso = Lasso(alpha=a, max_iter=10000)
    lasso.fit(X_train, y_train)
    y_pred_lasso = lasso.predict(X_test)
    rmse_lasso = mean_squared_error(y_test, y_pred_lasso, squared=False)
    lasso_results[a] = (lasso.coef_, rmse_lasso)

print("
Lasso‑Ergebnisse (alpha: Koeffizienten, RMSE):")
for a, (coef, rmse_lasso) in lasso_results.items():
    print(f"alpha={a}: coeff={np.round(coef,3)}, RMSE={rmse_lasso:.3f}")
```

Ridge und Lasso fügen Strafterme hinzu, um Overfitting zu vermeiden. Bei steigenden `alpha`‑Werten schrumpfen die Koeffizienten: Ridge‑Regression reduziert sie kontinuierlich, während Lasso sie auf genau 0 setzen kann (Feature‑Selektion). Die beste `alpha`‑Wahl ist diejenige mit dem niedrigsten Test‑RMSE. In unserem Beispiel bleiben die Koeffizienten bei Ridge ähnlich wie bei der klassischen Regression, während Lasso bei größeren `alpha`‑Werten dazu neigt, einzelne Merkmale zu eliminieren.
