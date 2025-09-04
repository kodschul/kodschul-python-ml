# Lösungen zu Lab 15.2 – Einführung in Deep Learning: einfache dichte Netze

## Lösung zu Aufgabe 1 – MLP-Klassifikation

```python
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt

# 1. Datensatz erzeugen
X, y = make_classification(n_samples=600, n_features=20, n_informative=15, n_classes=2, random_state=0)

# 2. Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# 3. MLP trainieren
mlp = MLPClassifier(hidden_layer_sizes=(10,), activation='relu', solver='adam', max_iter=200, random_state=0)
mlp.fit(X_train, y_train)

# 4. Loss-Kurve
plt.plot(mlp.loss_curve_)
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.title('MLP Loss-Kurve während des Trainings')
plt.show()

# 5. Bewertung
y_pred = mlp.predict(X_test)
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
print(f'MLP Accuracy: {acc:.3f}, F1-Score: {f1:.3f}')
```

Das neuronale Netz minimiert den Fehler iterativ. Die Loss-Kurve zeigt, wie der Verlust während des Trainings abnimmt. Die versteckte Schicht mit zehn Neuronen ermöglicht es dem Modell, nichtlineare Zusammenhänge zu lernen.

## Lösung zu Aufgabe 2 – Vergleich mit logistische Regression

```python
from sklearn.linear_model import LogisticRegression

# Logistische Regression trainieren
log_reg = LogisticRegression(max_iter=200, random_state=0)
log_reg.fit(X_train, y_train)
y_pred_lr = log_reg.predict(X_test)
acc_lr = accuracy_score(y_test, y_pred_lr)
f1_lr = f1_score(y_test, y_pred_lr)

print(f'Logistische Regression – Accuracy: {acc_lr:.3f}, F1-Score: {f1_lr:.3f}')
print(f'MLP – Accuracy: {acc:.3f}, F1-Score: {f1:.3f}')
```

Bei linearen Trennproblemen schneiden die logistische Regression und das MLP ähnlich ab. Bei komplexeren Zusammenhängen mit vielen Wechselwirkungen kann ein MLP besser generalisieren, da es versteckte Schichten zur Modellierung nichtlinearer Beziehungen nutzt.

## Lösung zu Aufgabe 3 – MLP-Regression

```python
from sklearn.datasets import make_regression
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression

# 1. Datensatz erzeugen
X_reg, y_reg = make_regression(n_samples=500, n_features=5, noise=15, random_state=1)

# Train/Test-Split
X_train, X_test, y_train, y_test = train_test_split(X_reg, y_reg, test_size=0.3, random_state=1)

# 2. MLPRegressor trainieren
mlp_reg = MLPRegressor(hidden_layer_sizes=(50,20), activation='relu', solver='adam', max_iter=500, random_state=1)
mlp_reg.fit(X_train, y_train)

# 3. Bewertung
y_pred_mlp = mlp_reg.predict(X_test)
r2_mlp = r2_score(y_test, y_pred_mlp)
rmse_mlp = mean_squared_error(y_test, y_pred_mlp, squared=False)

# Vergleich mit Linear Regression
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
y_pred_lin = lin_reg.predict(X_test)
r2_lin = r2_score(y_test, y_pred_lin)
rmse_lin = mean_squared_error(y_test, y_pred_lin, squared=False)

print(f'MLP Regression – R^2: {r2_mlp:.3f}, RMSE: {rmse_mlp:.2f}')
print(f'Lineare Regression – R^2: {r2_lin:.3f}, RMSE: {rmse_lin:.2f}')
```

Das MLP kann komplexere, nichtlineare Zusammenhänge erfassen, was zu einer höheren Güte der Vorhersage führen kann. Bei linearen Problemen oder kleinen Datensätzen kann jedoch die klassische lineare Regression ausreichend sein und schneller konvergieren.
