import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge,Lasso
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score, accuracy_score, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

# 1. Daten laden
df = pd.read_csv("../data/people_weight_stats.csv")
print(f"{len(df)} Data loaded for training")

# 2. Features erstellen
X = df[["Height"]]
y = df['Weight']

# 3. Test train split
print("Test splitting....")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)

pipe = Pipeline([
        ("poly", PolynomialFeatures(degree=2, include_bias=False)),
    ("scaler", StandardScaler()),
    ("model", Lasso(max_iter=1000)),
])


param_grid = {
    "model__alpha": [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000],
}

model = GridSearchCV(pipe, param_grid, cv=5, scoring="r2")
print(f"Training.... with {len(X_train)} data points")
model.fit(X_train, y_train)
print("Best alpha:", model.best_params_)
print("Best CV R²:", model.best_score_)


# 5. Model Bewertung 
print("Testing model")
y_pred = model.predict(X_test)

model_test_df = pd.DataFrame()
model_test_df["Height"] = X_test 
model_test_df["Weight (Predicted)"] = y_pred
model_test_df["Weight (Actual)"] = y_test


r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print(f"R^2: {r2:.2f}")
print(f"MSE: {mse:.2f}%")


plt.scatter(X_test, y_test, color="green")
plt.scatter(X_test, y_pred, color="blue")
plt.show()
exit()
# print(model_test_df)

# plt.scatter(X_train, y_train)
# plt.show()