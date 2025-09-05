import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score, mean_absolute_error

# 1. Daten laden
df = pd.read_csv("../data/people_weight_stats.csv")
print(f"{len(df)} Data loaded for training")

# 2. Features erstellen
X = df[["Height", "Age"]]
y = df['Weight']

# 3. Test train split
print("Test splitting....")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)

# 4. Model trainieren
model = Lasso(alpha=15)
print(f"Training.... with {len(X_train)} data points")
model.fit(X_train, y_train)
# X_test = X_train
# y_test = y_train

# 5. Model Bewertung 
print("Testing model")
y_pred = model.predict(X_test)

model_test_df = pd.DataFrame()
model_test_df["Height"] = X_test["Height"]
model_test_df["Age"] = X_test["Age"]
model_test_df["Weight (Predicted)"] = y_pred
model_test_df["Weight (Actual)"] = y_test

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
# accuracy = accuracy_score(y_test, y_pred)

print(f"R^2: {r2:.2f}")
print(f"MAE: {mae}")
# print(f"Accuracy: {accuracy}%")

plt.scatter(X_test["Height"], y_test, color="green")
plt.scatter(X_test["Height"], y_pred, color="blue")
plt.show()
exit()
# print(model_test_df)

# plt.scatter(X_train, y_train)
# plt.show()