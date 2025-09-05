import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score, mean_squared_error, mean_absolute_error
import numpy as np

n = 100000
df = pd.DataFrame()
heights = np.random.normal(loc=170, scale=10, size=n)
heights = np.clip(heights, 150, 190)

weights = np.random.normal(loc=80, scale=10, size=n)
weights = np.clip(weights, 45, 200)

# 2. Features erstellen
X = np.array(heights).reshape(-1, 1)
y = weights

# 3. Test train split
print("Test splitting....")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)

# 4. Model trainieren
lin_reg = LinearRegression()
print(f"Training.... with {len(X_train)} data points")
lin_reg.fit(X_train, y_train)

# 5. Model Bewertung 
print("Testing model")
y_pred = lin_reg.predict(X_test)

# model_test_df = pd.DataFrame()
# model_test_df["Height"] = X_test 
# model_test_df["Weight (Predicted)"] = y_pred
# model_test_df["Weight (Actual)"] = y_test


r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
# accuracy = accuracy_score(y_test, y_pred)

print(f"R^2: {r2:.2f}")
print(f"MAE: {mae:.2f}")
# print(f"Accuracy: {accuracy}%")


plt.scatter(X_test, y_test, color="green")
plt.scatter(X_test, y_pred, color="blue")
plt.show()
exit()
# print(model_test_df)

# plt.scatter(X_train, y_train)
# plt.show()