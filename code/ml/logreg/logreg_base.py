import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score

# 1. Daten laden
df = pd.read_csv("../data/titanic_train.csv")
# test_df = pd.read_csv("../data/titanic_test.csv")

print(f"{len(df)} Data loaded for training")

# ignore datasets without Age
df = df[df['Age'].notna()]
# X['Age'] = X['Age'].fillna(X['Age'].median())

# 2. Features erstellen
X = df[["Pclass", "Sex", "Age", "Fare", "SibSp", "Parch"]]
X = df[["Pclass", "Age", "Fare", "Parch" , "Sex", "SibSp"]]

# convert gender male/female -> 1/0 
X['Sex'] = list([1 if gender == "male" else 0 for gender in  X['Sex'] ])
y = df['Survived']

# X_test = test_df[["Pclass", "Sex", "Age", "Fare", "SibSp", "Parch"]]
# y_test = test_df['Survived']

# 3. Test train split
print("Test splitting....")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)

# 4. Model trainieren
model = LogisticRegression()
print(f"Training.... with {len(X_train)} data points")
model.fit(X_train, y_train)

# 5. Model Bewertung 
print("Testing model")
y_pred = model.predict(X_test)

model_test_df = X_test
# model_test_df[""] = X_test 
X_test["Survival (Predicted)"] = y_pred
X_test["Survival (Actual)"] = y_test
# model_test_df["Weight (Predicted)"] = y_pred
# model_test_df["Weight (Actual)"] = y_test
print(model_test_df)
model_test_df.to_csv("../data/titanic_test_survival.csv")
# model_test_df.to_csv("../")

accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy: {round(accuracy*100, 2)}%")

# plt.scatter(X_test[""], y_test, color="green")
# plt.scatter(X_test[""], y_pred, color="blue")
# plt.show()
exit()
# print(model_test_df)

# plt.scatter(X_train, y_train)
# plt.show()