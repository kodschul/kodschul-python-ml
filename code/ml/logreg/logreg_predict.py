import joblib
import pandas as pd

df = pd.read_csv("../data/titanic_test.csv")
df = df[df['Age'].notna()]
df = df[df['Fare'].notna()]

X = df[["Pclass", "Age", "Fare", "Parch" , "Sex", "SibSp"]]
# convert gender male/female -> 1/0 
X['Sex'] = list([1 if gender == "male" else 0 for gender in  df['Sex'] ])

model = joblib.load("../data/models/titanic.model")
survival_rate_pred = model.predict(X)

X['SurvivalRate'] = survival_rate_pred
print(X)


