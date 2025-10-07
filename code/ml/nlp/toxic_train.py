import pandas as pd 
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score



df = pd.read_csv("../data/toxic_comments.csv")

# 1. Data cleanup
print(f"Data cleanup.... {len(df)}")
# lowercase
df["comment_text"] = df["comment_text"].str.lower()


# Splitting
print(f"Splitting...")
train, test = train_test_split(df, test_size=0.3, random_state=42, shuffle=True)

train_text = train['comment_text']
test_text = test['comment_text']

vec = TfidfVectorizer(analyzer="word")
vec.fit(train_text)
vec.fit(test_text)

X_train = vec.transform(list(train_text))
y_train = train["toxic"]


X_test = vec.transform(test_text)
y_test = test["toxic"]


# Training
print(f"Training with {len(train)} data points...")
model = LogisticRegression(solver='sag')
model.fit(X_train, y_train)


# Testing
print(f"Testing with  {len(test)} data points...")
y_pred = model.predict(X_test)
accuracy_rate = accuracy_score(y_test, y_pred)

print(f"Accuracy: {round(accuracy_rate *100, 2)}%")

test_df = test[['comment_text', 'toxic']]
test_df['Predicted Toxic'] = y_pred
print(test_df)

# test_df.to_csv("../data/toxic_prediction.csv")


joblib.dump(vec, "../data/models/toxic.vec")
joblib.dump(model, "../data/models/toxic.model")