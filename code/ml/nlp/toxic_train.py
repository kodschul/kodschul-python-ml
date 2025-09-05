import pandas as pd 

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

vectorizer = TfidfVectorizer(analyzer="word")
vectorizer.fit(train_text)
# vectorizer.fit(test_text)

X_train = vectorizer.transform(train_text)
y_train = train["toxic"]


X_test = vectorizer.transform(test_text)
y_test = test["toxic"]


print(X_train.shape)
exit()

# Training
print(f"Training with  {len(X_train)} data points...")
model = LogisticRegression(solver='sag')
model.fit(X_train, y_train)

exit()

# Testing
print(f"Testing with  {len(X_test)} data points...")
y_pred = model.predict(X_test)
accuracy_rate = accuracy_score(y_test, y_pred)

print(f"Accuracy: {round(accuracy_rate *100, 2)}%")

test_df = test[['comment_text', 'toxic']]
test_df['Predicted Toxic'] = y_pred
print(test_df)