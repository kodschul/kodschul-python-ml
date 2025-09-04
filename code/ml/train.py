import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/people_stats.csv")

# print(df)
X = df[["Gender", "Height", "Age"]]
y = df['Weight']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=50)

plt.bar(['Train', 'Test'], [len(y_train), len(y_test)], color=['blue', 'green'])
plt.show()

