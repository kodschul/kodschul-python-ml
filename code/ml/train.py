import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/people_weight_stats.csv")

# print(df)
# X = df[["Gender", "Height", "Age"]]
X = df["Height"]
y = df['Weight']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=50)
# print(X_test, y_test)



x_line = range(2, 100, 2)

# plt.plot(x_line, [x*2 for x in x_line], color="red")

plt.scatter(X, y)
# plt.bar(['Train', 'Test'], [len(y_train), len(y_test)], color=['blue', 'green'])
plt.show()

