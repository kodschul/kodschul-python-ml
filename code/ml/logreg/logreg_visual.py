import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score

# 1. Daten laden
df = pd.read_csv("../data/titanic_train.csv")
# test_df = pd.read_csv("../data/titanic_test.csv")


plt.scatter(df['Age'], df['Survived'])
plt.show()