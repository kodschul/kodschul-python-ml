from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import numpy as np 
import pandas as pd

# random real data

n = 100000

df = pd.DataFrame()
heights = np.random.normal(loc=170, scale=10, size=n)
heights = np.clip(heights, 150, 190)

weights = np.random.normal(loc=80, scale=10, size=n)
weights = np.clip(weights, 45, 200)

# df["x"] = heights
# df["y"] = weights
# print(df)
# exit()

# Random regression data
x, y = make_regression(n_samples=10, n_features=1, noise=10.0)

df = pd.DataFrame()
df["x"] = np.array(x).reshape(10)
df["y"] = y

print(df)