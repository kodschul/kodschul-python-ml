import pandas as pd 

df= pd.read_csv("../data/Experience-Salary.csv")

df["exp"] = round(df["exp(in months)"].round(0) / 12, 1)
df["salary"] = df["salary(in thousands)"].round(1) * 1000

print(df)
