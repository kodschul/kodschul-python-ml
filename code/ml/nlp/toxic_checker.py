import joblib
import pandas as pd

texts = [
    "Die in hell you scum",
    "Kill yourself you idiot!",
    "You're amazing",
    "Love you forever",
]

vec = joblib.load("../data/models/toxic.vec")
model = joblib.load("../data/models/toxic.model")

vec_texts = vec.transform(texts)
toxicity_pred = model.predict(vec_texts)

df = pd.DataFrame()
df["texts"] = texts
df["toxic prob"] = toxicity_pred
print(df)
