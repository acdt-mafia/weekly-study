import numpy as np
import pandas as pd
import statsmodels.api as sm


# Load data
df = pd.read_csv("data_week1.csv")

df["x2"] = df["x"] ** 2

X = df[["x", "x2"]]
X = sm.add_constant(X)   # adds intercept
y = df["y"]

model = sm.OLS(y, X).fit()

print(model.summary())