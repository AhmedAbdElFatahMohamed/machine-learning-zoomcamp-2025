import pandas as pd
import numpy as np

# Q1. Pandas Version?
print(pd.__version__)

# Q2. Records count?
df = pd.read_csv("car_fuel_efficiency.csv")
print(len(df))

# Q3. Fuel types?
print(df["fuel_type"].unique())

# Q4. Missing values?
missing_columns = df.isnull().sum()
num_missing_cols = (missing_columns > 0).sum()
print(num_missing_cols)

# Q5. Max fuel efficiency?
max_asia_eff = df[df["origin"] == "Asia"]["fuel_efficiency_mpg"].max()
print(max_asia_eff)

# Q6. Median value of horsepower?
median_hp_before = df["horsepower"].median()
print(median_hp_before)

most_frequent_hp = df["horsepower"].mode()[0]
print(most_frequent_hp)

df["horsepower"] = df["horsepower"].fillna(most_frequent_hp)

median_hp_after = df["horsepower"].median()
print(median_hp_after)

# Q7. Sum of weights?

X = df[df["origin"] == "Asia"][["vehicle_weight", "model_year"]].head(7).values
XTX = X.T @ X
XTX_inv = np.linalg.inv(XTX)
y = np.array([1100, 1300, 800, 900, 1000, 1100, 1200])
w = XTX_inv @ X.T @ y
result = w.sum()
print(result)
