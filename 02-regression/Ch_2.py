import pandas as pd
import numpy as np

df = pd.read_csv("car_fuel_efficiency.csv")


# Q1. Missing values?

missing_cols = df.columns[df.isnull().any()]
print("Columns with missing values:", list(missing_cols))
print("__" * 50)


# Q2. Median for horse power?

median_hp = df["horsepower"].median()
print("Median horsepower:", median_hp)
print("__" * 50)

# Shuffle and split the dataset

df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
n = len(df_shuffled)
n_train, n_val = int(0.6 * n), int(0.2 * n)

df_train = df_shuffled.iloc[:n_train]
df_val = df_shuffled.iloc[n_train:n_train + n_val]
df_test = df_shuffled.iloc[n_train + n_val:]

print(f"Train: {len(df_train)/n:.2%}")
print(f"Val:   {len(df_val)/n:.2%}")
print(f"Test:  {len(df_test)/n:.2%}")
print("__" * 50)

# Helper Functions

def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred)**2))

def train_lin_reg(X, y):
    X = np.column_stack([np.ones(len(X)), X])
    return np.linalg.inv(X.T @ X) @ X.T @ y

def train_lin_reg_ridge(X, y, r=0.0):
    X = np.column_stack([np.ones(len(X)), X])
    XTX = X.T @ X + r * np.eye(X.shape[1])
    return np.linalg.inv(XTX) @ X.T @ y

features = ['engine_displacement', 'horsepower', 'vehicle_weight', 'model_year']

# Q3. Filling NAs?

X_train, X_val = df_train[features].copy(), df_val[features].copy()
y_train, y_val = df_train['fuel_efficiency_mpg'].values, df_val['fuel_efficiency_mpg'].values

for fill_value in [0, X_train['horsepower'].mean()]:
    X_train_filled = X_train.fillna({'horsepower': fill_value})
    X_val_filled = X_val.fillna({'horsepower': fill_value})

    w = train_lin_reg(X_train_filled.values, y_train)
    y_pred = np.column_stack([np.ones(len(X_val_filled)), X_val_filled.values]) @ w

    print(f"Fill with {fill_value if fill_value != 0 else 0}: RMSE = {round(rmse(y_val, y_pred), 2)}")

print("__" * 50)

# Q4. Best regularization?

X_train = df_train[features].fillna(0).values
X_val = df_val[features].fillna(0).values
y_train = df_train['fuel_efficiency_mpg'].values
y_val = df_val['fuel_efficiency_mpg'].values

X_train = np.column_stack([np.ones(len(X_train)), X_train])
X_val = np.column_stack([np.ones(len(X_val)), X_val])

for r in [0, 0.01, 0.1, 1, 5, 10, 100]:
    w = train_lin_reg_ridge(X_train[:, 1:], y_train, r)
    y_pred = X_val @ w
    print(f"r={r}: RMSE = {round(rmse(y_val, y_pred), 2)}")

print("__" * 50)

# Q5. RMSE Standard Deviation?

seeds = range(10)
scores = []

for seed in seeds:
    df_shuffled = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    n = len(df_shuffled)
    n_train, n_val = int(0.6 * n), int(0.2 * n)
    df_train = df_shuffled.iloc[:n_train]
    df_val = df_shuffled.iloc[n_train:n_train + n_val]

    X_train = df_train[features].fillna(0).values
    X_val = df_val[features].fillna(0).values
    y_train = df_train['fuel_efficiency_mpg'].values
    y_val = df_val['fuel_efficiency_mpg'].values

    w = train_lin_reg(X_train, y_train)
    y_pred = np.column_stack([np.ones(len(X_val)), X_val]) @ w
    scores.append(rmse(y_val, y_pred))

print("Standard deviation of RMSE:", round(np.std(scores), 3))
print("__" * 50)

# Q6. Evaluation on test?

df_shuffled = df.sample(frac=1, random_state=9).reset_index(drop=True)
n = len(df_shuffled)
n_train, n_val = int(0.6 * n), int(0.2 * n)

df_train = df_shuffled.iloc[:n_train]
df_val = df_shuffled.iloc[n_train:n_train + n_val]
df_test = df_shuffled.iloc[n_train + n_val:]
df_full_train = pd.concat([df_train, df_val])

X_full_train = df_full_train[features].fillna(0).values
y_full_train = df_full_train['fuel_efficiency_mpg'].values
X_test = df_test[features].fillna(0).values
y_test = df_test['fuel_efficiency_mpg'].values

w = train_lin_reg_ridge(X_full_train, y_full_train, r=0.001)
y_pred = np.column_stack([np.ones(len(X_test)), X_test]) @ w

print("Final Test RMSE:", round(rmse(y_test, y_pred), 2))
