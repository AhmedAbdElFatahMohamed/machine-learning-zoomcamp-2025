import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mutual_info_score, accuracy_score
from sklearn.linear_model import LogisticRegression

# Load dataset
df = pd.read_csv("course_lead_scoring.csv")

# Q1. Most frequent industry
most_frequent_industry = df['industry'].mode()[0]
print("Most frequent industry:", most_frequent_industry)

print("__" * 50)

# Q2. Biggest correlation
num_df = df.select_dtypes(include=['number'])
corr_matrix = num_df.corr()
corr_unstacked = corr_matrix.unstack().sort_values(ascending=False)
corr_unstacked = corr_unstacked[corr_unstacked < 1]
most_correlated_pair = corr_unstacked.idxmax()
highest_corr_value = corr_unstacked.max()
print("Most correlated features:", most_correlated_pair)
print("Correlation coefficient:", round(highest_corr_value, 3))

print("__" * 50)

# Split data 60/20/20
X = df.drop(columns=['converted'])
y = df['converted']

X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.25, random_state=42)

print(f"Train: {len(X_train)/len(df):.2%}")
print(f"Validation: {len(X_val)/len(df):.2%}")
print(f"Test: {len(X_test)/len(df):.2%}")

print("__" * 50)

# Q3. Mutual information for categorical features
cat_cols = X_train.select_dtypes(include=['object']).columns
X_train_filled = X_train[cat_cols].fillna('missing')

for col in cat_cols:
    score = mutual_info_score(X_train_filled[col], y_train)
    print(f"{col}: {round(score, 2)}")

print("__" * 50)

# Q4. Logistic Regression accuracy

cat_cols = X_train.select_dtypes(include=['object']).columns.tolist()
num_cols = X_train.select_dtypes(include=['number']).columns.tolist()

X_train_cat = X_train[cat_cols].fillna('missing')
X_val_cat = X_val[cat_cols].fillna('missing')

X_train_num = X_train[num_cols].fillna(0)
X_val_num = X_val[num_cols].fillna(0)

encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
X_train_encoded = encoder.fit_transform(X_train_cat)
X_val_encoded = encoder.transform(X_val_cat)

X_train_final = np.concatenate([X_train_num.values, X_train_encoded], axis=1)
X_val_final = np.concatenate([X_val_num.values, X_val_encoded], axis=1)

model = LogisticRegression(solver='liblinear', C=1.0, max_iter=1000, random_state=42)
model.fit(X_train_final, y_train)

y_val_pred = model.predict(X_val_final)

val_accuracy = accuracy_score(y_val, y_val_pred)
print(f"Validation accuracy: {round(val_accuracy, 2)}")

print("__" * 50)

# Q5. Feature Selection

num_feature_names = num_cols
cat_feature_names = encoder.get_feature_names_out(cat_cols).tolist()
all_feature_names = num_feature_names + cat_feature_names

model_baseline = LogisticRegression(solver='liblinear', C=1.0, max_iter=1000, random_state=42)
model_baseline.fit(X_train_final, y_train)
y_val_pred_baseline = model_baseline.predict(X_val_final)
baseline_accuracy = accuracy_score(y_val, y_val_pred_baseline)

feature_importance = {}

for i, feature_name in enumerate(all_feature_names):

    train_indices = [j for j in range(X_train_final.shape[1]) if j != i]
    val_indices = [j for j in range(X_val_final.shape[1]) if j != i]

    X_train_without_feature = X_train_final[:, train_indices]
    X_val_without_feature = X_val_final[:, val_indices]

    model = LogisticRegression(solver='liblinear', C=1.0, max_iter=1000, random_state=42)
    model.fit(X_train_without_feature, y_train)

    y_val_pred = model.predict(X_val_without_feature)
    accuracy_without_feature = accuracy_score(y_val, y_val_pred)

    accuracy_diff = baseline_accuracy - accuracy_without_feature

    feature_importance[feature_name] = {
        'accuracy_without': accuracy_without_feature,
        'difference': accuracy_diff
    }

sorted_features = sorted(feature_importance.items(), key=lambda x: x[1]['difference'])

for feature, scores in sorted_features[:10]:
    print(f"{feature}: diff = {scores['difference']:.6f}, acc_without = {scores['accuracy_without']:.6f}")
least_useful_feature = sorted_features[0][0]

print("__" * 50)

# Q6. Parameter tuning

C_values = [0.01, 0.1, 1, 10, 100]
results = {}

for C in C_values:
    model = LogisticRegression(solver='liblinear', C=C, max_iter=1000, random_state=42)
    model.fit(X_train_final, y_train)
    y_val_pred = model.predict(X_val_final)
    val_accuracy = accuracy_score(y_val, y_val_pred)
    val_accuracy_rounded = round(val_accuracy, 3)
    results[C] = val_accuracy_rounded
    print(f"C = {C:6.2f}: Validation Accuracy = {val_accuracy_rounded}")

print("__" * 50)
