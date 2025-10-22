import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score , f1_score
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("course_lead_scoring.csv")  # Replace with your actual filename

# Check for missing values
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
numerical_cols = df.select_dtypes(include=['number']).columns.tolist()

print(f"Categorical columns: {categorical_cols}")
print(f"Numerical columns: {numerical_cols}")
print("__" * 50)

for col in categorical_cols:
    df[col] = df[col].fillna('NA')

for col in numerical_cols:
    df[col] = df[col].fillna(0.0)

# Split data
train_df, temp_df = train_test_split(df, test_size=0.4, random_state=1)

# Second split: split temp into 50%/50% (which gives 20%/20% of original)
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=1)

print(f"Train size: {len(train_df)} ({len(train_df)/len(df):.1%})")
print(f"Validation size: {len(val_df)} ({len(val_df)/len(df):.1%})")
print(f"Test size: {len(test_df)} ({len(test_df)/len(df):.1%})")

print("__" * 50)

# Q1.  ROC AUC feature importance

y_train = train_df['converted']
numerical_cols = ['lead_score', 'number_of_courses_viewed', 'interaction_count', 'annual_income']

auc_scores = {}

for col in numerical_cols:
    score = train_df[col]
    auc = roc_auc_score(y_train, score)
    if auc < 0.5:
        auc_inverted = roc_auc_score(y_train, -score)
        auc_scores[col] = auc_inverted
        print(f"{col}: AUC = {auc:.3f} -> Inverted AUC = {auc_inverted:.3f}")
    else:
        auc_scores[col] = auc
        print(f"{col}: AUC = {auc:.3f}")
sorted_auc = sorted(auc_scores.items(), key=lambda x: x[1], reverse=True)

print("\nFeatures ranked by AUC:")
for feature, auc in sorted_auc:
    print(f"{feature}: {auc:.3f}")

print(f"Feature with highest AUC: {sorted_auc[0][0]}")
print(f"Highest AUC: {sorted_auc[0][1]:.3f}")

print("__" * 50)

# Q2.  Training the model

X_train = train_df.drop(columns=['converted'])
y_train = train_df['converted']

X_val = val_df.drop(columns=['converted'])
y_val = val_df['converted']

train_dicts = X_train.to_dict(orient='records')
val_dicts = X_val.to_dict(orient='records')

dv = DictVectorizer(sparse=False)
X_train_encoded = dv.fit_transform(train_dicts)
X_val_encoded = dv.transform(val_dicts)

model = LogisticRegression(solver='liblinear', C=1.0, max_iter=1000)
model.fit(X_train_encoded, y_train)

y_val_pred_proba = model.predict_proba(X_val_encoded)[:, 1]

auc = roc_auc_score(y_val, y_val_pred_proba)
auc_rounded = round(auc, 3)

print(f"AUC on validation set: {auc_rounded}")

print("__" * 50)

# Q3. Precision and Recall at different thresholds

y_val_pred_proba = model.predict_proba(X_val_encoded)[:, 1]

thresholds = np.arange(0.0, 1.01, 0.01)
precisions = []
recalls = []

for threshold in thresholds:
    y_val_pred = (y_val_pred_proba >= threshold).astype(int)

    precision = precision_score(y_val, y_val_pred, zero_division=0)
    recall = recall_score(y_val, y_val_pred, zero_division=0)

    precisions.append(precision)
    recalls.append(recall)

# plt.figure(figsize=(10, 6))
# plt.plot(thresholds, precisions, label='Precision', linewidth=2)
# plt.plot(thresholds, recalls, label='Recall', linewidth=2)
# plt.xlabel('Threshold', fontsize=12)
# plt.ylabel('Score', fontsize=12)
# plt.title('Precision and Recall vs Threshold', fontsize=14)
# plt.legend(fontsize=12)
# plt.grid(True, alpha=0.3)
# plt.xlim(0, 1)
# plt.ylim(0, 1)
# plt.show()

differences = np.abs(np.array(precisions) - np.array(recalls))

intersection_idx = np.argmin(differences)
intersection_threshold = thresholds[intersection_idx]
intersection_precision = precisions[intersection_idx]
intersection_recall = recalls[intersection_idx]

print(f"Intersection point:")
print(f"Threshold: {intersection_threshold:.2f}")
print(f"Precision: {intersection_precision:.3f}")
print(f"Recall: {intersection_recall:.3f}")
print(f"Difference: {differences[intersection_idx]:.4f}")

print("__" * 50)

# Q4. Removing the least important feature

thresholds = np.arange(0.0, 1.01, 0.01)
f1_scores = []
precisions = []
recalls = []

for threshold in thresholds:
    y_val_pred = (y_val_pred_proba >= threshold).astype(int)

    precision = precision_score(y_val, y_val_pred, zero_division=0)
    recall = recall_score(y_val, y_val_pred, zero_division=0)

    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0

    precisions.append(precision)
    recalls.append(recall)
    f1_scores.append(f1)

max_f1_idx = np.argmax(f1_scores)
max_f1_threshold = thresholds[max_f1_idx]
max_f1_value = f1_scores[max_f1_idx]
max_f1_precision = precisions[max_f1_idx]
max_f1_recall = recalls[max_f1_idx]

print(f"Maximum F1 Score:")
print(f"Threshold: {max_f1_threshold:.2f}")
print(f"F1 Score: {max_f1_value:.3f}")
print(f"Precision: {max_f1_precision:.3f}")
print(f"Recall: {max_f1_recall:.3f}")

print("__" * 50)

# Q5. 5-Fold CV

df_full_train = pd.concat([train_df, val_df], axis=0).reset_index(drop=True)

X_full = df_full_train.drop(columns=['converted'])
y_full = df_full_train['converted']

kfold = KFold(n_splits=5, shuffle=True, random_state=1)

auc_scores = []

for fold, (train_idx, val_idx) in enumerate(kfold.split(X_full), 1):

    X_train_fold = X_full.iloc[train_idx]
    y_train_fold = y_full.iloc[train_idx]
    X_val_fold = X_full.iloc[val_idx]
    y_val_fold = y_full.iloc[val_idx]

    train_dicts = X_train_fold.to_dict(orient='records')
    val_dicts = X_val_fold.to_dict(orient='records')

    dv = DictVectorizer(sparse=False)
    X_train_encoded = dv.fit_transform(train_dicts)
    X_val_encoded = dv.transform(val_dicts)

    model = LogisticRegression(solver='liblinear', C=1.0, max_iter=1000)
    model.fit(X_train_encoded, y_train_fold)

    y_val_pred_proba = model.predict_proba(X_val_encoded)[:, 1]

    auc = roc_auc_score(y_val_fold, y_val_pred_proba)
    auc_scores.append(auc)

print(f"\nAUC scores across folds: {[round(score, 3) for score in auc_scores]}")
print(f"Mean AUC: {np.mean(auc_scores):.3f}")
print(f"Standard Deviation: {np.std(auc_scores):.3f}")

print("__" * 50)

#Q6. Hyperparameter Tuning

# C values to test
C_values = [0.000001, 0.001, 1]

# Separate features and target from full training set
X_full = df_full_train.drop(columns=['converted'])  # Update if target name is different
y_full = df_full_train['converted']

results = {}

for C in C_values:

    kfold = KFold(n_splits=5, shuffle=True, random_state=1)
    auc_scores = []

    for fold, (train_idx, val_idx) in enumerate(kfold.split(X_full), 1):
        X_train_fold = X_full.iloc[train_idx]
        y_train_fold = y_full.iloc[train_idx]
        X_val_fold = X_full.iloc[val_idx]
        y_val_fold = y_full.iloc[val_idx]

        train_dicts = X_train_fold.to_dict(orient='records')
        val_dicts = X_val_fold.to_dict(orient='records')

        dv = DictVectorizer(sparse=False)
        X_train_encoded = dv.fit_transform(train_dicts)
        X_val_encoded = dv.transform(val_dicts)

        model = LogisticRegression(solver='liblinear', C=C, max_iter=1000)
        model.fit(X_train_encoded, y_train_fold)

        y_val_pred_proba = model.predict_proba(X_val_encoded)[:, 1]

        auc = roc_auc_score(y_val_fold, y_val_pred_proba)
        auc_scores.append(auc)

    mean_auc = round(np.mean(auc_scores), 3)
    std_auc = round(np.std(auc_scores), 3)

    results[C] = {
        'mean': mean_auc,
        'std': std_auc,
        'scores': auc_scores
    }

for C, metrics in results.items():
    print(f"C = {C:10.6f}: Mean = {metrics['mean']:.3f}, Std = {metrics['std']:.3f}")

