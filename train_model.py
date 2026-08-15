import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    IsolationForest,
    VotingClassifier,
    StackingClassifier
)

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score
)

from xgboost import XGBClassifier

import matplotlib.pyplot as plt
import seaborn as sns

print("\nLOADING DATASET...\n")

# Load Dataset
df = pd.read_csv("network_traffic_data.csv")

print("Dataset Shape :", df.shape)

# Remove duplicates and null values
df.drop_duplicates(inplace=True)
df.dropna(inplace=True)

# Convert labels into numeric
df["Label"] = df["Label"].map({
    "BENIGN": 0,
    "MALICIOUS": 1
})

# Select numeric columns
numeric_cols = df.select_dtypes(include=np.number).columns

# Feature Engineering
df["mean_feature"] = df[numeric_cols].mean(axis=1)
df["std_feature"] = df[numeric_cols].std(axis=1)
df["max_feature"] = df[numeric_cols].max(axis=1)
df["min_feature"] = df[numeric_cols].min(axis=1)

df["entropy_feature"] = (
    -df[numeric_cols]
    .apply(lambda x: np.sum(x * np.log1p(abs(x) + 1)), axis=1)
)

df["traffic_complexity"] = (
    df["std_feature"] *
    df["mean_feature"]
)

# Threat Intelligence Function
def threat_intelligence(row):

    score = 0

    if row["traffic_complexity"] > 1000:
        score += 40

    if row["entropy_feature"] < -5000:
        score += 25

    if row["std_feature"] > 100:
        score += 15

    if row["max_feature"] > 5000:
        score += 20

    return score

# Generate threat score
df["threat_score"] = df.apply(
    threat_intelligence,
    axis=1
)

# Features and labels
X = df.drop("Label", axis=1)
y = df["Label"]

# Scale data
scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Isolation Forest
anomaly_detector = IsolationForest(
    contamination=0.05,
    random_state=42
)

anomaly_detector.fit(X_train)

# Random Forest
rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=15,
    random_state=42
)

# Extra Trees
et = ExtraTreesClassifier(
    n_estimators=300,
    random_state=42
)

# Gradient Boosting
gb = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.05,
    random_state=42
)

# XGBoost
xgb = XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=8,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric='logloss',
    random_state=42
)

# Voting Classifier
voting_model = VotingClassifier(

    estimators=[
        ('rf', rf),
        ('et', et),
        ('gb', gb),
        ('xgb', xgb)
    ],

    voting='soft'

)

# Stacking Classifier
stacking_model = StackingClassifier(

    estimators=[
        ('rf', rf),
        ('et', et),
        ('gb', gb),
        ('xgb', xgb)
    ],

    final_estimator=LogisticRegression(),

    passthrough=True

)

print("\nTRAINING MODELS...\n")

# Train models
voting_model.fit(X_train, y_train)

stacking_model.fit(X_train, y_train)

# Predictions
voting_pred = voting_model.predict(X_test)

stacking_pred = stacking_model.predict(X_test)

# Final prediction logic
final_pred = []

for v, s in zip(voting_pred, stacking_pred):

    if v + s >= 1:
        final_pred.append(1)
    else:
        final_pred.append(0)

final_pred = np.array(final_pred)

# Metrics
accuracy = accuracy_score(y_test, final_pred)
precision = precision_score(y_test, final_pred)
recall = recall_score(y_test, final_pred)
f1 = f1_score(y_test, final_pred)

print("\nMODEL PERFORMANCE\n")

print("Accuracy  :", round(accuracy * 100, 2), "%")
print("Precision :", round(precision * 100, 2), "%")
print("Recall    :", round(recall * 100, 2), "%")
print("F1 Score  :", round(f1 * 100, 2), "%")

print("\nClassification Report\n")

print(classification_report(y_test, final_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, final_pred)

plt.figure(figsize=(7, 6))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=['Benign', 'Malicious'],
    yticklabels=['Benign', 'Malicious']
)

plt.title("Confusion Matrix")

plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.show()

# Train Random Forest separately
rf.fit(X_train, y_train)

# Feature Importance
importance = rf.feature_importances_

features = X.columns

feature_df = pd.DataFrame({
    "Feature": features,
    "Importance": importance
})

feature_df = feature_df.sort_values(
    by="Importance",
    ascending=True
)

plt.figure(figsize=(12, 8))

plt.barh(
    feature_df["Feature"],
    feature_df["Importance"]
)

plt.title("Feature Importance")

plt.xlabel("Importance Score")

plt.show()

# Threat Score Distribution
plt.figure(figsize=(8, 5))

sns.histplot(
    df["threat_score"],
    bins=30,
    kde=True
)

plt.title("Threat Score Distribution")

plt.xlabel("Threat Score")

plt.show()

# Save Models
pickle.dump(voting_model,
            open("voting_model.pkl", "wb"))

pickle.dump(stacking_model,
            open("stacking_model.pkl", "wb"))

pickle.dump(scaler,
            open("scaler.pkl", "wb"))

pickle.dump(anomaly_detector,
            open("anomaly_detector.pkl", "wb"))

print("\nPROJECT SAVED SUCCESSFULLY\n")

print("Saved Files:")
print("1. voting_model.pkl")
print("2. stacking_model.pkl")
print("3. scaler.pkl")
print("4. anomaly_detector.pkl")