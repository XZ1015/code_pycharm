import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve

file_path = "customer_churn.csv"
df = pd.read_csv(file_path)

df.columns = df.columns.str.strip()

df.drop(columns=["ID", "CHURN"], inplace=True)

df = pd.get_dummies(df, drop_first=True)

numeric_cols = ["Usage", "Age", "Est_Income", "LongDistance", "International", "Local"]
for col in numeric_cols:
    if col in df.columns:
        df[col] = df[col].astype(float)

# Check the type of data
print("\n==== type after converting ====\n")
print(df.dtypes)

df.fillna(df.median(numeric_only=True), inplace=True)

X = df.drop(columns=["CHURN_BINARY"])
y = df["CHURN_BINARY"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the models
models = {
    "RandomForest": RandomForestClassifier(),
    "GradientBoosting": GradientBoostingClassifier(),
    "SVM": SVC(probability=True)
}

param_grid = {
    "RandomForest": {"n_estimators": [100], "max_depth": [5, 10, None]},
    "GradientBoosting": {"n_estimators": [100], "learning_rate": [0.05, 0.1]},
    "SVM": {"C": [1, 10], "kernel": ["rbf"]}
}

# Choose the best model
best_model = None
best_auc = 0

for model_name, model in models.items():
    print(f"\nTraining {model_name}...")
    grid_search = GridSearchCV(model, param_grid[model_name], scoring="roc_auc", cv=5)
    grid_search.fit(X_train, y_train)

    best_candidate = grid_search.best_estimator_
    y_proba = best_candidate.predict_proba(X_test)[:, 1]
    auc_score = roc_auc_score(y_test, y_proba)

    print(f"{model_name} AUC: {auc_score:.4f}")

    if auc_score > best_auc:
        best_auc = auc_score
        best_model = best_candidate

# Procast
y_pred = best_model.predict(X_test)
y_proba = best_model.predict_proba(X_test)[:, 1]

# Curve the matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Not Churned", "Churned"], yticklabels=["Not Churned", "Churned"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# Curve the ROC
fpr, tpr, _ = roc_curve(y_test, y_proba)
roc_auc = roc_auc_score(y_test, y_proba)
plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, color="blue", label=f"ROC Curve (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], color="gray", linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()


df_risk = pd.DataFrame(X_test, columns=X.columns)
df_risk["CHURN_PROBABILITY"] = y_proba
high_risk_customers = df_risk[df_risk["CHURN_PROBABILITY"] >= 0.85]
high_risk_customers.to_csv("customer_churn_high_risk.csv", index=False)

print(f"\nBest model: {best_model}")
print("High-risk customers saved to customer_churn_high_risk.csv")

