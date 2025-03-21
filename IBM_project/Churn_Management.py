import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc

file_path = "Customers.csv"
df_raw = pd.read_csv(file_path, sep=";", encoding="utf-8", on_bad_lines="skip")
df_raw.columns = df_raw.columns.str.strip()

df_original = df_raw.copy()

# Convert CHURN to binary
if "CHURN" in df_raw.columns:
    df_raw["CHURN_BINARY"] = df_raw["CHURN"].apply(lambda x: 1 if x == "Cancelled" else 0)
    df_original["CHURN_BINARY"] = df_raw["CHURN_BINARY"]  # Keep binary in readable output

# Visualize the charts
if "Sex" in df_raw.columns or "Gender" in df_raw.columns:
    gender_col = "Sex" if "Sex" in df_raw.columns else "Gender"
    plt.figure(figsize=(6, 4))
    ax = sns.countplot(x=gender_col, hue="CHURN_BINARY", data=df_raw, palette="Set1")

    for p in ax.patches:
        ax.annotate(f"{int(p.get_height())}",
                    (p.get_x() + p.get_width() / 2, p.get_height()),
                    ha="center", va="bottom", fontsize=10, fontweight="bold")

    plt.xlabel("Gender")
    plt.ylabel("Count")
    plt.title("Customer Churn Distribution by Gender")
    plt.legend(["Not Churned", "Churned"])
    plt.tight_layout()
    plt.show()

# Drop unnecessary columns
columns_to_drop = ["ID"]
if "CHURN" in df_raw.columns:
    columns_to_drop.append("CHURN")

df_model = df_raw.drop(columns=columns_to_drop)
df_model = pd.get_dummies(df_model, drop_first=True)  # Encode
df_model.fillna(df_model.median(), inplace=True)      # Fill missing

# Split data
X = df_model.drop(columns=["CHURN_BINARY"])
y = df_model["CHURN_BINARY"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
clf = DecisionTreeClassifier(criterion="entropy", max_depth=5, random_state=42)
clf.fit(X_train, y_train)

# Predict
y_pred = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)[:, 1]

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Not Churned", "Churned"],
            yticklabels=["Not Churned", "Churned"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()

# ROC curve
fpr, tpr, _ = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, color="blue", label=f"ROC Curve (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], color="gray", linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.tight_layout()
plt.show()

# Find the customers（≥0.85）
df_readable = df_original.loc[X_test.index].copy()
df_readable["CHURN_PROBABILITY"] = y_proba
high_risk_customers = df_readable[df_readable["CHURN_PROBABILITY"] >= 0.85]

high_risk_customers.to_csv("customer_churn.csv", index=False)
print("High-risk customers saved to customer_churn.csv")
