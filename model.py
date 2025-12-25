import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from imblearn.over_sampling import SMOTE

# ===============================
# 1. Load Dataset
# ===============================
data = pd.read_csv("heart_1.csv")

# ===============================
# 2. Density Plots (LABELED)
# ===============================
numeric_data = data.select_dtypes(include=["int64", "float64"])

fig, axes = plt.subplots(4, 4, figsize=(14, 12))
axes = axes.flatten()

for i, col in enumerate(numeric_data.columns):
    numeric_data[col].plot(kind="density", ax=axes[i])
    axes[i].set_title(col)

for j in range(i + 1, len(axes)):
    axes[j].axis("off")

plt.suptitle("Density Distribution of Clinical Features", y=1.02)
plt.tight_layout()
plt.show()

# ===============================
# 3. Correlation Matrix (NUMERIC ONLY)
# ===============================
plt.figure(figsize=(12, 10))
sns.heatmap(
    numeric_data.corr(),
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    linewidths=0.5
)
plt.title("Correlation Matrix of Heart Disease Features")
plt.show()

# ===============================
# 4. Encoding
# ===============================
data_encoded = pd.get_dummies(
    data,
    columns=["Sex", "ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope"],
    drop_first=True
)

X = data_encoded.drop("HeartDisease", axis=1)
y = data_encoded["HeartDisease"]

# ===============================
# 5. Train-Test Split
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ===============================
# 6. Scaling
# ===============================
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ===============================
# 7. SMOTE
# ===============================
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

print("\nClass distribution after SMOTE:")
print(pd.Series(y_train_smote).value_counts())

# ===============================
# 8. Models
# ===============================
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42)
}

# ===============================
# 9. Individual Model Evaluation
# ===============================
plt.figure(figsize=(8, 6))

for name, model in models.items():
    model.fit(X_train_smote, y_train_smote)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print(f"\n{name} Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(4, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"{name} - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.2f})")

plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves - Individual Models")
plt.legend()
plt.show()

# ===============================
# 10. Hybrid Voting Classifier
# ===============================
voting_model = VotingClassifier(
    estimators=[
        ("lr", models["Logistic Regression"]),
        ("knn", models["KNN"]),
        ("rf", models["Random Forest"]),
        ("dt", models["Decision Tree"])
    ],
    voting="soft"
)

voting_model.fit(X_train_smote, y_train_smote)

y_pred = voting_model.predict(X_test)
y_prob = voting_model.predict_proba(X_test)[:, 1]

print("\nVoting Classifier Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# ===============================
# 11. Hybrid Confusion Matrix
# ===============================
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(4, 4))
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Greens",
    xticklabels=["No Disease", "Disease"],
    yticklabels=["No Disease", "Disease"]
)
plt.title("Hybrid Voting Classifier - Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# ===============================
# 12. Hybrid ROC Curve
# ===============================
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, color="green", label=f"Hybrid Model (AUC={roc_auc:.2f})")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Hybrid Voting Model")
plt.legend()
plt.show()

# ===============================
# 13. Feature Importance
# ===============================
rf_model = models["Random Forest"]
importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": rf_model.feature_importances_
}).sort_values(by="Importance", ascending=False)

plt.figure(figsize=(8, 6))
sns.barplot(x="Importance", y="Feature", data=importance_df.head(10))
plt.title("Top 10 Important Features (Random Forest)")
plt.show()

# ===============================
# 14. Save Artifacts
# ===============================
with open("static/model/model.sav", "wb") as f:
    pickle.dump(voting_model, f)

with open("static/model/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

with open("static/model/features.pkl", "wb") as f:
    pickle.dump(list(X.columns), f)

print("\n✅ Hybrid voting model, scaler, and features saved successfully.")
