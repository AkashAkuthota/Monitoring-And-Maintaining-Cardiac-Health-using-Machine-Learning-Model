# ============================================
# Monitoring and Maintaining Cardiac Health
# Hybrid Voting Classifier – Final Model Script
# ============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    roc_auc_score
)

from imblearn.over_sampling import SMOTE

# ============================================
# 1. Load Dataset
# ============================================
data = pd.read_csv("heart_1.csv")

# ============================================
# 2. Exploratory Data Analysis
# ============================================

# Correlation Heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(data.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap of Clinical Features")
plt.tight_layout()
plt.show()
plt.close()

# Density Plots (numeric only)
numeric_data = data.select_dtypes(include=["int64", "float64"])
numeric_data.plot(
    kind="density",
    subplots=True,
    layout=(4, 4),
    figsize=(14, 12),
    sharex=False
)
plt.suptitle("Density Distribution of Features", fontsize=14)
plt.tight_layout()
plt.show()
plt.close()

# ============================================
# 3. Encode Categorical Variables
# ============================================
data_encoded = pd.get_dummies(
    data,
    columns=[
        "Sex",
        "ChestPainType",
        "RestingECG",
        "ExerciseAngina",
        "ST_Slope"
    ],
    drop_first=True
)

# ============================================
# 4. Split Features & Target
# ============================================
X = data_encoded.drop("HeartDisease", axis=1)
y = data_encoded["HeartDisease"]

# ============================================
# 5. Train-Test Split
# ============================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ============================================
# 6. Handle Class Imbalance (SMOTE)
# ============================================
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

print("\nClass distribution after SMOTE:")
print(y_train_res.value_counts())

# ============================================
# 7. Feature Scaling
# ============================================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_res)
X_test_scaled = scaler.transform(X_test)

# ============================================
# 8. Base Models
# ============================================
lr = LogisticRegression(max_iter=1000)
knn = KNeighborsClassifier(n_neighbors=5)
dt = DecisionTreeClassifier(random_state=42)
rf = RandomForestClassifier(n_estimators=100, random_state=42)

models = {
    "Logistic Regression": lr,
    "KNN": knn,
    "Decision Tree": dt,
    "Random Forest": rf
}

for name, model in models.items():
    model.fit(X_train_scaled, y_train_res)
    preds = model.predict(X_test_scaled)
    print(f"\n{name} Accuracy:", accuracy_score(y_test, preds))
    print(classification_report(y_test, preds))

# ============================================
# 9. Hybrid Voting Classifier
# ============================================
voting_model = VotingClassifier(
    estimators=[
        ("lr", lr),
        ("knn", knn),
        ("dt", dt),
        ("rf", rf)
    ],
    voting="soft"
)

voting_model.fit(X_train_scaled, y_train_res)

# ============================================
# 10. Evaluation – Hybrid Model
# ============================================
y_pred = voting_model.predict(X_test_scaled)

print("\nHybrid Voting Classifier Accuracy:",
      accuracy_score(y_test, y_pred))

print("\nClassification Report:\n",
      classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix – Hybrid Voting Model")
plt.tight_layout()
plt.show()
plt.close()

# ============================================
# 11. ROC Curve & AUC (CRITICAL – PAPER)
# ============================================
y_prob = voting_model.predict_proba(X_test_scaled)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_prob)
auc_score = roc_auc_score(y_test, y_prob)

print("\nROC-AUC Score (Hybrid Voting Model):", auc_score)

plt.figure(figsize=(7, 6))
plt.plot(fpr, tpr, label=f"AUC = {auc_score:.3f}", linewidth=2)
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve – Hybrid Voting Classifier")
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
plt.close()

# ============================================
# 12. Save Model Artifacts
# ============================================
pickle.dump(voting_model, open("static/model/model.sav", "wb"))
pickle.dump(scaler, open("static/model/scaler.pkl", "wb"))
pickle.dump(list(X.columns), open("static/model/features.pkl", "wb"))

print("\n✅ Hybrid voting model, scaler, and features saved successfully.")
