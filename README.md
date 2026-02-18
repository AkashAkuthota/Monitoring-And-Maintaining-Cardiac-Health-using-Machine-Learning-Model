# 🫀 Monitoring and Maintaining Cardiac Health using Machine Learning

<p align="center">
  <b>Hybrid Voting Classifier–Based Heart Disease Prediction System</b><br>
</p>

<p align="center">
  <b>Research-Oriented | Hybrid ML Model | FastAPI Deployment</b><br>
</p>

---

## 📌 IEEE Published Research Implementation

This repository contains the implementation aligned with an **IEEE-published research paper** on heart disease prediction using hybrid machine learning models.

🔗 [https://ieeexplore.ieee.org/document/11081197](https://ieeexplore.ieee.org/document/11081197)

---

## 📌 Abstract

Cardiovascular diseases are among the leading causes of mortality worldwide.
Early and accurate detection using clinical parameters can significantly improve patient outcomes and preventive care.

This project implements a **Hybrid Machine Learning System** based on a **Voting Classifier**, integrating multiple supervised learning algorithms to enhance prediction accuracy, robustness, and generalization.
The trained hybrid model is deployed through a **Flask-based web application** for real-time cardiac risk prediction.

---

## 🎯 Key Objectives

✔ Design a **Hybrid (Ensemble) Machine Learning Model**
✔ Compare individual classifiers with a **Voting Classifier**
✔ Handle **class imbalance using SMOTE**
✔ Perform **EDA, correlation analysis, and statistical visualization**
✔ Deploy the final model using **Flask**
✔ Maintain **strict alignment with the IEEE research paper and presentation**

---

## 🧠 Dataset Information

### Baseline Dataset (Repository)

| Attribute | Description                      |
| --------- | -------------------------------- |
| File      | `heart_1.csv`                    |
| Records   | ~900                             |
| Features  | Clinical & diagnostic parameters |
| Target    | `HeartDisease` (0 = No, 1 = Yes) |

This dataset is retained in the repository for **reproducibility and reference**.

---

### Large-Scale Dataset (Final Training)

* Records: **~3200+**
* Features: **12–15 clinically relevant attributes**
* Used for: **final hybrid model training**
* Status: **Intentionally excluded from GitHub** via `.gitignore`

> This ensures a lightweight repository while preserving research-grade experimentation.

---

### Key Clinical Features

* Age
* Sex
* Chest Pain Type
* Resting Blood Pressure
* Serum Cholesterol
* Fasting Blood Sugar
* Resting ECG
* Maximum Heart Rate (Thalach)
* Exercise-Induced Angina
* ST Depression (Oldpeak)
* ST Segment Slope

All features are **consistent across datasets** and clinically relevant.

---

## 🔬 Exploratory Data Analysis (EDA)

EDA is performed entirely inside **`model.py`**, generating:

📊 Density plots (numerical features)
🔥 Correlation heatmap
📈 Feature distributions
🌲 Feature importance (Random Forest)

These visualizations are produced during model execution to support:

* Statistical interpretation
* IEEE paper & PPT figures
* Experimental reproducibility

> Note: Correlation and EDA plots are **dataset-level analyses**, not model-specific.

---

## ⚙️ Machine Learning Models Implemented

### Individual Classifiers

* Logistic Regression
* K-Nearest Neighbors (KNN)
* Decision Tree
* Random Forest

Each model is evaluated using:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix
* ROC Curve

---

## 🧩 Hybrid Voting Classifier (Core Contribution)

### Why Voting Classifier?

✔ Combines strengths of multiple models
✔ Reduces overfitting
✔ Improves stability
✔ Produces balanced predictions

### Ensemble Composition

* Logistic Regression
* KNN
* Decision Tree
* Random Forest

📌 **Soft Voting** is used to aggregate predicted probabilities.

---

## ⚖️ Handling Class Imbalance

To address skewed class distribution:

* **SMOTE (Synthetic Minority Oversampling Technique)** is applied
* Balances the training data before model fitting
* Improves recall and model fairness

```
Class distribution after SMOTE:
0 → 406
1 → 406
```

---

## 📊 Model Performance (Hybrid Model)

| Metric    | Value        |
| --------- | ------------ |
| Accuracy  | ~88–89%      |
| Precision | Balanced     |
| Recall    | Balanced     |
| F1-score  | Stable       |
| ROC-AUC   | High (~0.94) |

✔ Confusion Matrix
✔ ROC Curve
✔ Classification Report

---

## :🌐 Web Application (FastAPI)

The trained hybrid model is deployed using FastAPI for real-time inference and backend API handling.

## Application Features

Step-by-step clinical input form served via Jinja2 templates

Server-side input validation using FastAPI form handling

Automatic categorical encoding aligned with saved feature structure

Feature scaling using persisted StandardScaler

Risk classification with probability score returned in structured format

Clean UI with visual indicators and response rendering

## 📁 Strict separation of concerns:

model.py → Model training, EDA, evaluation, artifact saving

app.py → FastAPI-based prediction endpoints and inference logic

static/ & templates/ → Frontend assets and presentation layer

## Dietary Recommendation Integration

For higher-risk predictions, the application dynamically provides a dietary recommendation link to support preventive cardiac care and lifestyle awareness, demonstrating practical decision-support integration.

---

## 📁 Project Structure

```
Monitoring-And-Maintaining-Cardiac-Health-using-Machine-Learning-Model/
│
├── model.py                 # Training, EDA, evaluation, saving models
├── app.py                   # FastAPI application
├── heart_1.csv              # Baseline dataset
├── requirements.txt         # Dependencies
├── README.md                # Project documentation
│
├── static/
│   ├── model/
│   │   ├── model.sav        # Hybrid Voting Classifier
│   │   ├── scaler.pkl       # StandardScaler
│   │   └── features.pkl    # Feature order reference
│   ├── style/
│   └── script/
│
├── templates/
│   ├── home.html
│   ├── heart_disease.html
│   ├── result.html
│   ├── about.html
│   └── contact.html
```

---

## 🛠 Requirements

All dependencies are listed in `requirements.txt`.

Key libraries:

* Python 3.10
* NumPy
* Pandas
* Matplotlib
* Seaborn
* Scikit-learn
* Imbalanced-learn
* fastapi
* uvicorn
* python-multipart
* jinja2


Install using:

```bash
pip install -r requirements.txt
```

---

## ▶️ How to Run the Project

### Step 1: Train the Model

```bash
python model.py
```

✔ Performs EDA
✔ Trains individual & hybrid models
✔ Displays plots and metrics
✔ Saves model artifacts

---

### Step 2: Run the Web Application

```bash
python app.py
```

Open in browser:

```
http://127.0.0.1:5000/
```

---

## 📄 Research Publication

This project is aligned with our peer-reviewed research published on **IEEE Xplore**.

🔗 **Monitoring and Maintaining Cardiac Health Using Machine Learning Models**
[https://ieeexplore.ieee.org/document/11081197](https://ieeexplore.ieee.org/document/11081197)

**Publisher:** IEEE
**Platform:** IEEE Xplore Digital Library

**Focus Areas:**

* Hybrid Machine Learning models
* Ensemble / Voting Classifier approach
* Comparative ML evaluation
* Practical deployment considerations

---

## 👨‍💻 Author

**Akash Akuthota**
Computer Science Graduate

---

> *This project demonstrates the real-world application of ensemble machine learning techniques for healthcare risk prediction and decision support.*

---

