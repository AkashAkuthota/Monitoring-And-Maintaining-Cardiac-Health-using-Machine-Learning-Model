# Monitoring and Maintaining Cardiac Health using Machine Learning

An end-to-end machine learning–based system for **heart disease prediction**, combining multiple supervised learning models through an **ensemble (Voting / Hybrid Classifier)** approach and deploying the final model using a **Flask web application** for real-time prediction and decision support.

This project was developed as part of the **Bachelor of Engineering (Computer Science & Engineering)** final-year project and is supported by a **peer-reviewed conference paper**.

---

## 📌 Problem Statement

Heart disease remains one of the leading causes of mortality worldwide. Traditional diagnostic approaches often fail to capture complex, non-linear relationships between clinical parameters, leading to delayed or inaccurate diagnosis.

This project aims to:
- Predict the presence of heart disease using patient clinical data
- Improve prediction accuracy using **ensemble learning**
- Provide an **accessible web interface** for real-time diagnosis support
- Assist preventive care through **dietary recommendations**

---

## 🧠 Proposed Solution

The system uses a **Hybrid Voting Classifier** that combines the strengths of multiple supervised machine learning algorithms:

- Logistic Regression (LR)
- K-Nearest Neighbors (KNN)
- Random Forest (RF)
- Decision Tree (DT)

Predictions from individual models are aggregated using **hard/soft voting logic**, resulting in improved robustness and accuracy compared to standalone models.

---

## 🏗️ System Architecture

**Workflow Overview:**

1. Data Collection (Heart Disease Dataset)
2. Data Preprocessing
   - Handling missing values
   - Encoding categorical variables
   - Feature scaling (StandardScaler)
3. Feature Engineering & Selection
4. Model Training
   - LR, KNN, RF, DT
5. Ensemble Learning (Voting / Hybrid Model)
6. Model Evaluation
7. Deployment using Flask Web Application

The final system supports **real-time prediction** via a browser-based interface.

---

## 📊 Dataset Description

The dataset consists of commonly used **clinical parameters**, including:

- Age
- Sex
- Chest Pain Type (cp)
- Resting Blood Pressure (trestbps)
- Serum Cholesterol (chol)
- Fasting Blood Sugar (fbs)
- Resting ECG (restecg)
- Maximum Heart Rate Achieved (thalach)
- Exercise Induced Angina (exang)
- ST Depression (oldpeak)
- Slope of ST Segment (slope)
- Number of Major Vessels (ca)
- Thalassemia (thal)
- Target (Heart Disease: Yes/No)

---

## ⚙️ Technologies Used

**Programming & Frameworks**
- Python
- Flask

**Machine Learning & Data Processing**
- scikit-learn
- Pandas
- NumPy

**Visualization**
- Matplotlib
- Seaborn

**Model Persistence**
- Pickle

---

## 📈 Model Evaluation Metrics

The models were evaluated using standard classification metrics:

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix
- ROC–AUC Curve

### Individual Model Performance (from experiments)

| Model | Accuracy (%) |
|------|--------------|
| Logistic Regression | ~89% |
| KNN | ~87% |
| Random Forest | ~89% |
| Hybrid Voting Model | **Highest** |

The **Hybrid Voting Classifier** showed:
- Reduced false positives
- Reduced false negatives
- Better generalization across samples

---

## 🧪 Visual Analysis Included

- Feature density plots
- Correlation heatmap
- Confusion matrices (per model)
- ROC–AUC curve for Logistic Regression

These visualizations help validate feature relevance and model behavior.

---

## 🌐 Web Application (Flask)

The trained model is integrated into a Flask application that:

- Accepts patient clinical inputs via forms
- Applies preprocessing and scaling
- Generates prediction + probability score
- Displays results in a user-friendly UI

Routes include:
- Home
- About
- Heart Disease Predictor
- Result Page
- Contact

---

## 🥗 Additional Features

- **Personalized dietary recommendations** for patients diagnosed with heart disease
- Supports preventive healthcare alongside diagnosis

---

## 🚀 How to Run the Project

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/AkashAkuthota/Monitoring-And-Maintaining-Cardiac-Health-using-Machine-Learning-Model.git
cd Monitoring-And-Maintaining-Cardiac-Health-using-Machine-Learning-Model

