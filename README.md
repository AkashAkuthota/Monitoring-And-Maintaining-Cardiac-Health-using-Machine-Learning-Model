# 🫀 Monitoring and Maintaining Cardiac Health using Machine Learning  

<p align="center">
  <b>Hybrid Voting Classifier Based Heart Disease Prediction System</b><br>
</p>

<p align="center">
  <b>Research-Oriented | Hybrid ML Model | Flask Deployment</b><br>
</p>

---
## 📌 **IEEE Published Research Implementation**  
This repository contains the official implementation of an IEEE-published research paper on heart disease prediction using hybrid machine learning models.  
🔗 https://ieeexplore.ieee.org/document/11081197

## 📌 Abstract

Cardiovascular diseases are among the leading causes of death globally.  
Early and accurate detection of heart disease using clinical parameters can significantly improve patient outcomes.

This project implements a **Hybrid Machine Learning System** using a **Voting Classifier** that combines multiple supervised learning algorithms to improve prediction accuracy, stability, and generalization.  
The trained model is deployed through a **Flask-based web application** for real-time prediction.

---

## 🎯 Key Objectives

✔ Design a **Hybrid (Ensemble) Machine Learning Model**  
✔ Compare individual classifiers with a **Voting Classifier**  
✔ Handle **class imbalance using SMOTE**  
✔ Perform **EDA, correlation analysis, and statistical visualization**  
✔ Deploy the final model using **Flask**  
✔ Maintain **strict alignment with research paper & PPT**

---

## 🧠 Dataset Information

| Attribute | Description |
|---------|------------|
| Dataset | `heart_1.csv` |
| Records | 919 |
| Features | Clinical & diagnostic parameters |
| Target | `HeartDisease` (0 = No, 1 = Yes) |

### Key Features
- Age  
- Resting Blood Pressure  
- Cholesterol  
- Fasting Blood Sugar  
- Maximum Heart Rate  
- ST Depression (Oldpeak)  
- Chest Pain Type  
- Resting ECG  
- Exercise Angina  
- ST Slope  

---

## 🔬 Exploratory Data Analysis (EDA)

Performed entirely inside **`model.py`**, producing:

📊 Density plots (numeric features only)  
🔥 Correlation heatmap  
📈 Feature distributions  
🌲 Feature importance (Random Forest)

These plots appear **directly in the terminal execution** to support:
- Statistical interpretation
- Paper & PPT figures
- Result reproducibility

---

## ⚙️ Machine Learning Models Implemented

### Individual Classifiers
- Logistic Regression  
- K-Nearest Neighbors (KNN)  
- Decision Tree  
- Random Forest  

Each model is evaluated using:
- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix
- ROC Curve

---

## 🧩 Hybrid Voting Classifier (Core Contribution)

### Why Voting Classifier?
✔ Combines strengths of multiple models  
✔ Reduces overfitting  
✔ Improves stability  
✔ Produces balanced predictions  

### Models Used in Ensemble
- Logistic Regression  
- KNN  
- Decision Tree  
- Random Forest  

📌 **Soft Voting** is applied to leverage predicted probabilities.

---

## ⚖️ Handling Class Imbalance

To address skewed class distribution:

- **SMOTE (Synthetic Minority Oversampling Technique)** is applied
- Balances training data before model fitting
- Improves recall and fairness

```

Class distribution after SMOTE:
1 → 406
0 → 406

```

---

## 📊 Model Performance (Hybrid Model)

| Metric | Value |
|------|------|
| Accuracy | ~88–89% |
| Precision | Balanced |
| Recall | Balanced |
| F1-score | Stable |

✔ Confusion Matrix  
✔ ROC Curve  
✔ Classification Report  

---

## 🌐 Web Application (Flask)

The trained hybrid model is deployed using **Flask**.

### Application Features
- Step-by-step user input form
- Automatic feature encoding
- Feature scaling using saved scaler
- Prediction probability display
- Clean UI with result visualization

📁 Training and inference are **strictly separated**:
- `model.py` → training + evaluation
- `app.py` → prediction only

---

## 📁 Project Structure

```

Monitoring-And-Maintaining-Cardiac-Health-using-Machine-Learning-Model/
│
├── model.py                 # Training, EDA, evaluation, saving models
├── app.py                   # Flask inference application
├── heart_1.csv              # Dataset
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

````

---

## 🛠 Requirements

All dependencies are listed in `requirements.txt`.

Key libraries:
- Python 3.10
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- Imbalanced-learn
- Flask

Install using:
```bash
pip install -r requirements.txt
````

---

## ▶️ How to Run the Project

### Step 1: Train the Model

```bash
python model.py
```

✔ Performs EDA
✔ Trains individual + hybrid models
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

This project is directly aligned with and implemented based on our peer-reviewed research paper published on IEEE Xplore.

🔗 IEEE Publication Link

👉 Monitoring and Maintaining Cardiac Health Using Machine Learning Models
https://ieeexplore.ieee.org/document/11081197

📌 Publication Details

Publisher: IEEE

Platform: IEEE Xplore Digital Library

Focus:

Hybrid Machine Learning models for heart disease prediction

Ensemble / Voting Classifier approach

Performance comparison across ML algorithms

Practical deployment considerations

---

## 👨‍💻 Author

**Akash Akuthota**
Computer Science Graduate

---

> *This project demonstrates the practical application of ensemble machine learning techniques for real-world healthcare prediction problems.*

