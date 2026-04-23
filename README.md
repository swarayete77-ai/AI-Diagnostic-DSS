# 🏥 AI Diagnostic Decision Support System (DSS)

A Machine Learning-based web application that predicts the risk of three major diseases:

• Type 2 Diabetes  
• Heart Disease  
• Chronic Kidney Disease (CKD)

This project demonstrates an end-to-end ML pipeline — from data preprocessing and model training to deployment using Streamlit.

---

## 🎯 Project Objective

The goal of this project is to build a **Decision Support System (DSS)** that assists in early disease risk prediction using patient clinical data.

The system allows users to:
- Input patient health parameters
- Generate real-time predictions using ML models
- Visualize results using a graphical dashboard
- Interpret outputs using **Risk Scores and risk categories**

⚠️ This system is for **educational purposes only** and is not a substitute for professional medical advice.

---

## 🧠 Machine Learning Models

| Disease | Algorithm | Accuracy |
|--------|-----------|----------|
| Diabetes | Random Forest | 72% |
| Heart Disease | Random Forest | 99% |
| Chronic Kidney Disease | Random Forest | 99% |

📌 Note: High accuracy for Heart Disease and CKD is due to relatively small and well-structured datasets, which may lead to overfitting.

---

## 📊 Datasets Used

| Dataset | Target Column |
|--------|---------------|
| diabetes.csv | Outcome |
| heart.csv | target |
| kidney_disease.csv | classification |

All datasets were sourced from Kaggle.

---

## 🔬 Feature Selection (Key Improvement)

To improve usability and real-world applicability:

- Reduced feature complexity  
- Removed categorical and subjective inputs  
- Selected clinically relevant numerical features  

### Final Feature Sets:

**Heart Disease (5 features):**
- Age  
- Resting Blood Pressure  
- Cholesterol  
- Maximum Heart Rate  
- ST Depression (Oldpeak)

**CKD (11 features):**
- Age, Blood Pressure  
- Specific Gravity, Albumin, Sugar  
- Blood Glucose, Urea, Creatinine  
- Sodium, Potassium, Hemoglobin  

📌 This reduces input errors and improves consistency in predictions.

---

## ⚙️ Data Preprocessing (CKD)

- Dropped `id` column  
- Replaced missing values (`?`) with NaN  
- Label encoded categorical variables  
- Converted all features to numeric  
- Filled missing values using median  

---

## 🖥️ Tech Stack

**Language:** Python 3.11  
**IDE:** VS Code  
**Frontend / App:** Streamlit  

### Libraries Used:
- pandas  
- numpy  
- scikit-learn  
- matplotlib  
- seaborn  
- joblib  
- streamlit  

---

## 📂 Project Structure
