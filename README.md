# 🏥 AI Diagnostic Decision Support System (DSS)

A Machine Learning-based clinical web application that predicts the risk of three major chronic diseases using patient health parameters. Built as an end-to-end ML pipeline — from raw data preprocessing and model training to real-time deployment using Streamlit.

> ⚠️ **Disclaimer:** This system is for **educational and research purposes only** and is not a substitute for professional medical diagnosis or advice.

---

## 🎯 Project Objective

Chronic diseases like Diabetes, Heart Disease, and Kidney Disease are among the leading causes of death globally. Early detection significantly improves patient outcomes. This project builds a **Decision Support System (DSS)** that assists clinicians and researchers in predicting disease risk from routine clinical measurements.

The system allows users to:
- Input patient health parameters through a simple web interface
- Receive real-time risk predictions from trained ML models
- View confidence scores and risk categories for each disease
- Interpret results in a clean, physician-friendly dashboard

---

## 🧠 Machine Learning Models

| Disease | Algorithm | Accuracy | AUC-ROC |
|---|---|---|---|
| Type 2 Diabetes | Random Forest | 72% | 0.81 |
| Heart Disease | Random Forest | 99% | 1.00 |
| Chronic Kidney Disease (CKD) | Random Forest | 99% | 1.00 |

> 📌 **Note on high accuracy:** The Heart Disease and CKD datasets are relatively small and well-structured, which contributes to high in-sample accuracy. The Diabetes model (72%) is more representative of real-world performance on unseen data. All models use Random Forest classifiers with 100 estimators.

---

## 📊 Datasets Used

All datasets were sourced from [Kaggle](https://www.kaggle.com) and are based on publicly available UCI Machine Learning Repository data.

| Dataset | Source | Rows | Features | Target Column |
|---|---|---|---|---|
| `diabetes.csv` | Pima Indians Diabetes Database (UCI) | 768 | 9 | `Outcome` (0/1) |
| `heart.csv` | Heart Disease UCI | 1025 | 14 | `target` (0/1) |
| `kidney_disease.csv` | Chronic Kidney Disease (UCI) | 400 | 26 | `classification` (ckd/notckd) |

### Disease Descriptions

**Type 2 Diabetes**
Diabetes is a chronic condition where the body cannot regulate blood glucose properly. Features include glucose levels, BMI, age, insulin, and family history indicators. The dataset contains female patients of Pima Indian heritage aged 21 and above.

**Heart Disease**
Cardiovascular disease is the leading cause of death globally. Features include age, cholesterol, resting blood pressure, ECG results, and exercise-induced measurements. Target indicates presence (1) or absence (0) of heart disease.

**Chronic Kidney Disease (CKD)**
CKD is progressive loss of kidney function, often caused by Diabetes and Hypertension. Features include creatinine, hemoglobin, blood urea, urine tests, and comorbidity flags. The dataset contains patients from a hospital in Tamil Nadu, India.

---

## 🔬 Feature Engineering & Selection

To improve usability and real-world applicability, key clinically relevant features were selected for each disease:

**Type 2 Diabetes (8 features)**
Pregnancies, Glucose, Blood Pressure, Skin Thickness, Insulin, BMI, Diabetes Pedigree Function, Age

**Heart Disease (13 features)**
- Age  
- Resting Blood Pressure  
- Cholesterol  
- Maximum Heart Rate  
- ST Depression (Oldpeak)

**Chronic Kidney Disease (24 features)**
- Age, Blood Pressure  
- Specific Gravity, Albumin, Sugar  
- Blood Glucose, Urea, Creatinine  
- Sodium, Potassium, Hemoglobin 

---

## ⚙️ Data Preprocessing

**Diabetes**
- No missing values — dataset used as-is
- Features and target separated, 80/20 train-test split applied

**Heart Disease**
- No missing values — dataset used as-is
- Binary classification: 0 = no disease, 1 = disease present

**CKD**
- Dropped `id` column (non-informative)
- Replaced `?` and `\t?` placeholder values with NaN
- Label encoded all categorical text columns (e.g. yes/no → 1/0)
- Converted all columns to numeric using `pd.to_numeric`
- Filled remaining missing values using column median
- Target column mapped: `ckd` → 1, `notckd` → 0

---

## 🖥️ Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3.11 |
| IDE | VS Code |
| Web App | Streamlit |
| ML Library | scikit-learn |
| Model Serialization | joblib |

### Libraries Used

```
pandas
numpy
scikit-learn
matplotlib
seaborn
joblib
streamlit
```

---

## 📂 Project Structure

```
diagnostic-dss/
├── data/
│   ├── diabetes.csv
│   ├── heart.csv
│   └── kidney_disease.csv
├── notebooks/
│   └── 01_data_exploration.ipynb   ← data cleaning, EDA, model training
├── models/
│   ├── model_diabetes.pkl
│   ├── model_heart.pkl
│   └── model_ckd.pkl
├── app/
│   └── app.py                      ← Streamlit web application
├── venv/                           ← virtual environment (not committed to git)
└── README.md
```

---

## 🚀 How to Run the App Locally

**1. Clone the repository**
```bash
git clone https://github.com/yourusername/diagnostic-dss.git
cd diagnostic-dss
```

**2. Create and activate a virtual environment**
```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux
```

**3. Install dependencies**
```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib streamlit
```

**4. Train the models (run the notebook first)**
Open `notebooks/01_data_exploration.ipynb` and run all cells. This saves the three model files to the `models/` folder.

**5. Launch the Streamlit app**
```bash
cd app
streamlit run app.py
```

The app will open automatically in your browser at `http://localhost:8501`

---

## 📈 Model Performance Summary

**Diabetes Model**
The Random Forest classifier achieved 72% accuracy and an AUC-ROC of 0.81 on the held-out test set. Precision for positive cases (diabetic) is 61%, reflecting the inherent difficulty of the dataset. This is consistent with published results on the Pima Indians dataset.

**Heart Disease Model**
Achieved 99% accuracy and perfect AUC-ROC on the test set. This high performance is partly attributed to the structure of this version of the UCI Heart Disease dataset, which contains a balanced and well-separated class distribution.

**CKD Model**
Achieved 99% accuracy and perfect AUC-ROC. The CKD dataset has very clear clinical separators between CKD and non-CKD patients (e.g. creatinine, hemoglobin), making it highly learnable for tree-based models.

---

## 🔮 Future Improvements

- Add SHAP explainability charts to show which features drove each prediction
- Integrate a Bayesian network layer for probabilistic disease co-occurrence modelling
- Expand to additional diseases (e.g. hypertension, liver disease)
- Connect to real EHR systems via FHIR API
- Add patient history tracking across sessions
- Deploy to Streamlit Cloud for public access

---

## 👨‍💻 Author

Built as a portfolio project demonstrating an end-to-end AI-based Diagnostic Decision Support System using machine learning classification and probabilistic reasoning on Electronic Health Record data.
