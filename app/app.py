import streamlit as st
import joblib
import numpy as np

if "results" not in st.session_state:
    st.session_state.results = {
        "Diabetes": None,
        "Heart": None,
        "CKD": None
    }
    
    
# Load models
model_diabetes = joblib.load('../models/model_diabetes.pkl')
model_heart    = joblib.load('../models/model_heart.pkl')
model_ckd      = joblib.load('../models/model_ckd.pkl')

# Page config
st.set_page_config(page_title="AI Diagnostic DSS", page_icon="🏥", layout="centered")

st.title("🏥 AI Diagnostic Decision Support System")
st.write("Select a disease and enter patient details to get a prediction.")

# Sidebar
patient_name = st.sidebar.text_input("Enter Patient Name")
page = st.sidebar.radio("Navigation", ["Diagnosis", "Results Dashboard"])

disease = st.sidebar.selectbox("Select Disease to Diagnose", [
    "Type 2 Diabetes",
    "Heart Disease",
    "Chronic Kidney Disease (CKD)"
])
# ─────────────────────────────────────────
# DIABETES
# ─────────────────────────────────────────
if disease == "Type 2 Diabetes":
    st.header("🩸 Type 2 Diabetes Prediction")

    col1, col2 = st.columns(2)
    with col1:
        pregnancies = st.number_input("Pregnancies", 0, 20, 1)
        glucose     = st.slider("Glucose Level", 0, 200, 120)
        blood_pressure = st.slider("Blood Pressure", 0, 130, 70)
        skin_thickness = st.slider("Skin Thickness", 0, 100, 20)
    with col2:
        insulin     = st.slider("Insulin", 0, 900, 80)
        bmi         = st.slider("BMI", 0.0, 70.0, 25.0)
        dpf         = st.slider("Diabetes Pedigree Function", 0.0, 2.5, 0.5)
        age         = st.slider("Age", 1, 100, 30)

    if st.button("🔍 Predict Diabetes Risk"):
        input_data = np.array([[pregnancies, glucose, blood_pressure,
                                skin_thickness, insulin, bmi, dpf, age]])
        prediction = model_diabetes.predict(input_data)[0]
        probability = model_diabetes.predict_proba(input_data)[0][1] * 100
        st.session_state.results["Diabetes"] = probability

        if prediction == 1:
            st.error("⚠️ High Risk of diabetes ")
        else:
            st.success("✅ Low Risk of diabetes")
        st.progress(probability / 100)
        st.write(f"Confidence: {probability:.1f}%")

# ─────────────────────────────────────────
# HEART DISEASE
# ─────────────────────────────────────────
elif disease == "Heart Disease":
    st.header("❤️ Heart Disease Prediction")


    age      = st.slider("Age", 1, 100, 50)
    trestbps = st.slider("Resting Blood Pressure", 80, 200, 120)
    chol     = st.slider("Cholesterol", 100, 600, 240)
    thalach  = st.slider("Max Heart Rate", 60, 220, 150)
    oldpeak  = st.slider("ST Depression (Oldpeak)", 0.0, 7.0, 1.0)
    cp = st.selectbox("Chest Pain Type (0-3)", [0,1,2,3])

    if st.button("🔍 Predict Heart Disease Risk"):
        input_data = np.array([[age, trestbps, chol, thalach, oldpeak, cp]])
        prediction = model_heart.predict(input_data)[0]
        probability = model_heart.predict_proba(input_data)[0][1] * 100
        st.session_state.results["Heart"] = probability

        if prediction == 1:
            st.error("⚠️ High Risk of Heart Disease ")
        else:
            st.success("✅ Low Risk of Heart Disease")
        st.progress(probability / 100)
        st.write(f"Confidence: {probability:.1f}%")

# ─────────────────────────────────────────
# CKD
# ─────────────────────────────────────────
elif disease == "Chronic Kidney Disease (CKD)":
    st.header("🫘 Chronic Kidney Disease Prediction")

    age  = st.slider("Age", 1, 100, 50)
    bp   = st.slider("Blood Pressure", 50, 180, 80)
    sg   = st.selectbox("Specific Gravity", [1.005, 1.010, 1.015, 1.020, 1.025])
    al   = st.selectbox("Albumin (0-5)", [0, 1, 2, 3, 4, 5])
    su   = st.selectbox("Sugar (0-5)", [0, 1, 2, 3, 4, 5])

    bgr  = st.slider("Blood Glucose Random", 50, 500, 120)
    bu   = st.slider("Blood Urea", 1, 200, 40)
    sc   = st.slider("Serum Creatinine", 0.0, 20.0, 1.0)

    sod  = st.slider("Sodium", 100, 170, 135)
    pot  = st.slider("Potassium", 2.0, 10.0, 4.5)
    hemo = st.slider("Hemoglobin", 3.0, 18.0, 12.0)

    if st.button("🔍 Predict CKD Risk"):
        input_data = np.array([[age, bp, sg, al, su,
                                bgr, bu, sc,
                                sod, pot, hemo]])

        prediction = model_ckd.predict(input_data)[0]
        probability = model_ckd.predict_proba(input_data)[0][1] * 100
        st.session_state.results["CKD"] = probability

        if prediction == 1:
            st.error("⚠️ High Risk of CKD ")
        else:
            st.success("✅ Low Risk of CKD")
        st.progress(probability / 100)
        st.write(f"Confidence: {probability:.1f}%")
            

st.markdown("---")
st.caption("⚠️ This tool is for educational purposes only and is not a substitute for professional medical advice.")

if page == "Results Dashboard":
    st.header("📊 Patient Risk Summary")

    if patient_name:
        st.subheader(f"Patient: {patient_name}")
    else:
        st.subheader("Patient: Unknown")

    results = st.session_state.results

    if all(v is None for v in results.values()):
        st.warning("⚠️ No predictions made yet.")
    else:
        import pandas as pd

        data = {
            "Disease": [],
            "Risk (%)": []
        }

        for disease, value in results.items():
            if value is not None:
                data["Disease"].append(disease)
                data["Risk (%)"].append(value)

        df = pd.DataFrame(data)

        st.bar_chart(df.set_index("Disease"))

        st.write("### Summary:")
        for d, v in results.items():
            if v is not None:
                st.write(f"• {d}: {v:.1f}% risk")