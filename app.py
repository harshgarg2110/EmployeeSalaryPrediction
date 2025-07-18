
# app.py

import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("best_model.pkl")

st.set_page_config(page_title="Employee Salary Prediction", page_icon="💼", layout="centered")
st.title("💼 Employee Salary Prediction App")
st.markdown("Predict whether an employee earns >50K or ≤50K based on their details.")

# Sidebar inputs
st.sidebar.header("Input Employee Details")

age = st.sidebar.slider("Age", 18, 65, 30)
education = st.sidebar.selectbox("Education Level", [
    "Bachelors", "HS-grad", "11th", "Masters", "9th", "Some-college", "Assoc-acdm", "Assoc-voc",
    "7th-8th", "Doctorate", "Prof-school", "5th-6th", "10th", "1st-4th", "Preschool", "12th"
])
occupation = st.sidebar.selectbox("Occupation", [
    "Tech-support", "Craft-repair", "Other-service", "Sales", "Exec-managerial",
    "Prof-specialty", "Handlers-cleaners", "Machine-op-inspct", "Adm-clerical",
    "Farming-fishing", "Transport-moving", "Priv-house-serv", "Protective-serv", "Armed-Forces"
])
hours_per_week = st.sidebar.slider("Hours per week", 1, 80, 40)
experience = st.sidebar.slider("Years of Experience", 0, 47, 5)

# Encoding (same order as training)
education_map = {
    val: idx for idx, val in enumerate([
        "Bachelors", "HS-grad", "11th", "Masters", "9th", "Some-college", "Assoc-acdm", "Assoc-voc",
        "7th-8th", "Doctorate", "Prof-school", "5th-6th", "10th", "1st-4th", "Preschool", "12th"
    ])
}
occupation_map = {
    val: idx for idx, val in enumerate([
        "Tech-support", "Craft-repair", "Other-service", "Sales", "Exec-managerial",
        "Prof-specialty", "Handlers-cleaners", "Machine-op-inspct", "Adm-clerical",
        "Farming-fishing", "Transport-moving", "Priv-house-serv", "Protective-serv", "Armed-Forces"
    ])
}

input_df = pd.DataFrame([{
    'age': age,
    'education': education_map[education],
    'occupation': occupation_map[occupation],
    'hours-per-week': hours_per_week,
    # 'experience': experience
}])

st.write("### 🔍 Input Data Preview")
st.write(input_df)

# Prediction
if st.button("Predict Salary Class"):
    prediction = model.predict(input_df)[0]
    label = ">50K" if prediction == 1 else "<=50K"
    st.success(f"🎯 Prediction: Employee earns **{label}**")

# Batch Prediction
st.markdown("---")
st.markdown("### 📂 Batch Prediction")
uploaded_file = st.file_uploader("Upload a CSV file", type="csv")

if uploaded_file is not None:
    batch_data = pd.read_csv(uploaded_file)

    # Apply the same mapping
    batch_data['education'] = batch_data['education'].map(education_map)
    batch_data['occupation'] = batch_data['occupation'].map(occupation_map)

    st.write("✅ Cleaned Input Preview:")
    st.write(batch_data.head())

    batch_preds = model.predict(batch_data)
    batch_data['Predicted Income'] = ['>50K' if p == 1 else '<=50K' for p in batch_preds]

    st.write("### 📊 Results")
    st.write(batch_data)

    csv = batch_data.to_csv(index=False).encode('utf-8')
    st.download_button("Download Results CSV", csv, "predicted_results.csv", "text/csv")
