# 💼 Employee Salary Prediction App

This is a Machine Learning-powered **Streamlit web application** that predicts whether an employee earns **more than $50K (`>50K`)** or **less than or equal to $50K (`<=50K`)**, based on key demographic and professional features.

---

## 🔗 Live Demo

🌐 **Live App**: [https://employeesalaryprediction-xcvi.onrender.com](https://employeesalaryprediction-xcvi.onrender.com)

---

## 📌 Use Case

Organizations and HR teams can use this app to:
- Predict employee income levels based on candidate features.
- Perform quick screening of large batches of applicants via CSV upload.
- Integrate basic income prediction in recruitment pipelines or analytics dashboards.

---

## 🚀 Features

- 🎛️ User-friendly sidebar for single employee input
- 📊 Real-time prediction display: `>50K` or `<=50K`
- 📂 CSV batch prediction support for multiple employees
- 📥 Option to download predictions as a CSV file
- 🧠 Model trained using scikit-learn on structured employee data

---

## 🧠 ML Model Details

- Trained using: `RandomForestClassifier`
- Encoded categorical variables: `education`, `occupation`
- Features used:
  - `age`
  - `education` (encoded)
  - `occupation` (encoded)
  - `hours-per-week`
  - `experience`
- Target variable: `income` (`>50K` or `<=50K`)

---

## 🛠 Tech Stack

- 🔮 **Streamlit** — Interactive UI and web app framework
- 🧮 **scikit-learn** — ML model building and evaluation
- 📊 **pandas** — Data wrangling and analysis
- 💾 **joblib** — Model persistence
- ☁️ **Render** — Free hosting for Python web services

---

## 🏗️ Local Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/harshgarg2110/employee-salary-prediction.git
cd employee-salary-prediction

## 🏗️ Local Setup Instructions

### 1. Install Dependencies
```bash
pip install -r requirements.txt




