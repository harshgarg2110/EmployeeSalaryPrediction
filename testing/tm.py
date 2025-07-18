import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load your dataset (replace with actual path)
data = pd.read_csv(r"C:\Users\HP\Desktop\EmployeSalaryPrediction\data\adult3.csv")

# Encode 'education' and 'occupation' exactly as in the app
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

# Map values
data['education'] = data['education'].map(education_map)
data['occupation'] = data['occupation'].map(occupation_map)

# Drop rows with unknown/missing values (if any)
data = data.dropna()

# Define features and target
X = data[['age', 'education', 'occupation', 'hours-per-week']]
y = data['income'].apply(lambda x: 1 if x == ">50K" else 0)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(class_weight="balanced", random_state=42)
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "best_model.pkl")
print("✅ Model trained and saved as best_model.pkl")
