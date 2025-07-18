
# train_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load dataset
data = pd.read_csv(r"C:\Users\HP\Desktop\EmployeSalaryPrediction\data\adult3.csv")  # Replace with your dataset path

# Drop unnecessary columns
data = data.drop(columns=['fnlwgt', 'educational-num', 'marital-status', 'relationship',
                          'race', 'gender', 'capital-gain', 'capital-loss', 'native-country'])

# Drop rows with missing values
data = data.replace("?", pd.NA).dropna()

# Encode categorical variables
label_cols = ['workclass', 'education', 'occupation', 'income']
encoder = LabelEncoder()
for col in label_cols:
    data[col] = encoder.fit_transform(data[col])

# Feature selection
features = ['age', 'education', 'occupation', 'hours-per-week']
X = data[features].copy()
X['experience'] = (data['age'] - 18).clip(lower=0)  # Estimate experience
y = data['income']  # 1 for >50K, 0 for <=50K

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = GradientBoostingClassifier()
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")

# Save model
joblib.dump(model, "best_model.pkl")
print("Model saved as best_model.pkl")
