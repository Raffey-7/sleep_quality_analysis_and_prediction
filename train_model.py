import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# 1. Load Dataset
# Note: Kaggle ka standard 'Sleep_health_and_lifestyle_dataset.csv' use karein
df = pd.read_csv('G:\Ml Project\ML Mini-Project\sleep_quality_project\Sleep_health_and_lifestyle_dataset.csv')

print("Initial Data Shape:", df.shape)

# 2. Data Preprocessing
# Missing values in 'Sleep Disorder' indicate 'None'
df['Sleep Disorder'] = df['Sleep Disorder'].fillna('None')

# Drop identifier column
if 'Person ID' in df.columns:
    df.drop(columns=['Person ID'], inplace=True)

# Split Blood Pressure (e.g., "126/83" -> 126 and 83)
df[['Systolic_BP', 'Diastolic_BP']] = df['Blood Pressure'].str.split('/', expand=True).astype(int)
df.drop(columns=['Blood Pressure'], inplace=True)

# Handle categorical variables
categorical_cols = ['Gender', 'Occupation', 'BMI Category']
label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Encode Target Variable ('Quality of Sleep' or 'Sleep Disorder')
# Yahan hum 'Quality of Sleep' ko classify/predict kar rahe hain (Levels: Low, Medium, High)
# Direct 'Quality of Sleep' integer score (1-10) ko target banate hain
X = df.drop(columns=['Quality of Sleep', 'Sleep Disorder'])
y = df['Quality of Sleep']

# 3. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Model Training (Random Forest)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# 6. Evaluation
y_pred = model.predict(X_test_scaled)
print("\n--- Model Evaluation ---")
print("Accuracy:", round(accuracy_score(y_test, y_pred) * 100, 2), "%")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 7. Save Model and Artifacts
artifacts = {
    'model': model,
    'scaler': scaler,
    'label_encoders': label_encoders,
    'feature_names': list(X.columns)
}
joblib.dump(artifacts, 'sleep_model.pkl')
print("\nModel saved successfully as 'sleep_model.pkl'")