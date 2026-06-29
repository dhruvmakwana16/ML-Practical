import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ==========================
# Load Dataset
# ==========================
df = pd.read_csv("ML_Practical/bank.csv")

print("Dataset Loaded Successfully\n")
print(df.head())
print("\nColumns:", df.columns.tolist())

# ==========================
# Encode ALL Categorical Columns
# ==========================
encoders = {}

for col in df.columns:
    if not pd.api.types.is_numeric_dtype(df[col]):
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

print("\nDataset After Encoding:\n")
print(df.head())

# ==========================
# Features & Target
# ==========================
X = df.drop("y", axis=1)
y = df["y"]

# ==========================
# Train/Test Split
# ==========================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# ==========================
# Random Forest Model
# ==========================
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# ==========================
# Train Model
# ==========================
model.fit(X_train, y_train)

# ==========================
# Prediction
# ==========================
y_pred = model.predict(X_test)

# ==========================
# Results
# ==========================
print("\n==========================")
print("Accuracy")
print("==========================")
print(accuracy_score(y_test, y_pred))

print("\n==========================")
print("Confusion Matrix")
print("==========================")
print(confusion_matrix(y_test, y_pred))

print("\n==========================")
print("Classification Report")
print("==========================")
print(classification_report(y_test, y_pred))

# ==========================
# Save Model
# ==========================
joblib.dump(model, "random_forest_model.pkl")
print("\nModel Saved Successfully")

# ==========================
# Load Model
# ==========================
loaded_model = joblib.load("random_forest_model.pkl")
print("Model Loaded Successfully")

# ==========================
# Predict First Record
# ==========================
sample = X.iloc[[0]]

prediction = loaded_model.predict(sample)

print("\nPrediction (Encoded):", prediction[0])

if "y" in encoders:
    print("Prediction (Original):", encoders["y"].inverse_transform(prediction)[0])