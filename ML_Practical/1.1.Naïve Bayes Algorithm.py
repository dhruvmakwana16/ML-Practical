# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.naive_bayes import GaussianNB
# from sklearn.metrics import accuracy_score

# # Load Dataset

# df = pd.read_csv("ML_Practical/Untitled spreadsheet - Sheet1.csv")

# # Features
# X = df[['Age', 'GPA']]

# # Target
# y = df['Grade Level']

# # Split Dataset

# X_train, X_test, y_train, y_test = train_test_split(
#     X,
#     y,
#     test_size=0.2,
#     random_state=42
# )

# # Create Model

# model = GaussianNB()

# # Train Model

# model.fit(X_train, y_train)

# # Prediction

# y_pred = model.predict(X_test)

# # Accuracy

# accuracy = accuracy_score(y_test, y_pred)

# print("Accuracy:", accuracy)

# # New Student Prediction

# new_student = pd.DataFrame({
#     'Age': [16],
#     'GPA': [3.5]
# })

# prediction = model.predict(new_student)

# print("Predicted Grade Level:", prediction[0])


import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

# Load Dataset
df = pd.read_csv("ML_Practical/Untitled spreadsheet - Sheet1.csv")

# -----------------------------
# Data Preprocessing
# -----------------------------

# Remove unwanted columns
df = df[['Age',
         'GPA',
         'Extracurricular Activities',
         'Grade Level']]

# Encode Categorical Feature
activity_encoder = LabelEncoder()
df['Extracurricular Activities'] = activity_encoder.fit_transform(
    df['Extracurricular Activities']
)

# Encode Target
grade_encoder = LabelEncoder()
df['Grade Level'] = grade_encoder.fit_transform(
    df['Grade Level']
)

# Features
X = df[['Age',
        'GPA',
        'Extracurricular Activities']]

# Target
y = df['Grade Level']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# -----------------------------
# Model
# -----------------------------

model = GaussianNB()

model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# -----------------------------
# Performance Evaluation
# -----------------------------

print("Accuracy:")
print(accuracy_score(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# -----------------------------
# Multiple Predictions
# -----------------------------

new_students = pd.DataFrame({
    'Age':[14,16,18],
    'GPA':[3.8,3.2,2.5],
    'Extracurricular Activities':[
        'Robotics',
        'Drama Club',
        'Basketball'
    ]
})

new_students['Extracurricular Activities'] = activity_encoder.transform(
    new_students['Extracurricular Activities']
)

predictions = model.predict(new_students)

predictions = grade_encoder.inverse_transform(
    predictions
)

print("\nPredicted Grade Levels:")

for i,pred in enumerate(predictions):
    print(f"Student {i+1}: {pred}")