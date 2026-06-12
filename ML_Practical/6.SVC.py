import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# Load Dataset
df = pd.read_csv("ML_Practical/Untitled spreadsheet - Sheet1.csv")
# Features
X = df.drop("target", axis=1)

# Target
y = df["target"]

# Feature Scaling
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Create Model
svc = SVC(kernel='rbf')

# Train
svc.fit(X_train, y_train)

# Prediction
y_pred = svc.predict(X_test)

# Accuracy
print("SVC Accuracy:")
print(accuracy_score(y_test, y_pred))

# Confusion Matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))