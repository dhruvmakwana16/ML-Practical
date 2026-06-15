import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# Load Dataset
df = pd.read_csv("ML_Practical/Untitled spreadsheet - Sheet1.csv")

# Features and Target
X = df.drop("target", axis=1)
y = df["target"]

# Scaling
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# -------------------------
# SVC Model
# -------------------------

svc = SVC(kernel='rbf')

svc.fit(X_train, y_train)

svc_pred = svc.predict(X_test)

# -------------------------
# Naive Bayes Model
# -------------------------

nb = GaussianNB()

nb.fit(X_train, y_train)

nb_pred = nb.predict(X_test)

# -------------------------
# Evaluation Function
# -------------------------

def evaluate_model(name, y_test, y_pred):

    print("\n", "="*50)
    print(name)
    print("="*50)

    print("Accuracy:",
          accuracy_score(y_test, y_pred))

    print("Precision:",
          precision_score(y_test, y_pred))

    print("Recall:",
          recall_score(y_test, y_pred))

    print("F1 Score:",
          f1_score(y_test, y_pred))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))


evaluate_model("SVC", y_test, svc_pred)

evaluate_model("Naive Bayes", y_test, nb_pred)

# -------------------------
# Save Models
# -------------------------

joblib.dump(svc, "svc_model.pkl")
joblib.dump(nb, "naive_bayes_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("\nModels Saved Successfully")