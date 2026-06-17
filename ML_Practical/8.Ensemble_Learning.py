import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# ==================================
# Load Dataset
# ==================================

df = pd.read_csv("ML_Practical/bank.csv")

print("Dataset Loaded Successfully")
print(df.head())

# ==================================
# Convert Text Columns to Numeric
# ==================================

df = pd.get_dummies(df, drop_first=True)

print("\nDataset After Encoding:")
print(df.head())

# ==================================
# Features and Target
# ==================================

target_column = [col for col in df.columns if col.startswith("y_")][0]

X = df.drop(columns=[target_column])
y = df[target_column]

# ==================================
# Train Test Split
# ==================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ==================================
# Bagging Classifier
# ==================================

bagging = BaggingClassifier(
    estimator=DecisionTreeClassifier(),
    n_estimators=50,
    random_state=42
)

bagging.fit(X_train, y_train)

bag_pred = bagging.predict(X_test)

# ==================================
# Random Forest
# ==================================

rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf.fit(X_train, y_train)

rf_pred = rf.predict(X_test)

# ==================================
# AdaBoost
# ==================================

boost = AdaBoostClassifier(
    n_estimators=50,
    random_state=42
)

boost.fit(X_train, y_train)

boost_pred = boost.predict(X_test)

# ==================================
# Evaluation Function
# ==================================

def evaluate(name, y_true, y_pred):

    print("\n" + "=" * 50)
    print(name)
    print("=" * 50)

    print("Accuracy:", accuracy_score(y_true, y_pred))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))

# ==================================
# Results
# ==================================

evaluate("Bagging Classifier", y_test, bag_pred)

evaluate("Random Forest", y_test, rf_pred)

evaluate("AdaBoost", y_test, boost_pred)

print("\nProgram Executed Successfully")