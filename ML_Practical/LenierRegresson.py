# Linear Regression Example with Scikit-Learn

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Sample data (X = input, y = output)
X = [[1], [2], [3], [4], [5]]
y = [2, 4, 6, 8, 10]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
print("MSE:", mean_squared_error(y_test, y_pred))
print("Slope (Coefficient):", model.coef_)
print("Intercept:", model.intercept_)
