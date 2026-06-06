# To implement Simple Linear Regression using the Scikit-learn library 
# and predict values based on a linear relationship between variables.

# Theory

# Simple Linear Regression is a supervised Machine Learning algorithm used to 
# predict a dependent variable (Y) based on one independent variable (X).

# The equation of a simple linear regression line is:



import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Dataset
data = {
    'Experience': [1, 2, 3, 4, 5],
    'Salary': [25000, 30000, 35000, 40000, 45000]
}

df = pd.DataFrame(data)

# Independent Variable
X = df[['Experience']]

# Dependent Variable
y = df['Salary']

# Create Model
model = LinearRegression()

# Train Model
model.fit(X, y)

# Prediction
predicted_salary = model.predict([[6]])

print("Predicted Salary for 6 Years Experience:", predicted_salary[0])

# Plot
plt.scatter(X, y)
plt.plot(X, model.predict(X))
plt.xlabel("Experience")
plt.ylabel("Salary")
plt.title("Simple Linear Regression")
plt.show()