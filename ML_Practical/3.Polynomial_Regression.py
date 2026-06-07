import numpy as np
import pandas as pd

# Import Matplotlib library for data visualization
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Step 1: Create Dataset
# Dictionary containing input (RM) and output (Price) values
data = {
    'RM': [4, 5, 6, 7, 8],      # Number of rooms
    'Price': [15, 20, 30, 50, 80]  # House prices
}

# Convert dictionary into a Pandas DataFrame
df = pd.DataFrame(data)

# Step 2: Separate Features and Target
# Independent variable (Input Feature)
X = df[['RM']]

# Dependent variable (Target Value)
y = df['Price']


# Step 3: Create Polynomial Features
# Create polynomial features of degree 2
# Example:
# x = 4
# becomes [1, 4, 16]
poly = PolynomialFeatures(degree=2)

# Transform original feature into polynomial features
X_poly = poly.fit_transform(X)

# Step 4: Train Polynomial Regression Model

# Create Linear Regression object
model = LinearRegression()

# Train model using polynomial features
model.fit(X_poly, y)

# Step 5: Predict New Value

# Predict house price for 9 rooms
new_data = pd.DataFrame({'RM': [9]})
prediction = model.predict(poly.transform(new_data))

# Display predicted price
print("Predicted Price:", prediction[0])

# Step 6: Visualize Original Data
# Plot actual dataset points
plt.scatter(X, y)

# Step 7: Create Smooth Curve
# Generate values from minimum RM to maximum RM
# with step size 0.1
x_grid = np.arange(
    X['RM'].min(),
    X['RM'].max() + 1,
    0.1
)

# Convert 1D array into 2D array
# because Scikit-Learn expects 2D input
x_grid = x_grid.reshape(len(x_grid), 1)

# Step 8: Draw Polynomial Curve
# Plot regression curve
plt.plot(
    x_grid,
    model.predict(
        poly.transform(x_grid)
    )
)

# Step 9: Add Labels and Title

plt.xlabel("Rooms")
plt.ylabel("Price")
plt.title("Polynomial Regression")
plt.show()