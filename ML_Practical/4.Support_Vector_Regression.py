import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

# Sample Advertising Dataset
data = {
    'TV': [50, 100, 150, 200, 250, 300],
    'Sales': [10, 15, 22, 30, 38, 45]
}

# Create DataFrame
df = pd.DataFrame(data)

# Independent Variable
X = df[['TV']]

# Dependent Variable
y = df['Sales']

# Feature Scaling
sc_X = StandardScaler()
sc_y = StandardScaler()

X_scaled = sc_X.fit_transform(X)
y_scaled = sc_y.fit_transform(y.values.reshape(-1, 1))

# SVR Model
model = SVR(kernel='rbf')

# Train Model
model.fit(X_scaled, y_scaled.ravel())

# Prediction for TV Budget = 220

new_data = pd.DataFrame({'TV': [220]})

prediction = model.predict(
    sc_X.transform(new_data)
)

prediction = sc_y.inverse_transform(
    prediction.reshape(-1, 1)
)

print("Predicted Sales:", prediction[0][0])

# Visualization

plt.scatter(X, y)

# Generate smooth curve points
x_grid = np.arange(
    X['TV'].min(),
    X['TV'].max(),
    1
)

x_grid = x_grid.reshape(-1, 1)

# Predict SVR Curve
y_pred = sc_y.inverse_transform(
    model.predict(
        sc_X.transform(x_grid)
    ).reshape(-1, 1)
)

# Plot Curve
plt.plot(x_grid, y_pred)

plt.xlabel("TV Advertisement Budget")
plt.ylabel("Sales")
plt.title("Support Vector Regression")

plt.show()