import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Example dataset
data = {
    "Square_Footage": [1500, 2000, 2500, 3000, 3500],
    "Bedrooms": [3, 4, 3, 5, 4],
    "Bathrooms": [2, 3, 2, 4, 3],
    "Price": [300000, 400000, 350000, 500000, 450000]
}

# Create a DataFrame
df = pd.DataFrame(data)

# Features and target
X = df[["Square_Footage", "Bedrooms", "Bathrooms"]]
y = df["Price"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Results
print("Mean Squared Error (MSE):", mse)
print("R-squared (R²):", r2)

# Example prediction
new_house = np.array([[2800, 4, 3]])  # [Square_Footage, Bedrooms, Bathrooms]
predicted_price = model.predict(new_house)
print("Predicted Price for the new house:", predicted_price[0])
