# Linear-Model
Linear Regression Overview Linear regression is a basic and commonly used type of predictive analysis.
Example: Predicting House Prices
Let's say we want to predict house prices based on the size of the house (in square feet).

Step-by-Step Implementation in a Jupyter Notebook
Import the necessary libraries
Create example data
Visualize the data
Create and train the linear regression model
Make predictions
Visualize the regression line
Evaluate the model
Here is the complete implementation:

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Let's create some example data for house sizes (in square feet) and their corresponding prices (in $1000s)
data = {'House Size (sqft)': [750, 800, 850, 900, 950, 1000, 1050, 1100, 1150, 1200],'House Price ($1000s)': [150, 160, 170, 175, 180, 190, 195, 200, 210, 220]}



# Convert the dictionary to a DataFrame
df = pd.DataFrame(data)
df.head()

# Step 3: Visualize the data
plt.scatter(df['House Size (sqft)'], df['House Price ($1000s)'], color='blue')
plt.xlabel('House Size (sqft)')
plt.ylabel('House Price ($1000s)')
plt.title('House Size vs House Price')
plt.show()


# Step 4: Create and train the linear regression model
# Reshape the data
X = df['House Size (sqft)'].values.reshape(-1, 1)     #features
y = df['House Price ($1000s)'].values                 #target

# Create the model
model = LinearRegression()

# Train the model
model.fit(X, y)

# Step 5: Make predictions
y_pred = model.predict(X)

# Step 6: Visualize the regression line
plt.scatter(df['House Size (sqft)'], df['House Price ($1000s)'], color='blue')  # Original data
plt.plot(df['House Size (sqft)'], y_pred, color='red')  # Regression line
plt.xlabel('House Size (sqft)')
plt.ylabel('House Price ($1000s)')
plt.title('House Size vs House Price with Regression Line')
plt.show()



# Step 7: Evaluate the model
# Calculate the mean squared error and the R-squared value
mse = mean_squared_error(y, y_pred).round(2)
r2 = r2_score(y, y_pred).round(2)

print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")



# Display the coefficients
print(f"Intercept: {model.intercept_}")
print(f"Slope: {model.coef_[0]}")

# Summary
For your model:

Mean Squared Error (MSE) of 3.64 indicates that the average squared difference between the actual house prices and the predicted house prices is low, suggesting the predictions are close to the actual values.
R-squared (R²) of 0.99 indicates that 99% of the variance in house prices can be explained by the house size. This implies a very strong relationship between house size and house price in your data.


In conclusion, your linear regression model has performed very well, as evidenced by the low MSE and high R-squared value.
