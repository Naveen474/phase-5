import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Load the dataset
file_path = "MSFT.csv"
data = pd.read_csv(file_path)

# Convert the 'Date' column to a datetime object
data['Date'] = pd.to_datetime(data['Date'])

# Sort the data by date
data = data.sort_values(by='Date')

# Check for missing values
missing_values = data.isnull().sum()
print("Missing Values:\n", missing_values)

# Handle missing data (you can choose to fill or remove them)
# For simplicity, filling missing values with the previous day's value
data = data.fillna(method='ffill')

# Feature Engineering: Create lag features for past N days
lag_days = 5
for i in range(1, lag_days + 1):
    data[f'Price_Lag_{i}'] = data['Adj Close'].shift(i)

# Select features (X) and target variable (y)
X = data[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'] + [f'Price_Lag_{i}' for i in range(1, lag_days + 1)]]
y = data['Adj Close']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a regression model (Random Forest in this example)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate the Mean Squared Error
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Plotting for Random Forest model
plt.figure(figsize=(12, 6))
plt.plot(y_test.index, y_test.values, label='Actual Prices', color='blue')
plt.plot(y_test.index, y_pred, label='Random Forest Predictions', color='red')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.legend()
plt.title("Random Forest Model Predictions")
plt.show()