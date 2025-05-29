# Import necessary libraries
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Step 1: Download historical stock data
ticker = 'AAPL'  # Change this to any stock symbol you want
start_date = '2015-01-01'
end_date = '2024-12-31'

data = yf.download(ticker, start=start_date, end=end_date)
data = data[['Close']]  # We'll use only the closing price

# Step 2: Create lagged features
data['Prev_Close'] = data['Close'].shift(1)
data = data.dropna()

# Step 3: Define features and target
X = data[['Prev_Close']]
y = data['Close']

# Step 4: Split the data into training and testing sets (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Step 5: Train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 6: Make predictions
y_pred = model.predict(X_test)

# Step 7: Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse:.4f}")
print(f"R² Score: {r2:.4f}")

# Step 8: Visualize the results
plt.figure(figsize=(12, 6))
plt.plot(y_test.values, label='Actual Prices', color='blue')
plt.plot(y_pred, label='Predicted Prices', color='red')
plt.title(f'{ticker} Stock Price Prediction (Linear Regression)')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
