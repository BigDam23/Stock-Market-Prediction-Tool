import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

def fetch_stock_data(ticker):
    stock_data = yf.download(ticker, start="2010-01-01", end="2025-10-28")
    return stock_data

def prepare_data(data):
    features = ['Adj Close', 'High', 'Low', 'Volume']
    data['Prev Adj Close'] = data['Adj Close'].shift(1)
    data.dropna(inplace=True)  # Drop the first row which now contains NaN
    return data[features + ['Prev Adj Close']]

def train_model(X_train, y_train):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    return model, scaler

def evaluate_model(model, scaler, X_test, y_test):
    X_test_scaled = scaler.transform(X_test)
    predictions = model.predict(X_test_scaled)
    mse = mean_squared_error(y_test, predictions)
    print(f"Mean Squared Error: {mse}")
    return predictions

# Fetch and prepare data
data = fetch_stock_data('NVDA')
data = prepare_data(data)

# Sequential data split
train_size = int(len(data) * 0.8)
train_data = data[:train_size]
test_data = data[train_size:]

X_train = train_data[['Prev Adj Close', 'High', 'Low', 'Volume']].values
y_train = train_data['Adj Close'].values
X_test = test_data[['Prev Adj Close', 'High', 'Low', 'Volume']].values
y_test = test_data['Adj Close'].values

# Train the model
model, scaler = train_model(X_train, y_train)
predictions = evaluate_model(model, scaler, X_test, y_test)

# Predict the next day's price using the last known price
last_known_price = np.array([data[['Prev Adj Close', 'High', 'Low', 'Volume']].iloc[-1]])
last_known_price_scaled = scaler.transform(last_known_price)
predicted_next_price = model.predict(last_known_price_scaled)[0]

# Calculate the percentage change
last_actual_price = data['Adj Close'].iloc[-1]
percentage_change = ((predicted_next_price - last_actual_price) / last_actual_price) * 100

print(f"Predicted Close Price for Next Day: ${predicted_next_price:.2f}")
print(f"Percentage Change: {percentage_change:.2f}%")
