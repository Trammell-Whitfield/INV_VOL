import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
Sequential = tf.keras.models.Sequential
LSTM = tf.keras.layers.LSTM
Dense = tf.keras.layers.Dense
Dropout = tf.keras.layers.Dropout
import matplotlib.pyplot as plt


# 1. Data Collection
# Fetch historical data for a defense sector index or representative companies (e.g., Lockheed Martin)
ticker = "LMT"  # Lockheed Martin as an example
start_date = "2015-01-01"
end_date = "2025-07-11"  # Up to today

data = yf.download(ticker, start=start_date, end=end_date)
data = data[['Close']]  # Use closing prices

# 2. Calculate Volatility (e.g., daily returns standard deviation over a rolling window)
data['Daily_Return'] = data['Close'].pct_change()
data['Volatility'] = data['Daily_Return'].rolling(window=20).std()  # 20-day rolling volatility
data = data.dropna()

# 3. Prepare Data for LSTM
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data['Volatility'].values.reshape(-1, 1))

# Create sequences for LSTM
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:(i + seq_length)])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

seq_length = 60  # Number of time steps to look back
X, y = create_sequences(scaled_data, seq_length)

# Split into training and testing sets
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# 4. Build LSTM Model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(seq_length, 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(units=1))

model.compile(optimizer='adam', loss='mean_squared_error')

# 5. Train the Model
model.fit(X_train, y_train, epochs=60, batch_size=32, validation_data=(X_test, y_test))

# 6. Predict Volatility
y_pred = model.predict(X_test)
y_pred = scaler.inverse_transform(y_pred)
y_test = scaler.inverse_transform(y_test.reshape(-1, 1))

# 7. Visualize Results
plt.figure(figsize=(10, 6))
plt.plot(y_test, label='Actual Volatility')
plt.plot(y_pred, label='Predicted Volatility')
plt.title(f'{ticker} Defense Sector Volatility Prediction')
plt.xlabel('Time(last 30 days)')
plt.ylabel('Volatility')
plt.legend()
plt.show()

# 8. Function to Predict Future Volatility
def predict_future_volatility(model, data, seq_length, days_ahead):
    last_sequence = scaled_data[-seq_length:]
    future_predictions = []
    for _ in range(days_ahead):
        X_pred = last_sequence.reshape((1, seq_length, 1))
        next_pred = model.predict(X_pred, verbose=0)
        future_predictions.append(next_pred[0, 0])
        last_sequence = np.append(last_sequence[1:], next_pred, axis=0)
    return scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

# Predict next 30 days
future_days = 30
future_volatility = predict_future_volatility(model, scaled_data, seq_length, future_days)
print(f"Predicted volatility for the next {future_days} days:\n", future_volatility)