import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
Sequential = tf.keras.models.Sequential
LSTM = tf.keras.layers.LSTM
Dense = tf.keras.layers.Dense
Dropout = tf.keras.layers.Dropout
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta

# Custom accuracy metrics for volatility prediction
def calculate_accuracy_metrics(y_true, y_pred):
    """Calculate comprehensive accuracy metrics for volatility prediction"""
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true_clean = y_true[mask]
    y_pred_clean = y_pred[mask]
    
    if len(y_true_clean) == 0:
        return {}
    
    rmse = np.sqrt(mean_squared_error(y_true_clean, y_pred_clean))
    mae = mean_absolute_error(y_true_clean, y_pred_clean)
    mape = np.mean(np.abs((y_true_clean - y_pred_clean) / y_true_clean)) * 100
    r2 = r2_score(y_true_clean, y_pred_clean)
    
    if len(y_true_clean) > 1:
        true_direction = np.diff(y_true_clean) > 0
        pred_direction = np.diff(y_pred_clean) > 0
        directional_accuracy = np.mean(true_direction == pred_direction) * 100
    else:
        directional_accuracy = 0
    
    nrmse = rmse / np.mean(y_true_clean) * 100
    smape = np.mean(2 * np.abs(y_true_clean - y_pred_clean) / (np.abs(y_true_clean) + np.abs(y_pred_clean))) * 100
    theil_u = np.sqrt(np.mean((y_pred_clean - y_true_clean)**2)) / np.sqrt(np.mean(y_true_clean**2))
    
    return {
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape,
        'R2': r2,
        'Directional_Accuracy': directional_accuracy,
        'NRMSE': nrmse,
        'SMAPE': smape,
        'Theil_U': theil_u
    }

def print_accuracy_report(metrics, ticker_name):
    """Print a formatted accuracy report"""
    print(f"\n=== {ticker_name} Accuracy Metrics ===")
    print(f"Root Mean Square Error (RMSE):     {metrics['RMSE']:.6f}")
    print(f"Mean Absolute Error (MAE):         {metrics['MAE']:.6f}")
    print(f"Mean Absolute Percentage Error:    {metrics['MAPE']:.2f}%")
    print(f"R-squared (R²):                    {metrics['R2']:.4f}")
    print(f"Directional Accuracy:              {metrics['Directional_Accuracy']:.2f}%")
    print(f"Normalized RMSE:                   {metrics['NRMSE']:.2f}%")
    print(f"Symmetric MAPE:                    {metrics['SMAPE']:.2f}%")
    print(f"Theil's U statistic:               {metrics['Theil_U']:.4f}")
    
    print(f"\nInterpretation:")
    if metrics['R2'] > 0.7:
        print("- Excellent model performance (R² > 0.7)")
    elif metrics['R2'] > 0.5:
        print("- Good model performance (R² > 0.5)")
    elif metrics['R2'] > 0.3:
        print("- Moderate model performance (R² > 0.3)")
    else:
        print("- Poor model performance (R² < 0.3)")
    
    if metrics['Directional_Accuracy'] > 60:
        print(f"- Good directional prediction (>{metrics['Directional_Accuracy']:.1f}% accuracy)")
    else:
        print(f"- Poor directional prediction ({metrics['Directional_Accuracy']:.1f}% accuracy)")
    
    if metrics['MAPE'] < 10:
        print("- Very accurate predictions (MAPE < 10%)")
    elif metrics['MAPE'] < 20:
        print("- Reasonably accurate predictions (MAPE < 20%)")
    else:
        print("- Less accurate predictions (MAPE > 20%)")

# Function to create sequences for LSTM
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:(i + seq_length)])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

# Function to build and train LSTM model
def build_and_train_model(volatility_data, seq_length=60):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(volatility_data.values.reshape(-1, 1))
    X, y = create_sequences(scaled_data, seq_length)
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(seq_length, 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    history = model.fit(X_train, y_train, epochs=60, batch_size=32, validation_data=(X_test, y_test), verbose=0)
    y_pred = model.predict(X_test)
    y_pred = scaler.inverse_transform(y_pred)
    y_test = scaler.inverse_transform(y_test.reshape(-1, 1))
    
    return model, scaler, scaled_data, y_test, y_pred, train_size, history

# Function to predict future volatility
def predict_future_volatility(model, scaler, scaled_data, seq_length, days_ahead):
    last_sequence = scaled_data[-seq_length:]
    future_predictions = []
    for i in range(days_ahead):
        X_pred = last_sequence.reshape((1, seq_length, 1))
        next_pred = model.predict(X_pred, verbose=0)
        future_predictions.append(next_pred[0, 0])
        last_sequence = np.append(last_sequence[1:], next_pred, axis=0)
    return scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

# 1. Get user input for the stock ticker
ticker = input("Enter the stock ticker symbol (e.g., AAPL, MSFT): ").upper()

# 2. Validate the ticker
try:
    stock_info = yf.Ticker(ticker).info
    company_name = stock_info.get('longName', ticker)  # Use ticker if name not available
except Exception as e:
    print(f"Error: Ticker '{ticker}' not found or invalid. Please try again.")
    exit()

# 3. Data Collection
start_date = "2015-01-01"
end_date = "2025-07-11"  # Up to today
data = yf.download(ticker, start=start_date, end=end_date)
original_data = data.copy()  # Keep original data for price plotting
data = data[['Close']]  # Use closing prices

# Calculate Volatility
data['Daily_Return'] = data['Close'].pct_change()
data['Volatility'] = data['Daily_Return'].rolling(window=20).std()
data = data.dropna()

# 4. Train the model
seq_length = 60
future_days = 30

print(f"Training model for {ticker}...")
model, scaler, scaled_data, y_test, y_pred, train_size, history = build_and_train_model(data['Volatility'], seq_length)
future_volatility = predict_future_volatility(model, scaler, scaled_data, seq_length, future_days)

# Calculate accuracy metrics
metrics = calculate_accuracy_metrics(y_test.flatten(), y_pred.flatten())

# Print accuracy report
print_accuracy_report(metrics, company_name)

# 5. Visualization
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle(f'{company_name} ({ticker}) Volatility Analysis', fontsize=16, fontweight='bold')

# Get test period dates
test_start_idx = train_size + seq_length
test_dates = data.index[test_start_idx:]
test_prices = original_data['Close'].loc[test_dates]

# Create future dates for prediction
last_date = data.index[-1]
future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=future_days, freq='D')

# Volatility subplot
axes[0].plot(test_dates, y_test.flatten(), label='Actual Volatility', color='blue', linewidth=2)
axes[0].plot(test_dates, y_pred.flatten(), label='Predicted Volatility', color='blue', linewidth=2, linestyle='--', alpha=0.7)
axes[0].plot(future_dates, future_volatility.flatten(), label='Future Volatility Prediction', color='orange', linewidth=2, linestyle=':')
axes[0].set_ylabel('Volatility (Standard Deviation)', fontsize=10, fontweight='bold')
axes[0].set_title('Volatility Prediction', fontsize=12, fontweight='bold')
axes[0].legend(loc='upper right')
axes[0].grid(True, alpha=0.3)
axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1%}'))

# Add accuracy metrics
acc_text = f'R²: {metrics["R2"]:.3f}\nMAPE: {metrics["MAPE"]:.1f}%\nDir. Acc: {metrics["Directional_Accuracy"]:.1f}%'
axes[0].text(0.02, 0.98, acc_text, transform=axes[0].transAxes, fontsize=9, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

# Stock price subplot
axes[1].plot(test_dates, test_prices, label=f'{ticker} Stock Price', color='blue', linewidth=1.5)
axes[1].set_ylabel('Stock Price ($)', fontsize=10, fontweight='bold')
axes[1].set_title('Stock Price', fontsize=12, fontweight='bold')
axes[1].legend(loc='upper right')
axes[1].grid(True, alpha=0.3)
axes[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:.0f}'))

# Add price statistics
price_stats = f'Avg: ${np.mean(test_prices):.2f}\nMax: ${np.max(test_prices):.2f}\nMin: ${np.min(test_prices):.2f}'
axes[1].text(0.02, 0.98, price_stats, transform=axes[1].transAxes, fontsize=9,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

# Training loss subplot
axes[2].plot(history.history['loss'], label='Training Loss', color='blue', linewidth=2)
axes[2].plot(history.history['val_loss'], label='Validation Loss', color='red', linewidth=2)
axes[2].set_ylabel('Loss (MSE)', fontsize=10, fontweight='bold')
axes[2].set_xlabel('Epoch', fontsize=10, fontweight='bold')
axes[2].set_title('Training History', fontsize=12, fontweight='bold')
axes[2].legend(loc='upper right')
axes[2].grid(True, alpha=0.3)

# Format x-axis dates
for ax in axes[:2]:
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

plt.tight_layout()
plt.show()