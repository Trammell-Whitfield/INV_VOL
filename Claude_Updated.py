import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import warnings
import os
from typing import Tuple, Dict, Optional, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class VolatilityPredictor:
    """Enhanced stock volatility prediction using LSTM neural networks"""
    
    def __init__(self, seq_length: int = 60, volatility_window: int = 20):
        self.seq_length = seq_length
        self.volatility_window = volatility_window
        self.model = None
        self.scaler = None
        self.scaled_data = None
        self.history = None
        
    def calculate_advanced_volatility(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate multiple volatility measures"""
        data = data.copy()
        data['Daily_Return'] = data['Close'].pct_change()
        data['Volatility'] = data['Daily_Return'].rolling(window=self.volatility_window).std()
        data['EW_Volatility'] = data['Daily_Return'].ewm(span=self.volatility_window).std()
        if all(col in data.columns for col in ['High', 'Low', 'Open']):
            data['GK_Volatility'] = np.sqrt(
                0.5 * (np.log(data['High'] / data['Low']))**2 - 
                (2 * np.log(2) - 1) * (np.log(data['Close'] / data['Open']))**2
            )
        if all(col in data.columns for col in ['High', 'Low']):
            data['Parkinson_Volatility'] = np.sqrt(
                (1 / (4 * np.log(2))) * (np.log(data['High'] / data['Low']))**2
            )
        return data
    
    def calculate_accuracy_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate comprehensive accuracy metrics for volatility prediction"""
        mask = ~(np.isnan(y_true) | np.isnan(y_pred))
        y_true_clean = y_true[mask]
        y_pred_clean = y_pred[mask]
        if len(y_true_clean) == 0:
            logger.warning("No valid data points for accuracy calculation")
            return {}
        rmse = np.sqrt(mean_squared_error(y_true_clean, y_pred_clean))
        mae = mean_absolute_error(y_true_clean, y_pred_clean)
        r2 = r2_score(y_true_clean, y_pred_clean)
        mape = np.mean(np.abs((y_true_clean - y_pred_clean) / np.maximum(y_true_clean, 1e-8))) * 100
        nrmse = rmse / np.mean(y_true_clean) * 100 if np.mean(y_true_clean) != 0 else np.inf
        smape = np.mean(2 * np.abs(y_true_clean - y_pred_clean) / 
                       (np.abs(y_true_clean) + np.abs(y_pred_clean) + 1e-8)) * 100
        directional_accuracy = 0
        if len(y_true_clean) > 1:
            true_direction = np.diff(y_true_clean) > 0
            pred_direction = np.diff(y_pred_clean) > 0
            directional_accuracy = np.mean(true_direction == pred_direction) * 100
        theil_u = (np.sqrt(np.mean((y_pred_clean - y_true_clean)**2)) / 
                  np.sqrt(np.mean(y_true_clean**2))) if np.mean(y_true_clean**2) != 0 else np.inf
        hit_rate = np.mean(np.abs(y_pred_clean - y_true_clean) / y_true_clean < 0.1) * 100
        return {
            'RMSE': rmse,
            'MAE': mae,
            'MAPE': mape,
            'R2': r2,
            'Directional_Accuracy': directional_accuracy,
            'NRMSE': nrmse,
            'SMAPE': smape,
            'Theil_U': theil_u,
            'Hit_Rate_10pct': hit_rate
        }
    
    def print_accuracy_report(self, metrics: Dict[str, float], ticker_name: str):
        """Print a comprehensive formatted accuracy report"""
        print(f"\n{'='*50}")
        print(f"{ticker_name.center(50)}")
        print(f"{'VOLATILITY PREDICTION ACCURACY METRICS'.center(50)}")
        print(f"{'='*50}")
        print(f"\n📊 Error Metrics:")
        print(f"   Root Mean Square Error (RMSE):     {metrics['RMSE']:.6f}")
        print(f"   Mean Absolute Error (MAE):         {metrics['MAE']:.6f}")
        print(f"   Normalized RMSE:                   {metrics['NRMSE']:.2f}%")
        print(f"\n📈 Percentage Metrics:")
        print(f"   Mean Absolute Percentage Error:    {metrics['MAPE']:.2f}%")
        print(f"   Symmetric MAPE:                    {metrics['SMAPE']:.2f}%")
        print(f"   Hit Rate (±10%):                   {metrics['Hit_Rate_10pct']:.2f}%")
        print(f"\n🎯 Prediction Quality:")
        print(f"   R-squared (R²):                    {metrics['R2']:.4f}")
        print(f"   Directional Accuracy:              {metrics['Directional_Accuracy']:.2f}%")
        print(f"   Theil's U statistic:               {metrics['Theil_U']:.4f}")
        print(f"\n🔍 Model Performance Assessment:")
        if metrics['R2'] > 0.8:
            print("   ✅ Excellent explanatory power (R² > 0.8)")
        elif metrics['R2'] > 0.6:
            print("   ✅ Good explanatory power (R² > 0.6)")
        elif metrics['R2'] > 0.4:
            print("   ⚠️  Moderate explanatory power (R² > 0.4)")
        else:
            print("   ❌ Poor explanatory power (R² < 0.4)")
        if metrics['Directional_Accuracy'] > 65:
            print(f"   ✅ Strong directional prediction ({metrics['Directional_Accuracy']:.1f}% accuracy)")
        elif metrics['Directional_Accuracy'] > 55:
            print(f"   ⚠️  Moderate directional prediction ({metrics['Directional_Accuracy']:.1f}% accuracy)")
        else:
            print(f"   ❌ Weak directional prediction ({metrics['Directional_Accuracy']:.1f}% accuracy)")
        if metrics['MAPE'] < 15:
            print("   ✅ High prediction accuracy (MAPE < 15%)")
        elif metrics['MAPE'] < 25:
            print("   ⚠️  Moderate prediction accuracy (MAPE < 25%)")
        else:
            print("   ❌ Low prediction accuracy (MAPE > 25%)")
        if metrics['Hit_Rate_10pct'] > 70:
            print("   ✅ Excellent precision (>70% within ±10%)")
        elif metrics['Hit_Rate_10pct'] > 50:
            print("   ⚠️  Good precision (>50% within ±10%)")
        else:
            print("   ❌ Poor precision (<50% within ±10%)")
        print(f"{'='*50}\n")
    
    def create_sequences(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for LSTM training with improved error handling"""
        if len(data) <= self.seq_length:
            raise ValueError(f"Data length ({len(data)}) must be greater than sequence length ({self.seq_length})")
        X, y = [], []
        for i in range(len(data) - self.seq_length):
            X.append(data[i:(i + self.seq_length)])
            y.append(data[i + self.seq_length])
        return np.array(X), np.array(y)
    
    def build_model(self, input_shape: Tuple[int, int]) -> Any:
        """Build an improved LSTM model with better architecture"""
        model = Sequential()
        model.add(LSTM(units=100, return_sequences=True, input_shape=input_shape))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))
        model.add(LSTM(units=100, return_sequences=True))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))
        model.add(LSTM(units=50, return_sequences=False))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))
        model.add(Dense(units=50, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(units=25, activation='relu'))
        model.add(Dense(units=1))
        optimizer = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999)
        model.compile(optimizer=optimizer, loss='huber', metrics=['mae'])
        return model
    
    def train_model(self, volatility_data: pd.Series, validation_split: float = 0.2) -> Dict:
        """Train the LSTM model with advanced techniques"""
        logger.info("Preparing data for training...")
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.scaled_data = self.scaler.fit_transform(volatility_data.values.reshape(-1, 1))
        X, y = self.create_sequences(self.scaled_data)
        train_size = int(len(X) * (1 - validation_split))
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        logger.info(f"Training set size: {len(X_train)}, Test set size: {len(X_test)}")
        self.model = self.build_model((self.seq_length, 1))
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=8, min_lr=1e-7, verbose=1)
        ]
        logger.info("Training model...")
        self.history = self.model.fit(
            X_train, y_train,
            epochs=100,
            batch_size=32,
            validation_data=(X_test, y_test),
            callbacks=callbacks,
            verbose=1
        )
        y_pred = self.model.predict(X_test, verbose=0)
        y_pred = self.scaler.inverse_transform(y_pred)
        y_test = self.scaler.inverse_transform(y_test.reshape(-1, 1))
        return {
            'model': self.model,
            'scaler': self.scaler,
            'scaled_data': self.scaled_data,
            'y_test': y_test,
            'y_pred': y_pred,
            'train_size': train_size,
            'history': self.history
        }
    
    def predict_future_volatility(self, days_ahead: int = 30) -> np.ndarray:
        """Predict future volatility with confidence intervals"""
        if self.model is None or self.scaler is None:
            raise ValueError("Model must be trained before making predictions")
        logger.info(f"Predicting volatility for next {days_ahead} days...")
        last_sequence = self.scaled_data[-self.seq_length:].copy()
        future_predictions = []
        n_simulations = 100
        all_predictions = []
        for sim in range(n_simulations):
            predictions = []
            sequence = last_sequence.copy()
            for day in range(days_ahead):
                X_pred = sequence.reshape((1, self.seq_length, 1))
                next_pred = self.model.predict(X_pred, verbose=0)
                if sim > 0:
                    next_pred += np.random.normal(0, 0.01 * np.std(self.scaled_data))
                predictions.append(next_pred[0, 0])
                sequence = np.append(sequence[1:], next_pred, axis=0)
            all_predictions.append(predictions)
        all_predictions = np.array(all_predictions)
        mean_predictions = np.mean(all_predictions, axis=0)
        std_predictions = np.std(all_predictions, axis=0)
        future_predictions = self.scaler.inverse_transform(mean_predictions.reshape(-1, 1))
        confidence_lower = self.scaler.inverse_transform((mean_predictions - 2*std_predictions).reshape(-1, 1))
        confidence_upper = self.scaler.inverse_transform((mean_predictions + 2*std_predictions).reshape(-1, 1))
        return future_predictions, confidence_lower, confidence_upper
    
    def validate_ticker(self, ticker: str) -> Tuple[bool, str]:
        """Validate stock ticker and get company name"""
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            if not info or 'longName' not in info:
                test_data = stock.history(period="5d")
                if test_data.empty:
                    return False, ""
                company_name = info.get('longName', ticker)
            else:
                company_name = info['longName']
            return True, company_name
        except Exception as e:
            logger.error(f"Error validating ticker {ticker}: {e}")
            return False, ""
    
    def get_stock_data(self, ticker: str, start_date: str = "2018-01-01") -> pd.DataFrame:
        """Download and prepare stock data with error handling"""
        try:
            end_date = datetime.now().strftime("%Y-%m-%d")
            logger.info(f"Downloading data for {ticker} from {start_date} to {end_date}")
            data = yf.download(ticker, start=start_date, end=end_date, progress=False)
            if data.empty:
                raise ValueError(f"No data available for ticker {ticker}")
            if len(data) < self.seq_length + self.volatility_window + 50:
                raise ValueError(f"Insufficient data for {ticker}. Need at least {self.seq_length + self.volatility_window + 50} days")
            return data
        except Exception as e:
            logger.error(f"Error downloading data for {ticker}: {e}")
            raise
    
    def create_enhanced_visualization(self, data: pd.DataFrame, results: Dict, 
                                    future_volatility: np.ndarray, confidence_bounds: Tuple,
                                    metrics: Dict, ticker: str, company_name: str):
        """Create clean, organized visualization with properly structured subplots"""
        test_start_idx = results['train_size'] + self.seq_length
        test_dates = data.index[test_start_idx:]
        last_date = data.index[-1]
        future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=30, freq='D')
        confidence_lower, confidence_upper = confidence_bounds
        test_prices = data['Close'].loc[test_dates]
        recent_data = data.tail(252)

        # Create clean subplots with better layout
        from plotly.subplots import make_subplots
        fig = make_subplots(
            rows=2, cols=2,
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]],
            subplot_titles=(
                "Volatility Prediction with Confidence Intervals",
                "Training & Validation Loss",
                "Prediction Accuracy vs Actual",
                "Multiple Volatility Measures (Last Year)"
            ),
            vertical_spacing=0.12,
            horizontal_spacing=0.1
        )

        # 1. Main Volatility Prediction Chart (Top Left)
        fig.add_trace(
            go.Scatter(x=test_dates, y=results['y_test'].flatten(), 
                      name="Actual Volatility", 
                      line=dict(color="#1f77b4", width=2.5),
                      legendgroup="main"),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=test_dates, y=results['y_pred'].flatten(), 
                      name="Predicted Volatility", 
                      line=dict(color="#ff7f0e", width=2, dash="dash"),
                      legendgroup="main"),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=future_dates, y=future_volatility.flatten(), 
                      name="Future Prediction", 
                      line=dict(color="#d62728", width=3),
                      legendgroup="future"),
            row=1, col=1
        )
        
        # Confidence intervals
        fig.add_trace(
            go.Scatter(x=future_dates, y=confidence_upper.flatten(), 
                      name="95% Confidence", 
                      line=dict(color="rgba(214, 39, 40, 0)", width=0), 
                      showlegend=False,
                      hoverinfo='skip'),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=future_dates, y=confidence_lower.flatten(), 
                      name="95% Confidence Band", 
                      fill="tonexty", 
                      fillcolor="rgba(214, 39, 40, 0.2)", 
                      line=dict(color="rgba(214, 39, 40, 0)", width=0),
                      legendgroup="future"),
            row=1, col=1
        )

        # 2. Training History (Top Right)
        fig.add_trace(
            go.Scatter(x=list(range(1, len(results['history'].history['loss'])+1)), 
                      y=results['history'].history['loss'], 
                      name="Training Loss", 
                      line=dict(color="#2ca02c", width=2),
                      legendgroup="training"),
            row=1, col=2
        )
        fig.add_trace(
            go.Scatter(x=list(range(1, len(results['history'].history['val_loss'])+1)), 
                      y=results['history'].history['val_loss'], 
                      name="Validation Loss", 
                      line=dict(color="#ff7f0e", width=2),
                      legendgroup="training"),
            row=1, col=2
        )

        # 3. Prediction Accuracy Scatter (Bottom Left)
        min_val = min(results['y_test'].min(), results['y_pred'].min())
        max_val = max(results['y_test'].max(), results['y_pred'].max())
        
        fig.add_trace(
            go.Scatter(x=results['y_test'].flatten(), y=results['y_pred'].flatten(), 
                      mode="markers", 
                      name="Predictions", 
                      marker=dict(color="#9467bd", size=6, opacity=0.7),
                      legendgroup="accuracy"),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(x=[min_val, max_val], y=[min_val, max_val], 
                      mode="lines", 
                      name="Perfect Prediction", 
                      line=dict(color="#d62728", dash="dash", width=2),
                      legendgroup="accuracy"),
            row=2, col=1
        )

        # 4. Multiple Volatility Measures (Bottom Right)
        if 'EW_Volatility' in recent_data.columns:
            fig.add_trace(
                go.Scatter(x=recent_data.index, y=recent_data['Volatility'], 
                          name="Rolling Vol (20d)", 
                          line=dict(color="#1f77b4", width=2),
                          legendgroup="volatility"),
                row=2, col=2
            )
            fig.add_trace(
                go.Scatter(x=recent_data.index, y=recent_data['EW_Volatility'], 
                          name="Exp. Weighted Vol", 
                          line=dict(color="#ff7f0e", width=2),
                          legendgroup="volatility"),
                row=2, col=2
            )
        
        if 'GK_Volatility' in recent_data.columns:
            fig.add_trace(
                go.Scatter(x=recent_data.index, y=recent_data['GK_Volatility'], 
                          name="Garman-Klass Vol", 
                          line=dict(color="#2ca02c", width=1.5),
                          opacity=0.8,
                          legendgroup="volatility"),
                row=2, col=2
            )
        
        if 'Parkinson_Volatility' in recent_data.columns:
            fig.add_trace(
                go.Scatter(x=recent_data.index, y=recent_data['Parkinson_Volatility'], 
                          name="Parkinson Vol", 
                          line=dict(color="#d62728", width=1.5),
                          opacity=0.8,
                          legendgroup="volatility"),
                row=2, col=2
            )

        # Add performance metrics annotation
        fig.add_annotation(
            x=0.02, y=0.98, xref="x domain", yref="y domain",
            text=f"<b>Model Performance</b><br>" +
                 f"R²: {metrics['R2']:.3f}<br>" +
                 f"MAPE: {metrics['MAPE']:.1f}%<br>" +
                 f"Directional Accuracy: {metrics['Directional_Accuracy']:.1f}%<br>" +
                 f"Hit Rate (±10%): {metrics['Hit_Rate_10pct']:.1f}%",
            showarrow=False, 
            bgcolor="rgba(255, 255, 255, 0.9)", 
            bordercolor="lightgray",
            borderwidth=1,
            font=dict(size=10),
            row=1, col=1
        )

        # Update layout with clean styling
        fig.update_layout(
            title=dict(
                text=f"{company_name} ({ticker}) - Volatility Analysis & Prediction",
                font=dict(size=20),
                x=0.5
            ),
            showlegend=True,
            legend=dict(
                orientation="v",
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=1.02,
                font=dict(size=10)
            ),
            height=800,
            width=1400,
            template="plotly_white",
            margin=dict(l=80, r=120, t=80, b=60)
        )

        # Update axes with proper formatting
        # Volatility chart
        fig.update_xaxes(title_text="Date", row=1, col=1)
        fig.update_yaxes(title_text="Volatility", tickformat=".1%", row=1, col=1)
        
        # Training loss chart
        fig.update_xaxes(title_text="Epoch", row=1, col=2)
        fig.update_yaxes(title_text="Loss (Huber)", row=1, col=2)
        
        # Accuracy scatter
        fig.update_xaxes(title_text="Actual Volatility", tickformat=".1%", row=2, col=1)
        fig.update_yaxes(title_text="Predicted Volatility", tickformat=".1%", row=2, col=1)
        
        # Multiple volatility measures
        fig.update_xaxes(title_text="Date", row=2, col=2)
        fig.update_yaxes(title_text="Volatility", tickformat=".1%", row=2, col=2)

        return fig

def main():
    """Main execution function"""
    predictor = VolatilityPredictor(seq_length=60, volatility_window=20)
    print("🔮 Advanced Stock Volatility Predictor")
    print("="*50)
    while True:
        ticker = input("\n📈 Enter stock ticker symbol (e.g., AAPL, MSFT, TSLA): ").upper().strip()
        if not ticker:
            print("❌ Please enter a valid ticker symbol.")
            continue
        is_valid, company_name = predictor.validate_ticker(ticker)
        if not is_valid:
            print(f"❌ Ticker '{ticker}' not found or invalid. Please try again.")
            continue
        print(f"✅ Found: {company_name} ({ticker})")
        break
    try:
        data = predictor.get_stock_data(ticker)
        print(f"✅ Downloaded {len(data)} days of data")
        data = predictor.calculate_advanced_volatility(data)
        data = data.dropna()
        print(f"✅ Calculated volatility measures")
        print(f"📊 Using {len(data)} data points for analysis")
        results = predictor.train_model(data['Volatility'])
        metrics = predictor.calculate_accuracy_metrics(
            results['y_test'].flatten(), 
            results['y_pred'].flatten()
        )
        predictor.print_accuracy_report(metrics, company_name)
        future_volatility, confidence_lower, confidence_upper = predictor.predict_future_volatility(30)
        print(f"🔮 Generated 30-day volatility forecast with confidence intervals")
        fig = predictor.create_enhanced_visualization(
            data, results, future_volatility, (confidence_lower, confidence_upper),
            metrics, ticker, company_name
        )
        fig.show()
        print(f"\n📋 Future Volatility Summary (Next 30 days):")
        print(f"   Average Predicted Volatility: {np.mean(future_volatility):.2%}")
        print(f"   Maximum Predicted Volatility: {np.max(future_volatility):.2%}")
        print(f"   Minimum Predicted Volatility: {np.min(future_volatility):.2%}")
        print(f"\n✅ Analysis complete! Check the visualization for detailed insights.")
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        print(f"❌ Error: {e}")
        return

if __name__ == "__main__":
    main()