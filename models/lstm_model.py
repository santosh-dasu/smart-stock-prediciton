import os
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import logging
from datetime import datetime
import tensorflow as tf

# Attempt to enable NPU (via Intel Extension for TensorFlow) if available
try:
    import intel_extension_for_tensorflow as itex
    print("Intel Extension for TensorFlow is enabled for NPU acceleration.")
except ImportError:
    print("Intel Extension for TensorFlow not found, using default TensorFlow.")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Use an environment variable for model save directory (default to 'models/saved_models')
MODEL_SAVE_DIR = os.environ.get("MODEL_SAVE_DIR", "models/saved_models")

def validate_data(stock_data, seq_length):
    """Ensures enough data is available before processing."""
    if stock_data is None or stock_data.empty:
        logger.error("Stock data is empty. Please check the ticker symbol.")
        return False
    if len(stock_data) < seq_length:
        logger.error(f"Not enough data to process. Required: {seq_length}, Available: {len(stock_data)}")
        return False
    return True

def lstm_predict(ticker, months_to_predict):
    """Optimized LSTM prediction with enhanced training callbacks and model improvements."""
    try:
        # Set random seeds for reproducibility
        np.random.seed(42)
        tf.random.set_seed(42)
        
        # Fetch historical data (using 5 years of data)
        end_date = pd.Timestamp.now().strftime('%Y-%m-%d')
        start_date = (pd.Timestamp.now() - pd.Timedelta(days=5*365)).strftime('%Y-%m-%d')
        stock_data = yf.download(ticker, start=start_date, end=end_date)
        
        # Validate data
        seq_length = 60  # Look-back period; consider tuning based on data frequency
        if not validate_data(stock_data, seq_length):
            raise ValueError("Insufficient data for LSTM model.")
        
        # Preprocessing: scale the 'Close' price (expandable to include more features if needed)
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(stock_data[['Close']])
        
        # Create sequences for training
        X, y = [], []
        for i in range(seq_length, len(scaled_data)):
            X.append(scaled_data[i-seq_length:i, 0])
            y.append(scaled_data[i, 0])
        X, y = np.array(X), np.array(y)
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))
        
        # Split data for training & validation (80-20 split)
        split_idx = int(0.8 * len(X))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        # Improved model architecture with bidirectional LSTM, dropout, and potentially more layers
        model = Sequential([
            Bidirectional(LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2]))),
            Dropout(0.3),
            LSTM(32, return_sequences=False),
            Dropout(0.3),
            Dense(1)
        ])
        
        # Use AdamW optimizer for better convergence and regularization
        optimizer = AdamW(learning_rate=0.001, weight_decay=1e-4)
        model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae'])
        
        # Callbacks: EarlyStopping, ReduceLROnPlateau, and ModelCheckpoint to save the best model
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1, min_lr=1e-6)
        checkpoint_path = os.path.join(MODEL_SAVE_DIR, f"lstm_{ticker}_best.h5")
        model_checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_loss', save_best_only=True, verbose=1)
        
        # Train the model (increased epochs with checkpointing so the best epoch is preserved)
        history = model.fit(
            X_train, y_train,
            epochs=100,  # Increased epochs; EarlyStopping will prevent overfitting
            batch_size=32,
            validation_data=(X_val, y_val),
            callbacks=[early_stopping, reduce_lr, model_checkpoint],
            verbose=1
        )
        
        # Log training performance on training and validation sets
        y_pred_train = model.predict(X_train)
        y_pred_val = model.predict(X_val)
        train_mae = np.mean(np.abs(y_train - y_pred_train.flatten()))
        val_mae = np.mean(np.abs(y_val - y_pred_val.flatten()))
        train_rmse = np.sqrt(np.mean((y_train - y_pred_train.flatten())**2))
        val_rmse = np.sqrt(np.mean((y_val - y_pred_val.flatten())**2))
        logger.info(f"Training MAE: {train_mae:.4f}, RMSE: {train_rmse:.4f}")
        logger.info(f"Validation MAE: {val_mae:.4f}, RMSE: {val_rmse:.4f}")
        
        # Future prediction using iterative forecasting for business days (~21 per month)
        current_sequence = scaled_data[-seq_length:]
        predictions = []
        num_prediction_steps = months_to_predict * 21  # Approximate number of business days
        for _ in range(num_prediction_steps):
            x_input = current_sequence[-seq_length:].reshape(1, seq_length, 1)
            pred = model.predict(x_input, verbose=0)
            current_sequence = np.concatenate([current_sequence, pred.reshape(1, 1)], axis=0)
            predictions.append(pred[0, 0])
        
        # Inverse transform the predictions to get actual price values
        predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
        future_dates = pd.date_range(
            start=stock_data.index[-1] + pd.Timedelta(days=1),
            periods=len(predictions),
            freq='B'
        )
        
        # Visualization with dark background
        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(stock_data.index, stock_data['Close'], label='Historical Prices', linewidth=2, color='#1f77b4')
        ax.plot(future_dates, predictions, label='LSTM Forecast', linestyle='--', linewidth=2, color='#ff7f0e')
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.set_title(f"{ticker} Price Forecast - Next {months_to_predict} Months", fontsize=14, color='white')
        ax.set_xlabel("Date", fontsize=12, color='white')
        ax.set_ylabel("Price (USD)", fontsize=12, color='white')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45, color='white')
        plt.yticks(color='white')
        plt.tight_layout()
        
        forecast_df = pd.DataFrame({
            'Date': future_dates,
            'Predicted': predictions.flatten()
        })
        
        # Save the final trained model (also keep the best model from checkpointing)
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        final_model_path = os.path.join(MODEL_SAVE_DIR, f"lstm_{ticker}_{timestamp}.h5")
        try:
            model.save(final_model_path)
            logger.info(f"Final LSTM model saved at {final_model_path}")
        except Exception as file_e:
            logger.error(f"Error saving final LSTM model: {file_e}")
            final_model_path = None
        
        return fig, forecast_df, final_model_path

    except Exception as e:
        logger.error(f"LSTM Error: {e}")
        raise
