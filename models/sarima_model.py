import os
import pandas as pd
import numpy as np
import yfinance as yf
from pmdarima import auto_arima
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import logging
from sklearn.metrics import mean_absolute_error, mean_squared_error
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Make SARIMA tracing configurable via an environment variable (default is False)
TRACE_MODE = os.environ.get("SARIMA_TRACE", "False").lower() == "true"
# Use an environment variable for model save directory (default to 'models/saved_models')
MODEL_SAVE_DIR = os.environ.get("MODEL_SAVE_DIR", "models/saved_models")

def sarima_predict(ticker, days_to_predict):
    """SARIMA prediction with improved hyperparameter tuning and model saving."""
    try:
        # Fetch 3 years of historical data
        end_date = pd.Timestamp.now().strftime('%Y-%m-%d')
        start_date = (pd.Timestamp.now() - pd.Timedelta(days=365*3)).strftime('%Y-%m-%d')
        hist = yf.download(ticker, start=start_date, end=end_date)

        if hist.empty:
            raise ValueError("No data available for the given ticker.")

        # Convert 'Close' column to float
        hist['Close'] = hist['Close'].astype(float)

        # Use seasonal period m=5 (business days)
        model = auto_arima(
            hist['Close'], 
            seasonal=True, 
            m=5,
            stepwise=True, 
            suppress_warnings=True, 
            max_p=5, max_q=5, 
            max_P=3, max_Q=3, 
            d=1, D=1,
            error_action='ignore', 
            trace=TRACE_MODE
        )

        # Fit the SARIMA model
        sarima_model = SARIMAX(hist['Close'], order=model.order, seasonal_order=model.seasonal_order)
        sarima_fit = sarima_model.fit(disp=False)

        # Forecast for the specified number of days
        forecast = sarima_fit.get_forecast(steps=days_to_predict)
        predicted_prices = forecast.predicted_mean

        # Create a forecast index using business days
        forecast_index = pd.date_range(start=hist.index[-1], periods=days_to_predict + 1, freq='B')[1:]
        forecast_df = pd.DataFrame({'Date': forecast_index, 'Predicted': predicted_prices}).set_index('Date')

        # Evaluate on available actual values (if any)
        actual_prices = hist['Close'][-days_to_predict:].values
        predicted_values = forecast_df['Predicted'][:len(actual_prices)].values
        if len(actual_prices) > 0:
            mae = np.round(np.mean(np.abs(actual_prices - predicted_values)), 2)
            rmse = np.round(np.sqrt(np.mean((actual_prices - predicted_values)**2)), 2)
            logger.info(f"Model Evaluation - MAE: {mae}, RMSE: {rmse}")

        # Visualization
        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(14, 7), dpi=200)
        ax.plot(hist.index, hist['Close'], label='Historical Prices', linewidth=2.5, color='#00c8ff', alpha=0.9)
        ax.plot(forecast_df.index, forecast_df['Predicted'], label='SARIMA Forecast', linestyle='--', linewidth=3, color='#ff7300')
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        ax.set_title(f"{ticker} Price Forecast - Next {days_to_predict} Days", fontsize=16, fontweight='bold', color='#ffffff')
        ax.set_xlabel("Date", fontsize=14, fontweight='bold', color='#cccccc')
        ax.set_ylabel("Price (USD)", fontsize=14, fontweight='bold', color='#cccccc')
        ax.legend(fontsize=12, facecolor='#333333', edgecolor='#ffffff')
        ax.grid(True, linestyle='--', alpha=0.3, color='gray')
        plt.xticks(rotation=45, fontsize=11, color='#ffffff')
        plt.yticks(fontsize=11, color='#ffffff')
        plt.gca().set_autoscale_on(True)
        plt.gca().margins(x=0, y=0.1)
        plt.tight_layout()

        # Save the model
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
        model_path = os.path.join(MODEL_SAVE_DIR, f"sarima_{ticker}_{timestamp}.pkl")
        try:
            sarima_fit.save(model_path)
            logger.info(f"SARIMA model saved at {model_path}")
        except Exception as file_e:
            logger.error(f"Error saving SARIMA model: {file_e}")
            model_path = None

        return fig, forecast_df.reset_index(), model_path

    except Exception as e:
        logger.error(f"SARIMA Error: {e}")
        raise
