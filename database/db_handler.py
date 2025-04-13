import sqlite3
from sqlite3 import Error
import os
from datetime import datetime
from utils.logger import logger
from utils.security import encrypt_data, decrypt_data

# Database file path (stored inside the database folder)
DB_PATH = os.path.join(os.path.dirname(__file__), "predictions.db")

def init_db():
    """Initializes the database and creates the predictions table if it does not exist."""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker TEXT NOT NULL,
                    model_type TEXT NOT NULL,
                    prediction_date TEXT,
                    forecast_data TEXT,
                    model_path TEXT,
                    timestamp TEXT NOT NULL
                );
            """)
            conn.commit()
    except Error as e:
        logger.error(f"Database error during initialization: {e}")

def save_prediction(ticker, model_type, prediction_date, forecast_data, model_path):
    """
    Saves the prediction details into the database.
    
    Parameters:
      ticker (str): The stock ticker.
      model_type (str): "SARIMA" or "LSTM".
      prediction_date (str): The prediction date or period.
      forecast_data (DataFrame or JSON str): The forecasted data.
      model_path (str): File path where the model is saved.
    """
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            if not isinstance(forecast_data, str):
                forecast_data = forecast_data.to_json()
            cursor.execute("""
                INSERT INTO predictions (ticker, model_type, prediction_date, forecast_data, model_path, timestamp)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (ticker, model_type, prediction_date, forecast_data, model_path, timestamp))
            conn.commit()
    except Error as e:
        logger.error(f"Database error: {e}")
