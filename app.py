import streamlit as st
import pandas as pd
import yfinance as yf
import datetime
from utils.logger import logger
from utils.security import sanitize_input, validate_ticker
from models.sarima_model import sarima_predict
from models.lstm_model import lstm_predict
from database.db_handler import save_prediction, init_db

# Initialize database
init_db()

# Set Streamlit page config
st.set_page_config(page_title="Stock Price Predictor", layout="wide")

# Load custom CSS
with open("static/styles.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Main title and description
st.markdown("<h1 class='center-text'>SMART STOCK PREDICTOR</h1>", unsafe_allow_html=True)
st.markdown(
    "<p class='center-text'>Welcome to the SMART STOCK PREDICTOR! Use our SARIMA-LSTM hybrid model for accurate stock predictions.</p>",
    unsafe_allow_html=True
)

# Sidebar configuration
st.sidebar.header("⚡ Configuration Panel")

# Manage session state (Only to reset when ticker changes)
if "last_ticker" not in st.session_state:
    st.session_state["last_ticker"] = None

# Stock ticker input with validation
ticker = st.sidebar.text_input("Stock Ticker", help="Enter a valid stock ticker symbol (e.g., AAPL, MSFT, TSLA, AMZN).").upper().strip()

# Sanitize and validate ticker
if ticker:
    ticker = sanitize_input(ticker)
    if not validate_ticker(ticker):
        st.sidebar.error("❌ Invalid stock ticker. Please enter a valid one.")
        ticker = None

# Reset session state when ticker changes
if ticker and ticker != st.session_state["last_ticker"]:
    st.session_state["last_ticker"] = ticker

# Fetch stock data

if ticker:
    try:
        stock_info = yf.Ticker(ticker).info
        if not stock_info or "regularMarketPrice" not in stock_info:
            st.sidebar.error("❌ Invalid stock ticker. Please enter a valid ticker.")
            ticker = None
    except Exception as e:
        st.sidebar.error("❌ Failed to fetch data. Please check the ticker or try again later.")
        print(f"Error: {e}")
    else:
        # Display stock info card
        st.markdown(
            f"""
            <div class="stock-card">
                <h3 class='center-text'>{stock_info.get('longName', ticker)}</h3>
                <p><strong>Sector:</strong> {stock_info.get('sector', 'N/A')}</p>
                <p><strong>Current Price:</strong> ${stock_info.get('regularMarketPrice', 'N/A')}</p>
                <p><strong>Market Cap:</strong> ${stock_info.get('marketCap', 'N/A')}</p>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.sidebar.markdown(f"**Live Price: ${stock_info.get('regularMarketPrice', 'N/A')}**")

        # Prediction type selection
        prediction_type = st.sidebar.radio("Select Prediction Type:", ["Short Term (1 - 7 days)", "Long Term (1 - 12 months)"])

        if prediction_type == "Short Term (1 - 7 days)":
            min_date = datetime.date.today() + datetime.timedelta(days=1)
            max_date = datetime.date.today() + datetime.timedelta(days=7)
            selected_date = st.sidebar.date_input(
                "Select prediction date",
                min_date,
                min_value=min_date,
                max_value=max_date,
                help="Select a date within the next 7 days."
            )
            if st.sidebar.button("Predict"):
                days_to_predict = (selected_date - datetime.date.today()).days
                st.sidebar.info(f"Predicting for {days_to_predict} day(s)...")
                try:
                    with st.spinner("Running SARIMA model... Please wait."):
                        fig, forecast_df, model_path = sarima_predict(ticker, days_to_predict)
                        st.success(f"✅ SARIMA Prediction Completed for {ticker}")
                        st.pyplot(fig)
                        st.write("### Forecasted Prices:")
                        st.dataframe(forecast_df)
                        save_prediction(ticker, "SARIMA", str(selected_date), forecast_df, model_path)
                        st.sidebar.info(f"Model saved at {model_path}")
                except Exception as e:
                    st.error("❌ Prediction failed. Please try again.")
                    logger.error(f"SARIMA Prediction Error: {e}")

        elif prediction_type == "Long Term (1 - 12 months)":
            months_to_predict = st.sidebar.slider("Select prediction months", 1, 12, 1, help="Select the number of months for long-term prediction.")
            if st.sidebar.button("Predict"):
                st.sidebar.info(f"Predicting for {months_to_predict} month(s)...")
                try:
                    with st.spinner("Running LSTM model... Please wait."):
                        fig, forecast_df, model_path = lstm_predict(ticker, months_to_predict)
                        st.success(f"✅ LSTM Prediction Completed for {ticker}")
                        st.pyplot(fig)
                        st.write("### Forecasted Prices:")
                        st.dataframe(forecast_df)
                        save_prediction(ticker, "LSTM", f"{months_to_predict} months", forecast_df, model_path)
                        st.sidebar.info(f"Model saved at {model_path}")
                except Exception as e:
                    st.error("❌ Prediction failed. Please try again.")
                    logger.error(f"LSTM Prediction Error: {e}")
