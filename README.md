# 🎯 **Stock Prediction App** 📈

Welcome to the **Stock Prediction App**! 🚀 

This powerful app predicts future stock prices using two sophisticated models: **LSTM (Long Short-Term Memory)** for long-term forecasts, and **SARIMA (Seasonal AutoRegressive Integrated Moving Average)** for short-term predictions. Built with cutting-edge machine learning and data visualization techniques, this app empowers you to stay ahead in the stock market game!



## ✨ **Features**:

- 📊 **LSTM Model**: Predict long-term stock trends with deep learning.
- 📅 **SARIMA Model**: Get precise short-term forecasts with time series analysis.
- 🏦 **Database Integration**: Save and retrieve your stock predictions seamlessly.
- 🔒 **Encryption**: Your data is secure, thanks to robust encryption methods.
- 📈 **Interactive Visualizations**: Stunning stock price charts that show both historical and predicted prices.
- 🖥️ **User-Friendly Interface**: Designed with Streamlit to give you a smooth and engaging experience.



## 🔧 **Technologies Used**:

- **Backend**: Python (Flask)
- **Machine Learning Models**: LSTM (for deep learning-based predictions), SARIMA (for time series forecasting)
- **Data Source**: Yahoo Finance API (`yfinance`)
- **Database**: SQLite for saving predictions
- **Security**: Data encryption with `cryptography.fernet`
- **Frontend**: Streamlit for the UI, with custom CSS styling for a sleek experience


## 🚀 **Getting Started**

Ready to try it out? Here’s how you can get the **Stock Prediction App** up and running in no time:

### Step 1: Clone the repository
Start by cloning the project to your local machine:

```bash
git clone https://github.com/your-username/stock-prediction-app.git
cd stock-prediction-app
```

### Step 2: Install dependencies

It's time to install the required Python libraries. You can either set up a virtual environment or install the packages directly.

1. Create a virtual environment (optional, but recommended):

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

2. Install the dependencies:

    ```bash
    pip install -r requirements.txt
    ```

**requirements.txt** includes:

```txt
flask
streamlit
yfinance
pmdarima
statsmodels
matplotlib
pandas
numpy
scikit-learn
cryptography
sqlite3
```

### Step 3: Initialize the database

The app automatically creates the database when you run it, but you can also initialize it manually:

```bash
python db_handler.py
```

### Step 4: Run the app

Fire up the app using Streamlit:

```bash
streamlit run app.py
```

Your app will be live at `http://localhost:8501`!



## 🔮 **How to Use**:

1. **Enter the Stock Ticker**: Type in the stock symbol, such as "AAPL" for Apple or "TSLA" for Tesla.
2. **Choose Prediction Length**: Decide how many days you want to forecast (e.g., 30 days).
3. **Select Prediction Model**: Pick between the **LSTM** and **SARIMA** models.
4. **View Results**: Watch the app generate predictions, and visualize them with interactive graphs!

**Pro Tip**: You can even track multiple stocks and compare different models’ predictions side by side.


## 💻 **File Structure**

Here’s a quick look at the key files in the project:

```
stock-prediction-app/
│
├── app.py                # Main app file with Streamlit interface
├── lstm_model.py         # LSTM model implementation
├── sarima_model.py       # SARIMA model implementation
├── db_handler.py         # Database handler for saving predictions
├── style.css             # Custom CSS for beautiful styling
├── security.py           # Data encryption and security functions
├── logger.py             # Logger configuration for monitoring the app
├── requirements.txt      # Required Python dependencies
└── predictions.db        # SQLite database that stores predictions (auto-created)
```



## 🧠 **How the Models Work**:

### **LSTM (Long Short-Term Memory)**

LSTM is a type of neural network that’s great at learning from sequential data, such as stock prices. It predicts future prices based on past trends, and it works wonders for long-term forecasting.

### **SARIMA (Seasonal ARIMA)**

SARIMA is a classic time series model that works exceptionally well for short-term predictions, especially when the data exhibits seasonality (e.g., daily, weekly, or monthly trends).


## 🔒 **Security Features**:

Your data is safe! The app uses encryption to ensure that all sensitive information, such as stock predictions and model details, is securely stored in the database. Encryption is handled using the `cryptography.fernet` library.



## 📝 **Logging**

The app logs important events (including any errors) into `app.log`. It ensures that everything runs smoothly, and you can check the logs if anything goes wrong. 📜



## 🌱 **Contributing**:

We welcome contributions! Whether it’s fixing bugs, improving the UI, or adding new features, feel free to fork the repo and submit pull requests.



## 📢 **Stay Ahead of the Game**:

With the **Stock Prediction App**, you have a powerful tool at your fingertips to predict stock prices and make informed decisions. Whether you're an investor, data enthusiast, or just curious about stock forecasting, this app is your gateway to better insights. 🚀📈

