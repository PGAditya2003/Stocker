import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import utils

# App title
st.title("📈 Stock Price Prediction (LSTM)")

# Sidebar for user input
st.sidebar.header("Prediction Settings")
ticker = st.sidebar.text_input("Enter Stock Ticker", "AAPL")
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2015-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("today"))

# Load model
model = load_model("models/stock_lstm.keras")

# Main section
if st.sidebar.button("Predict"):
    st.write(f"### Fetching data for {ticker}...")
    data = utils.fetch_stock_data(ticker, start=start_date, end=end_date)
    data = utils.add_technical_indicators(data)

    st.write("### Historical Data")
    st.dataframe(data.tail(10))

    # Prepare data
    features = ['Close', 'Volume', 'rsi', 'macd', 'ema_20']
    X, y, scaler = utils.prepare_lstm_data(data, features)

    # Predict next day price
    last_seq = X[-1].reshape(1, X.shape[1], X.shape[2])
    prediction_scaled = model.predict(last_seq)
    
    # Invert scaling
    padded_pred = np.concatenate((prediction_scaled, np.zeros((prediction_scaled.shape[0], X.shape[2]-1))), axis=1)
    predicted_price = scaler.inverse_transform(padded_pred)[0][0]

    st.success(f"Predicted Next Day Price for {ticker}: **${predicted_price:.2f}**")

    # Plot actual vs predicted (last 200 points for better view)
    st.write("### Actual vs Predicted (Last 200 Points)")
    preds = model.predict(X)
    padded_preds = np.concatenate((preds, np.zeros((preds.shape[0], X.shape[2]-1))), axis=1)
    actual_predicted = scaler.inverse_transform(padded_preds)[:, 0]
    actual_prices = scaler.inverse_transform(np.concatenate((y.reshape(-1, 1), np.zeros((len(y), X.shape[2]-1))), axis=1))[:, 0]

    plt.figure(figsize=(10, 5))
    plt.plot(actual_prices[-200:], color='blue', label='Actual')
    plt.plot(actual_predicted[-200:], color='red', label='Predicted')
    plt.title(f'{ticker} Price Prediction')
    plt.legend()
    st.pyplot(plt)

