import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout

st.title("📈 Stock Price Prediction (LSTM)")
st.write("Train and predict stock prices in real-time using LSTM.")

# --- User Input ---
ticker = st.text_input("Enter Stock Ticker", "AAPL")
start_date = st.date_input("Start Date", pd.to_datetime("2018-01-01"))
end_date = st.date_input("End Date", pd.to_datetime("2023-01-01"))

# --- Fetch Stock Data ---
st.subheader("Stock Data")
@st.cache_data
def get_data(ticker, start, end):
    data = yf.download(ticker, start=start, end=end)
    return data

data = get_data(ticker, start_date, end_date)
st.line_chart(data["Close"])

# --- Prepare Data for LSTM ---
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(data["Close"].values.reshape(-1,1))

seq_len = 60
X, y = [], []
for i in range(seq_len, len(scaled_data)):
    X.append(scaled_data[i-seq_len:i])
    y.append(scaled_data[i])

X, y = np.array(X), np.array(y)

# --- Build & Train Model (Cached) ---
@st.cache_resource
def train_model(X, y):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(25))
    model.add(Dense(1))

    model.compile(optimizer="adam", loss="mean_squared_error")
    model.fit(X, y, batch_size=32, epochs=5, verbose=0)  # Reduce epochs for speed
    return model

with st.spinner("Training model... This may take 10–20 seconds"):
    model = train_model(X, y)
st.success("✅ Model trained!")

# --- Prediction ---
last_seq = scaled_data[-seq_len:]
last_seq = np.expand_dims(last_seq, axis=0)
prediction_scaled = model.predict(last_seq)
prediction = scaler.inverse_transform(prediction_scaled)

st.subheader("Next Day Predicted Price")
st.write(f"**${prediction[0][0]:.2f}**")

# --- Plot ---
fig, ax = plt.subplots()
ax.plot(data.index, data["Close"], label="Historical Prices")
ax.axhline(prediction[0][0], color="green", linestyle="--", label="Prediction")
ax.set_title(f"{ticker} Price Prediction")
ax.legend()
st.pyplot(fig)
