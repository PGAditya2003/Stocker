import yfinance as yf
import pandas as pd
import ta
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Fetch stock data
def fetch_stock_data(ticker="AAPL", start="2015-01-01", end="2024-12-31"):
    data = yf.download(ticker, start=start, end=end)
    # Flatten multi-index columns (if any)
    data.columns = data.columns.get_level_values(0)
    data.dropna(inplace=True)
    return data

# Add technical indicators
def add_technical_indicators(df):
    df['rsi'] = ta.momentum.RSIIndicator(df['Close']).rsi()
    df['macd'] = ta.trend.MACD(df['Close']).macd()
    df['ema_20'] = df['Close'].ewm(span=20, adjust=False).mean()
    df.dropna(inplace=True)
    return df

# Prepare LSTM data
def prepare_lstm_data(df, features, seq_length=60):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[features])
    
    X, y = [], []
    for i in range(seq_length, len(scaled_data)):
        X.append(scaled_data[i-seq_length:i])
        y.append(scaled_data[i, 0])  # Predicting Close
    
    X, y = np.array(X), np.array(y)
    return X, y, scaler
