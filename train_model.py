import utils
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# 1. Fetch & Prepare Data
print("Fetching stock data...")
data = utils.fetch_stock_data("AAPL")  # Apple stock
data = utils.add_technical_indicators(data)
features = ['Close', 'Volume', 'rsi', 'macd', 'ema_20']

X, y, scaler = utils.prepare_lstm_data(data, features)

# 2. Split Data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)
print(f"Training samples: {X_train.shape}, Testing samples: {X_test.shape}")

# 3. Build Model
model = Sequential([
    LSTM(128, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(0.2),
    LSTM(64, return_sequences=False),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')

# 4. Train Model
print("Training model...")
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.1,
    verbose=1
)

# ✅ Step 8: Evaluation
print("Evaluating model...")
pred = model.predict(X_test)

# Plot Actual vs Predicted
plt.figure(figsize=(12, 6))
plt.plot(y_test, color='blue', label='Actual Price')
plt.plot(pred, color='red', label='Predicted Price')
plt.title('Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Normalized Price')
plt.legend()
plt.show()

input("Press Enter to exit...")

# Save the model
# In train_model.py
model.export("models/stock_lstm")  # Exports as TensorFlow SavedModel format

print("Model saved to models/stock_lstm.h5")
