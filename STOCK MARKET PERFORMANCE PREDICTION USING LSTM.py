# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================

import math
import numpy as np
import pandas as pd
import pandas_datareader.data as pdr
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense


# ============================================================
# 2. CHECK LIBRARY VERSIONS
# ============================================================

print("pandas:", pd.__version__)

try:
    import pandas_datareader
    print("pandas_datareader:", pandas_datareader.__version__)
except AttributeError:
    print("pandas_datareader is installed")


# ============================================================
# 3. GET TIINGO API KEY
# ============================================================

# Enter your Tiingo API key here
api_key = input("Enter your Tiingo API key: ").strip()


# ============================================================
# 4. FETCH APPLE STOCK DATA
# ============================================================

print("\nFetching Apple stock data from Tiingo...")

df = pdr.get_data_tiingo(
    "AAPL",
    api_key=api_key
)

print("Data fetched successfully.")


# ============================================================
# 5. DISPLAY STOCK DATA
# ============================================================

print("\nFirst 5 rows:")
print(df.head())

print("\nLast 5 rows:")
print(df.tail())

print("\nData shape:")
print(df.shape)

print("\nData information:")
print(df.info())


# ============================================================
# 6. SELECT CLOSING PRICE
# ============================================================

# Extract only the closing price
close_price = df["close"].astype(float)

print("\nClosing prices:")
print(close_price.head())


# ============================================================
# 7. PLOT CLOSING PRICE
# ============================================================

plt.figure(figsize=(10, 5))

plt.plot(close_price.values)

plt.title("Apple Stock Closing Price")
plt.xlabel("Time")
plt.ylabel("Price")

plt.show()


# ============================================================
# 8. SCALE THE DATA
# ============================================================

# LSTM models usually perform better when the input
# values are scaled between 0 and 1.

scaler = MinMaxScaler(feature_range=(0, 1))

close_scaled = scaler.fit_transform(
    np.array(close_price).reshape(-1, 1)
)


# ============================================================
# 9. SPLIT DATA INTO TRAINING AND TESTING
# ============================================================

# 65% of the data is used for training
train_size = int(len(close_scaled) * 0.65)

# Remaining 35% is used for testing
test_size = len(close_scaled) - train_size

train_data = close_scaled[:train_size]
test_data = close_scaled[train_size:]

print("\nTraining data size:", train_size)
print("Testing data size:", test_size)


# ============================================================
# 10. CREATE DATASET FUNCTION
# ============================================================

# Converts the time series into input/output sequences.
#
# Example with step = 3:
#
# X = [price1, price2, price3]
# y = price4

def create_dataset(data, step):

    X = []
    y = []

    for i in range(len(data) - step - 1):

        # Previous 'step' prices
        X.append(data[i:(i + step), 0])

        # Next price
        y.append(data[i + step, 0])

    return np.array(X), np.array(y)


# ============================================================
# 11. PREPARE TRAINING AND TESTING DATA
# ============================================================

# Number of previous days used to predict the next day
time_steps = 100

X_train, y_train = create_dataset(
    train_data,
    time_steps
)

X_test, y_test = create_dataset(
    test_data,
    time_steps
)

print("\nX_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)

print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)


# ============================================================
# 12. RESHAPE DATA FOR LSTM
# ============================================================

# LSTM expects data in the format:
#
# (samples, time_steps, features)

X_train = X_train.reshape(
    X_train.shape[0],
    X_train.shape[1],
    1
)

X_test = X_test.reshape(
    X_test.shape[0],
    X_test.shape[1],
    1
)


# ============================================================
# 13. BUILD LSTM MODEL
# ============================================================

model = Sequential()

# First LSTM layer
model.add(
    LSTM(
        50,
        return_sequences=True,
        input_shape=(time_steps, 1)
    )
)

# Second LSTM layer
model.add(
    LSTM(
        50,
        return_sequences=True
    )
)

# Third LSTM layer
model.add(
    LSTM(50)
)

# Output layer
model.add(Dense(1))


# ============================================================
# 14. COMPILE MODEL
# ============================================================

model.compile(
    loss="mean_squared_error",
    optimizer="adam"
)


# Display model architecture
model.summary()


# ============================================================
# 15. TRAIN THE MODEL
# ============================================================

print("\nTraining LSTM model...")

model.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    epochs=100,
    batch_size=64,
    verbose=1
)


# ============================================================
# 16. MAKE PREDICTIONS
# ============================================================

print("\nMaking predictions...")

train_pred = model.predict(
    X_train,
    verbose=0
)

test_pred = model.predict(
    X_test,
    verbose=0
)


# ============================================================
# 17. INVERSE SCALE PREDICTIONS
# ============================================================

# Convert predictions back to actual stock prices

train_pred_actual = scaler.inverse_transform(
    train_pred
)

test_pred_actual = scaler.inverse_transform(
    test_pred
)

# Convert actual y values back to original price scale

y_train_actual = scaler.inverse_transform(
    y_train.reshape(-1, 1)
)

y_test_actual = scaler.inverse_transform(
    y_test.reshape(-1, 1)
)


# ============================================================
# 18. CALCULATE RMSE
# ============================================================

train_rmse = math.sqrt(
    mean_squared_error(
        y_train_actual,
        train_pred_actual
    )
)

test_rmse = math.sqrt(
    mean_squared_error(
        y_test_actual,
        test_pred_actual
    )
)

print("\n==============================")
print("MODEL EVALUATION")
print("==============================")

print("Training RMSE:", train_rmse)
print("Testing RMSE:", test_rmse)


# ============================================================
# 19. PREPARE DATA FOR PREDICTION PLOT
# ============================================================

# Empty array for training predictions
train_plot = np.empty_like(close_scaled)

train_plot[:] = np.nan

train_plot[
    time_steps:len(train_pred) + time_steps
] = train_pred


# Empty array for testing predictions
test_plot = np.empty_like(close_scaled)

test_plot[:] = np.nan

test_start = (
    len(train_pred)
    + (time_steps * 2)
    + 1
)

test_end = test_start + len(test_pred)

if test_end <= len(test_plot):

    test_plot[test_start:test_end] = test_pred


# ============================================================
# 20. PLOT ACTUAL VS PREDICTED PRICES
# ============================================================

actual_prices = scaler.inverse_transform(
    close_scaled
)

train_predictions = scaler.inverse_transform(
    train_plot
)

test_predictions = scaler.inverse_transform(
    test_plot
)


plt.figure(figsize=(12, 6))

plt.plot(
    actual_prices,
    label="Actual Price"
)

plt.plot(
    train_predictions,
    label="Train Prediction"
)

plt.plot(
    test_predictions,
    label="Test Prediction"
)

plt.title("Apple Stock Price Prediction")

plt.xlabel("Time")
plt.ylabel("Price")

plt.legend()

plt.show()


# ============================================================
# 21. PREDICT NEXT 30 DAYS
# ============================================================

future_days = 30

future_output = []

# Take the last 100 scaled prices
temp_input = close_scaled[-time_steps:].flatten().tolist()


# Predict one day at a time
for i in range(future_days):

    # Use the latest 100 values
    x_input = np.array(
        temp_input[-time_steps:],
        dtype=np.float32
    ).reshape(1, time_steps, 1)

    # Predict next value
    yhat = model.predict(
        x_input,
        verbose=0
    )

    predicted_value = float(yhat[0][0])

    # Add prediction to input sequence
    temp_input.append(predicted_value)

    # Store prediction
    future_output.append(predicted_value)


# ============================================================
# 22. CONVERT FUTURE PREDICTIONS TO ORIGINAL PRICE SCALE
# ============================================================

future_output = np.array(
    future_output
).reshape(-1, 1)

future_output = scaler.inverse_transform(
    future_output
)


# ============================================================
# 23. PLOT 30-DAY FORECAST
# ============================================================

# Show the last 200 historical prices
historical_prices = scaler.inverse_transform(
    close_scaled[-200:]
)

plt.figure(figsize=(12, 6))

plt.plot(
    range(200),
    historical_prices,
    label="Historical"
)

plt.plot(
    range(200, 200 + future_days),
    future_output,
    label="Future Prediction"
)

plt.title("Apple Stock Price - Next 30 Days Forecast")

plt.xlabel("Days")

plt.ylabel("Price")

plt.legend()

plt.show()


# ============================================================
# 24. DISPLAY FUTURE PREDICTIONS
# ============================================================

print("\n==============================")
print("NEXT 30 DAYS PREDICTION")
print("==============================")

for day, price in enumerate(
    future_output.flatten(),
    start=1
):

    print(
        f"Day {day}: ${price:.2f}"
    )