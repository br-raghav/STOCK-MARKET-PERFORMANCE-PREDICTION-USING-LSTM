# STOCK-MARKET-PERFORMANCE-PREDICTION-USING-LSTM

# Stock Market Performance Prediction Using LSTM

## Overview

This project explores **stock market time-series forecasting using a Long Short-Term Memory (LSTM) neural network**.

Historical stock market data is collected from the **Tiingo API**, processed using Python, and used to train a stacked LSTM model to learn patterns in historical closing prices.

The project uses **Apple Inc. (AAPL)** as the example security.

The complete workflow is:

```text
Tiingo API
    ↓
Historical Stock Data
    ↓
Data Cleaning
    ↓
Closing Price Selection
    ↓
Min-Max Scaling
    ↓
Train / Test Split
    ↓
Time-Series Sequence Creation
    ↓
Stacked LSTM Model
    ↓
Model Training
    ↓
Price Prediction
    ↓
RMSE Evaluation
    ↓
30-Day Forecast
```

---

## Project Objective

The objective is to investigate whether an LSTM-based deep-learning model can learn patterns from historical stock closing prices and use those patterns to generate short-term forecasts.

The project demonstrates the application of:

* Financial API integration
* Time-series data processing
* Feature scaling
* Sequential data preparation
* Deep-learning model development
* Model evaluation
* Recursive forecasting
* Financial data visualization

---

# Project Architecture

```text
                    ┌─────────────────┐
                    │   Tiingo API    │
                    └────────┬────────┘
                             │
                             ▼
                    Historical Prices
                             │
                             ▼
                    ┌─────────────────┐
                    │     Pandas      │
                    │ Data Processing │
                    └────────┬────────┘
                             │
                             ▼
                     Closing Prices
                             │
                             ▼
                    Min-Max Scaling
                             │
                             ▼
                   Time-Series Windows
                             │
                             ▼
                    ┌─────────────────┐
                    │   LSTM Network  │
                    │                 │
                    │ LSTM → LSTM →   │
                    │ LSTM → Dense    │
                    └────────┬────────┘
                             │
                             ▼
                       Predictions
                             │
                    ┌────────┴────────┐
                    ▼                 ▼
                 RMSE             30-Day
               Evaluation         Forecast
```

---

# Project Workflow

## 1. Retrieve Financial Data

Historical AAPL stock data is retrieved from the Tiingo API.

The dataset contains market information such as:

* Date
* Open price
* High price
* Low price
* Closing price
* Adjusted closing price
* Trading volume

The project primarily uses the **closing price** as the prediction variable.

---

## 2. Explore the Dataset

The retrieved dataset is loaded into a Pandas DataFrame.

Basic dataset inspection includes:

```python
df.head()
df.tail()
df.shape
df.info()
```

This helps understand:

* Number of observations
* Available columns
* Data types
* Missing values
* Dataset structure

---

# Closing Price Analysis

The project extracts the closing price:

```python
close_price = df["close"]
```

The historical closing prices are plotted to visualize the overall movement of the security over time.

This provides an initial understanding of the time-series data before model training.

---

# Data Preprocessing

## Min-Max Scaling

LSTM models generally perform better when numerical inputs are normalized.

The closing prices are therefore scaled to a range between **0 and 1** using `MinMaxScaler`.

```text
Original Price
      ↓
Min-Max Scaling
      ↓
0 ───────────── 1
```

The scaler is later used to convert model predictions back into their original price scale.

---

# Train / Test Split

The historical dataset is divided into:

```text
65% → Training Data
35% → Testing Data
```

The training dataset is used to learn historical patterns.

The testing dataset is kept separate for evaluating how the model performs on unseen observations.

---

# Time-Series Sequence Creation

LSTM networks require sequential input.

The project uses a **100-day lookback window**.

Conceptually:

```text
Previous 100 prices
        ↓
      LSTM
        ↓
Next predicted price
```

For example:

```text
Day 1 ─┐
Day 2  │
Day 3  │
 ...   ├──→ LSTM ──→ Day 101
Day 99 │
Day100 ┘
```

The sequence-generation function converts the historical price series into supervised-learning samples.

---

# LSTM Model

The project uses a stacked LSTM architecture built with **TensorFlow/Keras**.

The architecture is approximately:

```text
Input
  │
  ▼
LSTM — 50 units
  │
  ▼
LSTM — 50 units
  │
  ▼
LSTM — 50 units
  │
  ▼
Dense — 1 unit
  │
  ▼
Predicted Price
```

### Why LSTM?

Long Short-Term Memory networks are designed to work with sequential data and can retain information across multiple time steps.

This makes them suitable for experimentation with:

* Stock prices
* Financial time series
* Sensor data
* Demand forecasting
* Other sequential datasets

---

# Model Training

The model is trained using:

* Optimizer: **Adam**
* Loss function: **Mean Squared Error**
* Epochs: **100**
* Batch size: **64**

The model is trained using historical sequences and their corresponding next-day closing prices.

Validation data is used to monitor performance on the test dataset during training.

---

# Prediction

After training, the model generates predictions for:

### Training Data

Used to understand how well the model learned the historical training patterns.

### Testing Data

Used to evaluate performance on previously unseen observations.

The predicted values are transformed back from the normalized 0–1 scale into actual stock-price values.

---

# Model Evaluation

The project uses **Root Mean Squared Error (RMSE)** to evaluate prediction error.

```text
RMSE = √(Mean Squared Error)
```

Lower RMSE generally indicates that predictions are closer to the actual values.

The project evaluates:

```text
Training RMSE
Testing RMSE
```

The testing RMSE is particularly important because it measures performance on data that was not used to train the model.

---

# Actual vs Predicted Prices

The project visualizes:

```text
Actual Price
     vs
Training Prediction
     vs
Testing Prediction
```

This makes it easier to visually inspect whether the model is following major historical price movements.

---

# 30-Day Forecast

The project also performs a recursive forecast for the next **30 time steps**.

The process is:

```text
Last 100 Historical Prices
          ↓
      LSTM Model
          ↓
    Prediction #1
          ↓
Add prediction to sequence
          ↓
      LSTM Model
          ↓
    Prediction #2
          ↓
         ...
          ↓
    Prediction #30
```

This produces a 30-step forecast based on the model's previous predictions.

---

# Technology Stack

| Technology             | Purpose                   |
| ---------------------- | ------------------------- |
| Python                 | Core programming language |
| Pandas                 | Data manipulation         |
| NumPy                  | Numerical computation     |
| Matplotlib             | Data visualization        |
| Requests / Tiingo      | Financial data retrieval  |
| Scikit-learn           | Scaling and evaluation    |
| TensorFlow             | Deep-learning framework   |
| Keras                  | LSTM model development    |
| Google Colab / Jupyter | Development environment   |

---

# Project Structure

```text
stock-market-lstm/
│
├── README.md
│
└── Stock_Market_Performance_Prediction_LSTM.ipynb
```

The Jupyter notebook contains the complete implementation.

---

# Installation

Install the required Python packages:

```bash
pip install pandas numpy matplotlib scikit-learn tensorflow requests
```

If using Google Colab, the required packages can be installed directly inside the notebook.

---

# Tiingo API

The project uses Tiingo as the historical market-data provider.

The API key should **not be hard-coded** into a public GitHub repository.

Instead, provide it at runtime or through an environment variable.

Example:

```python
import getpass

api_key = getpass.getpass(
    "Enter Tiingo API Token: "
)
```

---

# Running the Project

### Step 1

Open the notebook using:

* Google Colab
* Jupyter Notebook
* JupyterLab

### Step 2

Enter your Tiingo API token when prompted.

### Step 3

Run the notebook from top to bottom.

### Step 4

The notebook will:

```text
Retrieve AAPL data
       ↓
Inspect dataset
       ↓
Plot historical prices
       ↓
Scale data
       ↓
Create sequences
       ↓
Train LSTM
       ↓
Generate predictions
       ↓
Calculate RMSE
       ↓
Plot predictions
       ↓
Generate 30-step forecast
```

---

# Example Output

The notebook produces outputs including:

### Dataset Preview

```text
Date          Open     High     Low      Close
------------------------------------------------
2025-01-02    ...      ...      ...      ...
2025-01-03    ...      ...      ...      ...
...
```

### Model Evaluation

```text
Training RMSE : <calculated value>
Testing RMSE  : <calculated value>
```

### Forecast

```text
Future Day 1  → predicted price
Future Day 2  → predicted price
Future Day 3  → predicted price
...
Future Day 30 → predicted price
```

The exact results depend on the historical data returned by Tiingo and the model training process.

---

# Key Machine-Learning Concepts Demonstrated

This project demonstrates practical experience with:

### Time-Series Modeling

Historical observations are converted into sequential training samples.

### Feature Scaling

Financial prices are normalized before being passed to the neural network.

### Sequence Learning

A 100-step historical window is used to predict the following value.

### Deep Learning

A stacked LSTM architecture is used to model temporal relationships.

### Model Evaluation

RMSE is used to quantify prediction error.

### Recursive Forecasting

Predicted values are repeatedly fed back into the model to generate future forecasts.

---

# Limitations

Stock-price forecasting is inherently difficult because financial markets are affected by many factors that are not represented by historical closing prices alone.

This project has several limitations:

* It uses primarily historical closing prices.
* It does not incorporate news or sentiment.
* It does not include macroeconomic variables.
* It does not model company fundamentals.
* Recursive forecasting can accumulate prediction errors.
* Historical patterns do not guarantee future performance.
* The model should not be interpreted as a reliable trading strategy.

Therefore, the project should be considered an **experimental machine-learning forecasting exercise**, rather than a financial prediction or investment system.

---

# Future Improvements

The project could be extended by adding:

### Multiple Features

Instead of using only closing prices:

```text
Open
High
Low
Close
Volume
```

could be used as model features.

### Technical Indicators

Potential additions include:

* Moving averages
* RSI
* MACD
* Bollinger Bands

### External Information

The model could incorporate:

* Financial news
* Market sentiment
* Earnings information
* Macroeconomic indicators

### Model Comparison

The LSTM could be compared against:

* ARIMA
* Random Forest
* XGBoost
* GRU
* Transformer-based time-series models

### Better Evaluation

A more robust evaluation could include:

* MAE
* RMSE
* MAPE
* Walk-forward validation
* Baseline comparison

---

# Portfolio Context

This project represents the **machine-learning and financial-data component** of my portfolio.

It demonstrates how I progressed from working with financial data and traditional data-processing techniques toward machine-learning and AI-based applications.

My broader portfolio also explores:

```text
Financial Data
      ↓
Data Quality / MDM
      ↓
Machine Learning
      ↓
Generative AI
      ↓
LLM Evaluation & Observability
```

This provides exposure to different layers of modern data and AI engineering.

---

# Author

**Raghav B R**

---

# Disclaimer

This project is intended for educational and portfolio demonstration purposes.

The predictions generated by the model should **not** be interpreted as financial advice, investment recommendations, or guaranteed future stock prices.
