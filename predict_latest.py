import joblib
import pandas as pd
import yfinance as yf

data = yf.Ticker("AAPL")
data = data.history(period="max")

del data["Dividends"]
del data["Stock Splits"]

data["Tomorrow"] = data["Close"].shift(-1)
data = data.loc["1996-01-01":]
data["Target"] = (data["Tomorrow"] > data["Close"]).astype(int)

predictors = ["Open", "High", "Low", "Close", "Volume"]

def compute_ratio(data, window):
    return data['Close'] / data['Close'].rolling(window=window).mean()

def compute_trend(data, window):
    return data["Target"].shift(1).rolling(window=window).sum()

def compute_rsi(data, window):
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def compute_sma(data, window):
    return data['Close'].rolling(window=window).mean()

def compute_ema(data, window):
    return data['Close'].ewm(span=window, adjust=False).mean()

def compute_macd(data, short_window, long_window, signal_window):
    short_ema = data['Close'].ewm(span=short_window, adjust=False).mean()
    long_ema = data['Close'].ewm(span=long_window, adjust=False).mean()
    macd = short_ema - long_ema
    signal = macd.ewm(span=signal_window, adjust=False).mean()
    return macd, signal
windows = [2, 5, 10, 50, 100, 300, 500, 1000]

for window in windows:
    ratio_column = f"Close_Ratio_{window}"
    trend_column = f"Trend_{window}"
    rsi_column = f"RSI_{window}"
    sma_column = f"SMA_{window}"
    ema_column = f"EMA_{window}"
    macd_column = f"MACD_{window}"
    signal_column = f"Signal_{window}"
    data.loc[:, ratio_column] = compute_ratio(data, window)
    data.loc[:, trend_column] = compute_trend(data, window)
    data.loc[:, rsi_column] = compute_rsi(data, window)
    data.loc[:, sma_column] = compute_sma(data, window)
    data.loc[:, ema_column] = compute_ema(data, window)
    data.loc[:, macd_column], data.loc[:, signal_column] = compute_macd(data, window, window * 2, int(window * 1.5))
    predictors += [ratio_column, trend_column, rsi_column, sma_column, ema_column, macd_column, signal_column]

data = data.dropna()

today = data.iloc[-1]
today = today[predictors].values.reshape(1, -1)

models = ["ada_model", "xgb_model", "rf_model", "voting_model"]
for model_name in models:
    model = joblib.load(f"{model_name}.joblib")
    prediction = model.predict(today)[0]
    print(f"{model_name}: The price will probably go {'up' if prediction else 'down'}")

