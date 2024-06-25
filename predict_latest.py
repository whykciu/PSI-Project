import joblib
import pandas as pd
import yfinance as yf
from data_loader import load

data, predictors = load()

today = data.iloc[-1]
today = today[predictors].values.reshape(1, -1)

models = ["ada_model", "xgb_model", "rf_model", "voting_model"]
for model_name in models:
    model = joblib.load(f"{model_name}.joblib")
    prediction = model.predict(today)[0]
    print(f"{model_name}: The price will probably go {'up' if prediction else 'down'}")

