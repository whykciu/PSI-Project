import yfinance as yf
import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split, TimeSeriesSplit
from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
data = yf.Ticker("AAPL")
data = data.history(period="max")
data
data.plot(y="Close")
del data["Dividends"]
del data["Stock Splits"]
data["Tomorrow"] = data["Close"].shift(-1)
data = data.loc["1996-01-01":]
data.plot(y="Close")
data["Target"] = (data["Tomorrow"] > data["Close"]).astype(int)
data
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
data
train_data, test_data = train_test_split(data, test_size=0.2, shuffle=False)

X_train, y_train = train_data[predictors], train_data["Target"]
X_test, y_test = test_data[predictors], test_data["Target"]
tscv = TimeSeriesSplit(n_splits=5)
ada_model = AdaBoostClassifier(random_state=1, algorithm='SAMME')
param_grid_ada = {
    'n_estimators': [100, 300, 500, 1000, 2000],
    'learning_rate': [0.01, 0.1, 0.2, 0.5]
}


grid_search_ada = GridSearchCV(estimator=ada_model, param_grid=param_grid_ada, cv=tscv, n_jobs=-1, verbose=2, scoring='precision')
grid_search_ada.fit(X_train, y_train)

best_params_ada = grid_search_ada.best_params_
ada_model.set_params(**best_params_ada)
print(best_params_ada)

ada_model.fit(X_train, y_train)

ada_test_preds  = ada_model.predict(X_test)
print('Test Precision score: ', precision_score(y_test, ada_test_preds))
print('Test Accuracy score: ', accuracy_score(y_test, ada_test_preds))
print('Test Recall score: ', recall_score(y_test, ada_test_preds))
print('Test F1 score: ', f1_score(y_test, ada_test_preds))

rf_model = RandomForestClassifier(random_state=1)
param_grid_rf = {
    'n_estimators': [100, 300, 500, 1000, 2000],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 10, 20, 50, 100]
}

grid_search_rf = GridSearchCV(estimator=rf_model, param_grid=param_grid_rf, cv=tscv, n_jobs=-1, verbose=2, scoring='precision')
grid_search_rf.fit(X_train, y_train)

best_params_rf = grid_search_rf.best_params_
print(best_params_rf)

rf_model.set_params(**best_params_rf)
rf_model.fit(X_train, y_train)

rf_test_preds  = rf_model.predict(X_test)
print('Test Precision score: ', precision_score(y_test, rf_test_preds))
print('Test Accuracy score: ', accuracy_score(y_test, rf_test_preds))
print('Test Recall score: ', recall_score(y_test, rf_test_preds))
print('Test F1 score: ', f1_score(y_test, rf_test_preds))
xgb_model = XGBClassifier(random_state=1)
param_grid_xgb = {
    'n_estimators': [100, 300, 500, 1000, 2000],
    'learning_rate': [0.01, 0.1, 0.2, 0.5],
    'max_depth': [3, 5, 7, 10]
}

grid_search_xgb = GridSearchCV(estimator=xgb_model, param_grid=param_grid_xgb, cv=tscv, n_jobs=-1, verbose=2, scoring='precision')
grid_search_xgb.fit(X_train, y_train)

best_params_xgb = grid_search_xgb.best_params_
print(best_params_xgb)

xgb_model.set_params(**best_params_xgb)
xgb_model.fit(X_train, y_train)

test_preds_xgb = xgb_model.predict(X_test)
print('Test Precision score: ', precision_score(y_test, test_preds_xgb))
print('Test Accuracy score: ', accuracy_score(y_test, test_preds_xgb))
print('Test Recall score: ', recall_score(y_test, test_preds_xgb))
print('Test F1 score: ', f1_score(y_test, test_preds_xgb))
voting_model = VotingClassifier(estimators=[('xgb', xgb_model), ('rf', rf_model), ('ada', ada_model)], voting='soft')
voting_model.fit(X_train, y_train)
test_preds_voting = voting_model.predict(X_test)
print('Test Precision score: ', precision_score(y_test, test_preds_voting))
print('Test Accuracy score: ', accuracy_score(y_test, test_preds_voting))
print('Test Recall score: ', recall_score(y_test, test_preds_voting))
print('Test F1 score: ', f1_score(y_test, test_preds_voting))