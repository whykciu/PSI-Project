import yfinance as yf
import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split, TimeSeriesSplit
from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
import joblib
from data_loader import load
data, predictors = load()
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

joblib.dump(ada_model, 'ada_model.joblib')

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

joblib.dump(rf_model, 'rf_model.joblib')

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

joblib.dump(xgb_model, 'xgb_model.joblib')

voting_model = VotingClassifier(estimators=[('xgb', xgb_model), ('rf', rf_model), ('ada', ada_model)], voting='soft')
voting_model.fit(X_train, y_train)
test_preds_voting = voting_model.predict(X_test)
print('Test Precision score: ', precision_score(y_test, test_preds_voting))
print('Test Accuracy score: ', accuracy_score(y_test, test_preds_voting))
print('Test Recall score: ', recall_score(y_test, test_preds_voting))
print('Test F1 score: ', f1_score(y_test, test_preds_voting))

joblib.dump(voting_model, 'voting_model.joblib')