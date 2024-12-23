import os
import joblib
import xgboost
import numpy as np
from typing import Tuple
from sklearn.linear_model import HuberRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def evaluate_model(y_true, y_pred, model_name):
  mse = mean_squared_error(y_true, y_pred)
  mae = mean_absolute_error(y_true, y_pred)
  r2 = r2_score(y_true, y_pred)

  print(f"\n{model_name} Results:")
  print(f"MSE: {mse:.4f}")
  print(f"RMSE: {np.sqrt(mse):.4f}")
  print(f"MAE: {mae:.4f}")
  print(f"R2 Score: {r2:.4f}")


def load_data(data_dir: str) -> Tuple[np.ndarray, ...]:
  print("Loading data...")
  train_feat = np.load(os.path.join(data_dir, "train_features.npy"))
  train_ratings = np.load(os.path.join(data_dir, "train_ratings.npy")).flatten()
  test_feat = np.load(os.path.join(data_dir, "test_features.npy"))
  test_ratings = np.load(os.path.join(data_dir, "test_ratings.npy")).flatten()
  return train_feat, train_ratings, test_feat, test_ratings


if __name__ == "__main__":
  data_dir = "../data/allMiniLM/"

  train_feat, train_ratings, test_feat, test_ratings = load_data(data_dir)

  n_samples = len(train_ratings)
  ratings = np.unique(train_ratings)
  class_weights = np.array([
    n_samples / (len(ratings) * (train_ratings == rating).sum()) for rating in ratings
  ])
  sample_weights = class_weights[train_ratings - 1]

  rf_params = {
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 10, 15, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
  }

  gb_params = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 5, 7],
    'subsample': [0.7, 0.8, 0.9]
  }

  xgb_params = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 5, 7],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.7, 0.8, 0.9]
  }

  huber_params = {
    'epsilon': [1.1, 1.35, 1.5],
    'alpha': [0.0001, 0.001, 0.01],
    'max_iter': [100, 200, 300]
  }

  models = {
    'Huber': (HuberRegressor(), huber_params),
    'XGBoost': (xgboost.XGBRegressor(random_state=42), xgb_params),
    'Gradient Boosting': (GradientBoostingRegressor(random_state=42), gb_params),
    'Random Forest': (RandomForestRegressor(random_state=42), rf_params),
  }

  best_score = float('inf')
  best_model_name = None
  best_model = None

  for name, (model, params) in models.items():
    print(f"\nTuning {name}...")
    grid_search = GridSearchCV(
      model, params, cv=5, scoring='neg_mean_squared_error', n_jobs=-1, verbose=2
    )

    if name in ['Random Forest', 'Gradient Boosting', 'XGBoost']:
      grid_search.fit(train_feat, train_ratings, sample_weight=sample_weights)
    else:
      grid_search.fit(train_feat, train_ratings)

    print(f"\nBest parameters for {name}:")
    print(grid_search.best_params_)

    y_pred = grid_search.predict(test_feat)
    evaluate_model(test_ratings, y_pred, name)

    current_score = mean_squared_error(test_ratings, y_pred)
    if current_score < best_score:
      best_score = current_score
      best_model_name = name
      best_model = grid_search.best_estimator_

  print(f"\nBest performing model: {best_model_name}")
  print(f"Best RMSE: {np.sqrt(best_score):.4f}")

  os.makedirs(data_dir, exist_ok=True)
  joblib.dump(best_model, os.path.join(data_dir, f"best_model_{best_model_name}.joblib"))
  print(f"\nBest model saved as: best_model_{best_model_name}.joblib")
