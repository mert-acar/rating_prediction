import re
import os
import joblib
import numpy as np
from typing import Tuple
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from model_registry import ModelRegistry
from preprocess import *


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

  model = Ridge(random_state=9001)
  params = {"alpha": [0, 0.2, 0.5, 0.8, 1.0, 1.5, 2]}
  grid_search = GridSearchCV(model, params, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
  grid_search.fit(train_feat, train_ratings)
  print(f"Best parameters:")
  print(grid_search.best_params_)

  y_pred = grid_search.predict(test_feat)

  print(f"Results:")

  mse = mean_squared_error(test_ratings, y_pred)
  print(f"+ MSE: {mse:.4f}")
  rmse = float(np.sqrt(mse))
  print(f"+ RMSE: {rmse:.4f}")

  mae = mean_absolute_error(test_ratings, y_pred)
  print(f"+ MAE: {mae:.4f}")

  r2 = r2_score(test_ratings, y_pred)
  print(f"+ R2 Score: {r2:.4f}")

  ans = input("Save model to the registry? [Y/n]: ")
  if ans.lower() == "y":
    registry = ModelRegistry()
    preprocessor = joblib.load(os.path.join(data_dir, "preprocessor.joblib"))
    pipe = Pipeline([
      ("preprocess", preprocessor),
      ("prediction", model),
    ])

    version = ""
    while len(version) == 0:
      version = input("Model version (v[<desired_version_str>]: ")
    version = re.sub(r"^(v\.?|V\.?)", "", version)

    description = input("Short model description: ")

    registry.register_model(
      pipe,
      f"../models/model_v{version}",
      version,
      description,
      {
        "MSE": np.round(mse, 3),
        "RMSE": np.round(rmse, 3),
        "MAE": np.round(mae, 3),
        "R2": np.round(r2, 3)
      },
    )
