import re
import os
import joblib
import numpy as np
from typing import Tuple
from sklearn.pipeline import Pipeline
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


def compute_sample_weights(ratings: np.ndarray) -> np.ndarray:
  """Compute sample weights to handle rating imbalance"""
  n_samples = len(ratings)
  unique_ratings = np.unique(ratings)
  class_weights = np.array([
    n_samples / (len(unique_ratings) * (ratings == rating).sum()) for rating in unique_ratings
  ])
  return class_weights[ratings.astype(int) - 1]


def main(data_dir: str):
  train_feat, train_ratings, test_feat, test_ratings = load_data(data_dir)
  # sample_weights = compute_sample_weights(train_ratings)

  random_state = 9001
  from sklearn.linear_model import Ridge
  model = Ridge(random_state=random_state)
  params = {"alpha": [0, 1e-4, 1e-3, 1e-2, 0.5, 1]}


  grid_search = GridSearchCV(
    model,
    params,
    cv=5,
    scoring='neg_mean_squared_error',
    n_jobs=-1,
  )

  print("Training model...")
  grid_search.fit(train_feat, train_ratings)
  print(f"Best parameters:")
  print(grid_search.best_params_)

  y_pred = grid_search.predict(test_feat)
  y_pred = np.clip(y_pred, 1, 10)

  print(f"\nResults:")
  mse = mean_squared_error(test_ratings, y_pred)
  print(f"+ MSE: {mse:.4f}")
  rmse = np.sqrt(mse)
  print(f"+ RMSE: {rmse:.4f}")
  mae = mean_absolute_error(test_ratings, y_pred)
  print(f"+ MAE: {mae:.4f}")
  r2 = r2_score(test_ratings, y_pred)
  print(f"+ R2 Score: {r2:.4f}")

  # Detailed error analysis by rating
  print("\nError analysis by rating:")
  for rating in sorted(np.unique(test_ratings)):
    mask = test_ratings == rating
    rating_mse = mean_squared_error(test_ratings[mask], y_pred[mask])
    rating_mae = mean_absolute_error(test_ratings[mask], y_pred[mask])
    print(f"Rating {rating}:")
    print(f"  - Count: {mask.sum()}")
    print(f"  - MSE: {rating_mse:.4f}")
    print(f"  - MAE: {rating_mae:.4f}")

  registry = ModelRegistry()
  if len(registry) > 0:
    print("\nCurrent Registry:")
    registry.print_versions()

  ans = input("\nSave model to the registry? [Y/n]: ")
  if ans.lower() != "y":
    return

  preprocessor = joblib.load(os.path.join(data_dir, "preprocessor.joblib"))
  pipe = Pipeline([
    ("preprocess", preprocessor),
    ("prediction", grid_search.best_estimator_),
  ])

  version = ""
  while len(version) == 0:
    version = input("Model version (v[<desired_version_str>]: ")
  version = re.sub(r"^(v\.?|V\.?)", "", version)

  description = input("Short model description: ")

  registry.register_model(
    pipe,
    f"model_v{version}",
    version,
    description,
    {
      "MSE": float(np.round(mse, 3)),
      "RMSE": float(np.round(rmse, 3)),
      "MAE": float(np.round(mae, 3)),
      "R2": float(np.round(r2, 3))
    },
  )
  print(f"+ Model is registered and saved to {pipe}")
  print("\nCurrent Registry:")
  registry.print_versions()


if __name__ == "__main__":
  from fire import Fire
  Fire(main)
