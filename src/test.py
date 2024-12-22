import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from yaml import full_load

from model import RatingPredictor


def test(model: torch.nn.Module, dataset: pd.DataFrame, device: str = "mps") -> np.ndarray:
  model.eval()
  pbar = tqdm(range(len(dataset)), ncols=94)
  mae, rmse = 0, 0
  predictions = []
  for i in pbar:
    batch = dataset.iloc[i:i + 1]
    x = batch["review_text"].tolist()
    y = torch.from_numpy(batch["rating"].to_numpy()).long().to(device)
    with torch.inference_mode():
      out = model(x)
    out = (out.squeeze() * 9) + 1
    predictions.append(torch.round(out))
    mae += torch.abs(out - ((y * 9) + 1)).mean()
    rmse += torch.sqrt(((out - ((y * 9) + 1))**2).mean())
  mae /= len(pbar)
  rmse /= len(pbar)
  return torch.cat(predictions).cpu().numpy()


def main(experiment_path: str):
  with open(os.path.join(experiment_path, "ExperimentSummary.yaml"), "r") as f:
    config = full_load(f)
  device = torch.device(config["training"]["device"])
  dataset = pd.read_csv(os.path.join(config["training"]["data_path"], "test_df.csv"))
  model = RatingPredictor(**config["model"]).to(device)
  predictions = test(model, dataset, device)
  pd.concat(
    [dataset["rating"], pd.DataFrame(predictions, columns=["predicted_rating"])],
    axis=1
  ).to_csv(os.path.join(experiment_path, "predictions.csv"))


if __name__ == "__main__":
  from fire import Fire
  Fire(main)
