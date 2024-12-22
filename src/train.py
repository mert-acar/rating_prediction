import os
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from time import time
from yaml import full_load
import matplotlib.pyplot as plt
from shutil import copyfile, rmtree

from model import RatingPredictor

if __name__ == "__main__":
  with open("./config.yaml", "r") as f:
    config = full_load(f)
  train_config = config["training"]

  # Create the checkpoint output path
  if os.path.exists(train_config["output_path"]):
    c = input(
      f"Output path {train_config['output_path']} is not empty! Do you want to delete the folder [y / n]: "
    )
    if "y" == c.lower():
      rmtree(train_config["output_path"], ignore_errors=True)
    else:
      print("Exit!")
      raise SystemExit

  os.makedirs(train_config["output_path"])
  copyfile("./config.yaml", os.path.join(train_config["output_path"], "ExperimentSummary.yaml"))

  device = torch.device(train_config["device"])
  print(f"[INFO] Running on {device}")

  datasets = {
    "train": pd.read_csv(os.path.join(train_config["data_path"], "train_df.csv")),
    "test": pd.read_csv(os.path.join(train_config["data_path"], "test_df.csv")),
  }

  # Store original ratings (1-10) for classification
  for phase in datasets:
    datasets[phase]['original_rating'] = datasets[phase]['rating']
    datasets[phase]['rating'] = (datasets[phase]['rating'] - 1) / 9.0

  model = RatingPredictor(**config["model"]).to(device)
  model.freeze_embedding_model()

  # Loss functions
  regression_criterion = torch.nn.HuberLoss(delta=1.0)
  classification_criterion = torch.nn.CrossEntropyLoss()
  
  optimizer = torch.optim.AdamW(model.parameters(), **train_config["optimizer_args"])
  scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, **train_config["scheduler_args"]
  )

  tick = time()
  best_epoch = -1
  best_error = 999999
  phases = ["train", "test"]
  metrics = ["Loss", "RMSE", "MAE", "Accuracy"]
  metrics = {metric: {phase: [] for phase in phases} for metric in metrics}

  for epoch in range(1, train_config["num_epochs"] + 1):
    print("-" * 20)
    print(f"Epoch {epoch} / {train_config['num_epochs']}")
    for phase in ["train", "test"]:
      if phase == "train":
        model.train()
      else:
        model.eval()
      running_error = 0
      running_mae = 0
      running_rmse = 0
      running_acc = 0
      dataset = datasets[phase]

      if phase == "train":
        dataset = dataset.sample(frac=1).reset_index(drop=True)

      pbar = tqdm(range(0, len(dataset), train_config["batch_size"]), ncols=94)
      with torch.set_grad_enabled(phase == "train"):
        for i in pbar:
          batch = dataset.iloc[i:i + train_config["batch_size"]]
          x = batch["review_text"].tolist()
          y_reg = torch.from_numpy(batch["rating"].to_numpy()).float().to(device)
          y_cls = torch.from_numpy(batch["original_rating"].to_numpy()).long().to(device) - 1  # 0-9 for CrossEntropyLoss

          optimizer.zero_grad()
          reg_out, cls_out = model(x)
          reg_out = reg_out.squeeze()

          # Calculate losses
          reg_loss = regression_criterion(reg_out, y_reg)
          cls_loss = classification_criterion(cls_out, y_cls)
          
          # Combined loss (equal weighting)
          loss = reg_loss + cls_loss

          if phase == "train":
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

          # Calculate metrics
          mae = torch.abs((reg_out * 9) + 1 - ((y_reg * 9) + 1)).mean()
          rmse = torch.sqrt((((reg_out * 9) + 1 - ((y_reg * 9) + 1)) ** 2).mean())
          acc = (cls_out.argmax(dim=1) == y_cls).float().mean()
          
          running_error += loss.item()
          running_mae += mae.item()
          running_rmse += rmse.item()
          running_acc += acc.item()
          
          pbar.set_description(f"L:{loss.item():.3f}|MAE:{mae:.2f}|ACC:{acc:.2f}")

      # Average metrics
      num_batches = len(pbar)
      running_error /= num_batches
      running_mae /= num_batches
      running_rmse /= num_batches
      running_acc /= num_batches
      
      print(f"Loss: {running_error:.5f} | MAE: {running_mae:.3f} | RMSE: {running_rmse:.3f} | ACC: {running_acc:.3f}")
      
      metrics["Loss"][phase].append(running_error)
      metrics["MAE"][phase].append(running_mae)
      metrics["RMSE"][phase].append(running_rmse)
      metrics["Accuracy"][phase].append(running_acc)
      if phase == "test":
        scheduler.step(running_error)
        if running_error < best_error:
          best_error = running_error
          best_epoch = epoch
          ckpt_path = os.path.join(train_config["output_path"], "checkpoint.pt")
          print(f"+ Saving the model to {ckpt_path}...")
          torch.save(model.state_dict(), ckpt_path)

    # If no validation improvement has been recorded for "early_stop" number of epochs
    # stop the training.
    if epoch - best_epoch >= train_config["early_stop_patience"]:
      print(f"No improvements in {train_config['early_stop_patience']} epochs, stop!")
      break

  total_time = time() - tick
  m, s = divmod(total_time, 60)
  h, m = divmod(m, 60)
  print(f"Training took {int(h):d} hours {int(m):d} minutes {s:.2f} seconds.")

  fig, axs = plt.subplots(1, len(metrics), tight_layout=True, figsize=(15, 5))
  epochs = list(range(1, epoch + 1))
  for i, (metric, arr) in enumerate(metrics.items()):
    for phase, val in arr.items():
      axs[i].plot(epochs, val, label=phase)
    axs[i].set_xlabel("Epochs")
    axs[i].set_ylabel(metric)
    axs[i].legend()
    axs[i].grid(True)
  fig.suptitle("Model Performance Across Epochs")
  plt.savefig(
    os.path.join(train_config["output_path"], "performance_curves.png"), bbox_inches="tight"
  )
