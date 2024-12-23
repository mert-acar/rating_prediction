import os
import yaml
import joblib
from tabulate import tabulate
from datetime import datetime
from typing import Dict, Optional
from collections import defaultdict
from sklearn.pipeline import Pipeline


class ModelRegistry:
  def __init__(self, registry_path: str = "../models"):
    self.registry_path = registry_path
    os.makedirs(self.registry_path, exist_ok=True)

    self.index_path = os.path.join(registry_path, "model_index.yaml")
    if os.path.exists(self.index_path):
      with open(self.index_path, "r") as f:
        self.index = yaml.full_load(f)
    else:
      self.index = {'models': {}}
      self._save_index()

  def _save_index(self):
    with open(self.index_path, "w") as f:
      yaml.dump(self.index, f)

  def register_model(
    self,
    pipe: Pipeline,
    pipe_path: str,
    model_version: str,
    description: str,
    metrics: Optional[Dict[str, float]] = None
  ):
    joblib.dump(pipe, pipe_path)
    self.index['models'][model_version] = {
      'timestamp': datetime.now().isoformat(),
      'description': description,
      'metrics': metrics or {},
      'pipe_path': pipe_path
    }
    self._save_index()
    print(f"+ Model is registered and saved to {pipe_path}")

  def delete_model(self, version: str):
    model = self.index["models"].get(version, None)
    if model is None:
      raise ValueError(f"Model with version {version} does not exists!")
    os.remove(model["pipe_path"])
    del self.index["models"][version]

  def load_model(self, version: str) -> Pipeline:
    if version not in self.index['models']:
      raise ValueError(f"Model version {version} not found in registry")
    model_info = self.index['models'][version]
    pipe = joblib.load(model_info['pipe_path'])
    return pipe

  def get_latest_version(self) -> str:
    """Get the latest model version based on timestamp."""
    if not self.index['models']:
      raise ValueError("No models registered")
    return max(self.index['models'].keys(), key=lambda v: self.index['models'][v]['timestamp'])

  def list_versions(self, tobeprinted: bool = False) -> Optional[dict]:
    if len(self.index["models"]) == 0 and tobeprinted:
      print("No model in the registry yet! Train a model using the train script to register.")

    versiondict = defaultdict(list)
    for version, info in self.index["models"].items():
      versiondict["version"].append(version)
      for key, val in info.items():
        if key == "metrics":
          for m, s in val.items():
            versiondict[m].append(s)
        else:
          versiondict[key].append(val)

    if not tobeprinted:
      return versiondict

    print("\n" + tabulate(versiondict, headers=list(versiondict.keys())) + "\n")


if __name__ == "__main__":
  ModelRegistry().list_versions(True)
