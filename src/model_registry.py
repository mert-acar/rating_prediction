import os
import yaml
import joblib
from tabulate import tabulate
from datetime import datetime
from typing import Dict, Optional
from collections import defaultdict
from sklearn.pipeline import Pipeline


class ModelRegistry:
  def __init__(self):
    self.registry_path = os.path.abspath(
      os.path.join(os.path.dirname(os.path.abspath(__file__)), "../models/")
    )
    os.makedirs(self.registry_path, exist_ok=True)
    self.index_path = os.path.join(self.registry_path, "model_index.yaml")
    if os.path.exists(self.index_path):
      with open(self.index_path, "r") as f:
        self.index = yaml.full_load(f)
    else:
      self.index = {'models': {}}
      self._save_index()

  def _save_index(self):
    with open(self.index_path, "w") as f:
      yaml.dump(self.index, f)

  def __len__(self) -> int:
    return len(self.index["models"])

  def register_model(
    self,
    pipe: Pipeline,
    pipe_name: str,
    model_version: str,
    description: str,
    metrics: Optional[Dict[str, float]] = None
  ):
    pipe_path = os.path.join(self.registry_path, pipe_name)
    joblib.dump(pipe, pipe_path)

    self.index['models'][model_version] = {
      'timestamp': datetime.now().isoformat(),
      'description': description,
      'metrics': metrics or {},
      'pipe_name': pipe_name
    }
    self._save_index()

  def delete_model(self, version: str):
    model = self.index["models"].get(version, None)
    if model is None:
      raise ValueError(f"Model with version {version} does not exists!")
    os.remove(os.path.join(self.registry_path, os.path.basename(model["pipe_name"])))
    del self.index["models"][version]

  def load_model(self, version: str) -> Pipeline:
    if version not in self.index['models']:
      raise ValueError(f"Model version {version} not found in registry")
    model_info = self.index['models'][version]
    pipe_name = model_info['pipe_name']
    pipe_path = os.path.join(self.registry_path, os.path.basename(pipe_name))
    pipe = joblib.load(pipe_path)
    return pipe

  def get_latest_version(self) -> str:
    """Get the latest model version based on timestamp."""
    if not self.index['models']:
      raise ValueError("No models registered")
    return max(self.index['models'].keys(), key=lambda v: self.index['models'][v]['timestamp'])

  def print_versions(self):
    if len(self.index["models"]) == 0:
      print("No model in the registry yet!")
    else:
      versiondict = defaultdict(list)
      for version, info in self.index["models"].items():
        versiondict["version"].append(version)
        for key, val in info.items():
          if key == "metrics":
            for m, s in val.items():
              versiondict[m].append(s)
          else:
            versiondict[key].append(val)
      print("\n" + tabulate(versiondict, headers=list(versiondict.keys())) + "\n")

  def get_versions(self) -> list[dict]:
    return [{"version": version, **info} for (version, info) in self.index["models"].items()]


def main(verb: str = "show", version: Optional[str] = None):
  if verb == "show":
    ModelRegistry().print_versions()
  elif verb == "delete":
    if version is None:
      print("Please specify a version to delete!")
      return
    ModelRegistry().delete_model(version)
  else:
    print(
      "usage: python3 model_registry.py <verb: OneOf[show (default), delete]> <Optional[version]>"
    )


if __name__ == "__main__":
  from fire import Fire
  Fire(main)
