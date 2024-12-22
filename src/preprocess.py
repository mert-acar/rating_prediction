import os
import re
import torch
import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from typing import Tuple, Union, Optional, List, Dict


@dataclass(frozen=True)
class DatasetConfig:
  """Configuration for dataset preprocessing."""
  test_size: float
  random_state: int
  discount_bins: List[float]
  top_categories: List[str]
  price_stats: Dict[str, float]


class PreprocessPipeline:
  """Handles feature transformation for both training and inference."""
  def __init__(self, config: Optional[DatasetConfig] = None, device: Optional[str] = None):
    self.config = config or DatasetConfig()
    self.device = device or torch.device("cuda" if torch.cuda.is_available() else "mps")

  def _transform_price(self, price: float) -> float:
    """Transform price using log transformation and standardization."""
    log_price = (np.log1p(price) - self.config.price_stats["mean"]) / self.config.price_stats["std"]
    return log_price.astype(np.float32)

  def _transform_category(self, category: str) -> np.ndarray:
    """Transform product category into one-hot encoding."""
    category = category.split("|")[0].lower().strip()
    category_vec = np.zeros(
      len(self.config.top_categories) + 1,  # +1 for OTHER
      dtype=np.float32
    )
    if category in self.config.top_categories:
      category_vec[self.config.top_categories.index(category)] = 1
    else:
      category_vec[-1] = 1
    return category_vec

  def _transform_review_text(
    self,
    review_text: Union[str, List[str]],
  ):
    """Clean up product review text from emojies and HTML tags"""
    input_is_string = isinstance(review_text, str)
    texts_to_clean = [review_text] if input_is_string else review_text

    emoji_pattern = re.compile(
      "[\U0001F600-\U0001F64F"  # emoticons
      "\U0001F300-\U0001F5FF"  # symbols & pictographs
      "\U0001F680-\U0001F6FF"  # transport & map symbols
      "\U0001F1E0-\U0001F1FF"  # flags (iOS)
      "\U00002700-\U000027BF"  # Dingbats
      "\U000024C2-\U0001F251"  # Enclosed characters
      "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
      "\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
      "\U00002600-\U000026FF"  # Miscellaneous Symbols
      "\U0001F018-\U0001F270"  # Various Asian Characters
      "]+",
      flags=re.UNICODE
    )

    # Remove all of the HTML tags from the strings
    cleaned_texts = [re.sub(r"<[^>]+>", "", emoji_pattern.sub(r'', text)) for text in texts_to_clean]
    return cleaned_texts[0] if input_is_string else cleaned_texts

  def _transform_discount(self, discount: float) -> np.ndarray:
    """Transform discount rate into binned one-hot encoding."""
    discount_vec = np.zeros(len(self.config.discount_bins), dtype=np.float32)
    for i in range(len(self.config.discount_bins) - 1):
      if self.config.discount_bins[i] <= discount < self.config.discount_bins[i + 1]:
        discount_vec[i] = 1
        break
    else:
      discount_vec[-1] = 1
    return discount_vec

  def transform(
    self,
    row: Union[pd.Series, dict],
  ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Transform a single row of data."""

    # Feature transformations
    log_price = self._transform_price(row["price"])
    category_vec = self._transform_category(row["product_category"])
    discount_vec = self._transform_discount(float(row["discount_rate"]))
    review_text = self._transform_review_text(str(row["review_text"]))
    rating = float(row["rating"])
    features = torch.cat([
      torch.from_numpy(discount_vec).to(self.device),
      torch.from_numpy(category_vec).to(self.device),
      torch.Tensor([log_price]).to(self.device), review_text
    ], dim=0)
    rating = torch.Tensor([rating])
    return features, rating

  def process_dataset(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # 1. Price preprocessing
    df["log_price"] = self._transform_price(df["price"])

    # 2. Discount preprocessing
    df["discount_bracket"] = pd.cut(
      df["discount_rate"],
      bins=[-float("inf")] + self.config.discount_bins,
      labels=[f"{i}" for i in range(len(self.config.discount_bins))]
    )
    discount_one_hot = pd.get_dummies(df["discount_bracket"], prefix="discount", dtype=np.float32)

    # 3. Category preprocessing
    df["main_category"] = df["product_category"].str.split("|").str[0].str.strip()
    df["category_grouped"] = df["main_category"].apply(
      lambda x: x if x in self.config.top_categories else "Other"
    )
    category_one_hot = pd.get_dummies(df["category_grouped"], prefix="category", dtype=np.float32)
    df["review_text"] = self._transform_review_text(df["review_text"].tolist())
    features = pd.concat([
      discount_one_hot, category_one_hot, df[["log_price"]], df[["review_text"]]
    ], axis=1)

    y = df["rating"]
    X_train, X_test, y_train, y_test = train_test_split(
      features,
      y,
      test_size=self.config.test_size,
      stratify=y,
      random_state=self.config.random_state
    )
    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)

    print("  - Training Data Statistics:")
    print_feature_stats(train_df, "binary")
    print_feature_stats(train_df, "numeric")

    print("  - Test Data Statistics:")
    print_feature_stats(test_df, "binary")
    print_feature_stats(test_df, "numeric")
    return train_df, test_df


def print_feature_stats(X, feature_type):
  if feature_type == "binary":
    cols = [col for col in X.columns if col.startswith(("discount_", "category_"))]
  elif feature_type == "numeric":
    cols = ["log_price"]
  else:
    raise ValueError

  subset = X[cols]
  print(f"\t- {feature_type.capitalize()} Features Statistics:")
  print(f"\t\tMean: {subset.values.mean():.4f}")
  print(f"\t\tStd: {subset.values.std():.4f}")
  print(f"\t\tMin: {subset.values.min():.4f}")
  print(f"\t\tMax: {subset.values.max():.4f}")


def prepare_train_test_split(
  csv_path: str, output_dir: str = "../data/", config_path: str = "./config.yaml"
):
  """Prepare and save train/test splits."""

  from yaml import full_load
  with open(config_path, "r") as f:
    config = DatasetConfig(**full_load(f)["dataset"])

  pipeline = PreprocessPipeline(config)
  df = pd.read_csv(csv_path)
  train_df, test_df = pipeline.process_dataset(df)

  os.makedirs(output_dir, exist_ok=True)
  train_df.to_csv(os.path.join(output_dir, "train_df.csv"), index=False)
  test_df.to_csv(os.path.join(output_dir, "test_df.csv"), index=False)


if __name__ == "__main__":
  from fire import Fire
  Fire(prepare_train_test_split)

  # from transformers import AutoTokenizer
  # tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
  # model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
  # max_seq_length = int(model.get_max_seq_length())
  # print(f"Max sequence length for model: {max_seq_length}")
  # reviews = pd.read_csv("../data/product_user_reviews.csv")["review_text"].tolist()
  # lengths = [len(tokenizer.encode(re.sub(r"<[^>]+>", "", review))) for review in reviews]
  # # 1858
  # print(f"Max length: {max(lengths)}")
  # # 54.3
  # print(f"Mean length: {sum(lengths)/len(lengths)}")
  # # 1.45%
  # print(f"% exceeding {max_seq_length}: {sum(l > max_seq_length for l in lengths)/len(lengths)*100:.2f}%")
