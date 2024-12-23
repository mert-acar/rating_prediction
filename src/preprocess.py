import re
import numpy as np
import pandas as pd
from typing import Tuple
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sentence_transformers import SentenceTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder
from sklearn.utils.validation import check_is_fitted

# TODO:
# + Dont normalize the embeddings


class TextCleaner(BaseEstimator, TransformerMixin):
  def fit(self, X, y=None):
    return self

  def clean_text(self, text: str) -> str:
    """ Clean text from emojies and html tags """
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
    return re.sub(r"<[^>]+>", "", emoji_pattern.sub(r'', text))

  def transform(self, X: np.ndarray) -> np.ndarray:
    return np.array([self.clean_text(text) for text in X])


class DeNoiser(BaseEstimator, TransformerMixin):
  def __init__(self, threshold: float = 0.95):
    self.pca = None
    self.threshold = threshold
    self.is_fitted_ = False

  def fit(self, X, y=None):
    try:
      if self.pca is None:
        self.pca = PCA(n_components=self.threshold)
        self.pca.fit(X)
        self.is_fitted_ = True
    except Exception as e:
      print(f"Error fitting PCA: {str(e)}")
      raise
    return self

  def transform(self, X: np.ndarray) -> np.ndarray:
    """Transform texts into vector embeddings using the LLM"""
    if self.pca is None:
      raise ValueError("PCA not initialized. Call fit() first.")
    return self.pca.transform(X)


class LLMEncoder(BaseEstimator, TransformerMixin):
  def __init__(self, encoding_model_name: str = "all-MiniLM-L6-v2"):
    self.encoding_model_name = encoding_model_name
    self.encoding_model = None
    self.is_fitted_ = False

  def fit(self, X, y=None):
    try:
      if self.encoding_model is None:
        self.encoding_model = SentenceTransformer(self.encoding_model_name)
        self.is_fitted_ = True
    except Exception as e:
      print(f"Error loading model: {str(e)}")
      raise
    return self

  def transform(self, X: np.ndarray) -> np.ndarray:
    """Transform texts into vector embeddings using the LLM"""
    if self.encoding_model is None:
      raise ValueError("Model not initialized. Call fit() first.")

    return self.encoding_model.encode(
      X, convert_to_numpy=True, normalize_embeddings=False, show_progress_bar=False
    )


class CategoryMapper(BaseEstimator, TransformerMixin):
  def __init__(self, top_categories: Tuple[str]):
    self.top_categories = top_categories

  def fit(self, X, y=None):
    return self

  def transform(self, X: np.ndarray) -> np.ndarray:
    X = np.asarray(X).flatten()
    mapped_categories = []
    for val in X:
      val = val.split("|")[0].strip()
      if val in self.top_categories:
        mapped_categories.append(val)
      else:
        mapped_categories.append("Other")
    return np.array(mapped_categories).reshape(-1, 1)


class NumBinner(BaseEstimator, TransformerMixin):
  def __init__(self, discount_bins: Tuple[float]):
    self.discount_bins = discount_bins
    self.bin_labels = [f"bin_{i}" for i in range(len(discount_bins) + 1)]

  def fit(self, X, y=None):
    return self

  def transform(self, X: np.ndarray) -> np.ndarray:
    bins = [-np.inf] + list(self.discount_bins) + [np.inf]
    return np.array(pd.cut(X, bins=bins, labels=self.bin_labels)).reshape(-1, 1)


class LogTransformer(BaseEstimator, TransformerMixin):
  def fit(self, X, y=None):
    return self

  def transform(self, X: np.ndarray) -> np.ndarray:
    X = np.asarray(X).reshape(-1, 1)
    return np.log1p(X)


def get_text_pipeline(encoding_model_name: str) -> Pipeline:
  return Pipeline([
    ("cleaner", TextCleaner()),
    ("encoder", LLMEncoder(encoding_model_name=encoding_model_name)),
    # ("denoiser", DeNoiser()),
  ])


def get_discount_pipeline(discount_bins: Tuple[float]) -> Pipeline:
  return Pipeline([("binner", NumBinner(discount_bins)),
                   ("one_hot_encoder", OneHotEncoder(drop="first", sparse_output=False))])


def get_category_pipeline(top_categories: Tuple[str]) -> Pipeline:
  return Pipeline([("groupper", CategoryMapper(top_categories)),
                   ("one_hot_encoder", OneHotEncoder(drop="first", sparse_output=False))])


def get_price_pipeline() -> Pipeline:
  return Pipeline([("log_transformer", LogTransformer()), ("scaler", StandardScaler()),
                   ("minmax", MinMaxScaler((-1, 1)))])


def get_preprocess_pipeline(
  discount_bins: Tuple[float], top_categories: Tuple[str], encoding_model_name: str
) -> ColumnTransformer:
  return ColumnTransformer(
    transformers=[
      ("text", get_text_pipeline(encoding_model_name), "review_text"),
      ("category", get_category_pipeline(top_categories), "product_category"),
      ("discount", get_discount_pipeline(discount_bins), "discount_rate"),
      ("price", get_price_pipeline(), "price"),
    ]
  )


if __name__ == "__main__":
  import os
  import joblib
  from sklearn.model_selection import train_test_split

  test_size = 0.2
  random_state = 9001
  discount_bins = (0.0, 0.2, 0.6, 1.0)
  top_categories = ['Sports & Outdoors', 'Health & Personal Care', 'AMAZON FASHION']
  encoding_model_name = "all-MiniLM-L6-v2"
  output_dir = "../data/allMiniLM_NN/"
  os.makedirs(output_dir, exist_ok=True)

  df = pd.read_csv("../data/product_user_reviews.csv")
  X, y = df.drop("rating", axis=1), df[["rating"]]
  X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=test_size, random_state=random_state
  )

  preprocessor = get_preprocess_pipeline(discount_bins, top_categories, encoding_model_name)

  print("+ Processing training data...")
  processed_X_train = preprocessor.fit_transform(X_train, y_train)
  with open(os.path.join(output_dir, "train_features.npy"), "wb") as f:
    np.save(f, processed_X_train)

  with open(os.path.join(output_dir, "train_ratings.npy"), "wb") as f:
    np.save(f, y_train.to_numpy())

  print("+ Processing testing data...")
  processed_X_test = preprocessor.transform(X_test)
  with open(os.path.join(output_dir, "test_features.npy"), "wb") as f:
    np.save(f, processed_X_test)

  with open(os.path.join(output_dir, "test_ratings.npy"), "wb") as f:
    np.save(f, y_test.to_numpy())

  joblib.dump(preprocessor, os.path.join(output_dir, "preprocessor.joblib"))
  print(f"Processed datasets and fitted preprocessor saved in {output_dir}")
