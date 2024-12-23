import os
import unittest

import numpy as np
import pandas as pd
from preprocess import get_text_pipeline, get_discount_pipeline, get_category_pipeline, get_price_pipeline, get_preprocess_pipeline


class TestPreprocessingPipeline(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cls.n = 64
    cls.df = pd.read_csv(os.path.join(root_path, "data", "product_user_reviews.csv"))
    cls.config = {
      "discount_bins": (0.0, 0.2, 0.6, 1.0),
      "top_categories": ['Sports & Outdoors', 'Health & Personal Care', 'AMAZON FASHION'],
      "encoding_model_name": "all-MiniLM-L6-v2",
      "output_dir": "../data/allMiniLM/",
    }

  def test_text_pipe(self):
    pipe = get_text_pipeline(self.config["encoding_model_name"])
    data = self.df.sample(n=self.n)
    processed = pipe.fit_transform(data["review_text"])
    self.assertSequenceEqual(processed.shape, (self.n, 384))
    self.assertTrue(all(np.round(np.linalg.norm(p), 4) == 1.0 for p in processed))

  def test_price_pipe(self):
    pipe = get_price_pipeline()
    data = self.df.sample(n=self.n)
    processed = pipe.fit_transform(data["price"])
    self.assertTrue(all(-1 <= np.round(price, 5) <= 1 for price in processed.flatten()))

  def test_category_pipe(self):
    pipe = get_category_pipeline(self.config["top_categories"])
    data = self.df.sample(n=self.n)
    processed = pipe.fit_transform(data["product_category"])
    self.assertSequenceEqual(processed.shape, (self.n, 3))
    self.assertTrue(all([row <= 1 for row in processed.sum(1)]))

  def test_discount_pipe(self):
    pipe = get_discount_pipeline(self.config["discount_bins"])
    data = self.df.sample(n=self.n)
    processed = pipe.fit_transform(data["discount_rate"])
    self.assertSequenceEqual(processed.shape, [self.n, 3])
    self.assertTrue(all([row <= 1 for row in processed.sum(1)]))

  def test_pipe(self):
    pipe = get_preprocess_pipeline(
      self.config["discount_bins"],
      self.config["top_categories"],
      self.config["encoding_model_name"],
    )
    data = self.df.sample(n=self.n)
    processed = pipe.fit_transform(data)
    self.assertSequenceEqual(processed.shape, (self.n, 391))


if __name__ == "__main__":
  unittest.main()
