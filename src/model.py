import torch
import torch.nn.functional as F
from typing import List, Optional
from transformers import AutoTokenizer, AutoModel


class RatingPredictor(torch.nn.Module):
  def __init__(
    self,
    encoding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    in_dim: Optional[int] = None
  ):
    super().__init__()
    self.tokenizer = AutoTokenizer.from_pretrained(encoding_model)
    self.encoding_model = AutoModel.from_pretrained(encoding_model)
    if in_dim is None:
      in_dim = self.encoding_model.config.hidden_size

    # Simpler classifier with one hidden layer and stronger regularization
    self.classifier = torch.nn.Sequential(
      torch.nn.Linear(in_dim, 64),
      torch.nn.ReLU(),
      torch.nn.Dropout(0.5),  # Increased dropout
      torch.nn.Linear(64, 1),
      torch.nn.Sigmoid()
    )

    # Initialize weights with smaller values
    for m in self.classifier.modules():
      if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_normal_(m.weight, gain=0.1)
        torch.nn.init.constant_(m.bias, 0)

  def _mean_pooling(self, model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
      input_mask_expanded.sum(1), min=1e-9
    )

  def freeze_embedding_model(self):
    for p in self.encoding_model.parameters():
      p.requires_grad = False

  def get_device(self):
    return next(self.parameters()).device

  def forward(self, review: List[str]) -> torch.Tensor:
    device = self.get_device()
    encoded_str = self.tokenizer(
      review, padding=True, truncation=True, return_tensors='pt'
    ).to(device)
    out = self.encoding_model(**encoded_str)
    embeddings = self._mean_pooling(out, encoded_str["attention_mask"])
    embeddings = F.normalize(embeddings, p=2, dim=1)
    return self.classifier(embeddings)
