import torch
import torch.nn.functional as F
from typing import List, Optional, Tuple
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
      in_dim = self.encoding_model.config.hidden_size + 9

    # Shared layers
    self.regression_head = torch.nn.Sequential(
      torch.nn.Linear(in_dim, 64), torch.nn.ReLU(), torch.nn.Dropout(0.2), torch.nn.Linear(64, 1),
      torch.nn.Sigmoid()
    )

  def _mean_pooling(self, model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
      input_mask_expanded.sum(1), min=1e-9
    )

  def freeze_embedding_model(self, freeze_until: Optional[int] = None):
    for p in self.encoding_model.parameters():
      p.requires_grad = False

    if freeze_until is not None:
      for layer in self.encoding_model.encoder.layer[freeze_until:]:
        for param in layer.parameters():
          param.requires_grad = True
      for param in self.encoding_model.pooler.parameters():
        param.requires_grad = True

  def get_device(self):
    return next(self.parameters()).device

  def forward(self, review: List[str], features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    device = self.get_device()
    encoded_str = self.tokenizer(
      review, padding=True, truncation=True, return_tensors='pt'
    ).to(device)
    out = self.encoding_model(**encoded_str)
    embeddings = self._mean_pooling(out, encoded_str["attention_mask"])
    embeddings = F.normalize(embeddings, p=2, dim=1)
    feat = torch.cat([features, embeddings], dim=1)
    return self.regression_head(feat)
