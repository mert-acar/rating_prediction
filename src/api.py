import numpy as np
import pandas as pd
from typing import Optional
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException
from model_registry import ModelRegistry
import warnings

app = FastAPI(title="Review Rating Predictor")
model_registry = ModelRegistry()

warnings.simplefilter("ignore", FutureWarning)


class PredictionRequest(BaseModel):
  review_text: str = Field(..., description="Customer review text")
  price: float = Field(..., description="Product price")
  discount_rate: float = Field(..., ge=0, le=1, description="Discount rate (0-1)")
  product_category: str = Field(..., description="Product category")
  model_version: Optional[str] = Field(None, description="Model version to use")


class PredictionResponse(BaseModel):
  predicted_rating: float
  model_version: str


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest) -> PredictionResponse:
  try:
    version = request.model_version or model_registry.get_latest_version()
    pipeline = model_registry.load_model(version)
    input_data = pd.DataFrame({
      'review_text': [request.review_text],
      'price': [request.price],
      'discount_rate': [request.discount_rate],
      'product_category': [request.product_category]
    })
    prediction = np.clip(np.round(pipeline.predict(input_data)[0]), 1, 10)
    return PredictionResponse(predicted_rating=float(prediction))
  except ValueError as e:
    raise HTTPException(status_code=400, detail=str(e))
  except Exception as e:
    raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.get("/models")
async def list_models():
  return model_registry.list_versions()


if __name__ == "__main__":
  import uvicorn
  uvicorn.run(app, host="0.0.0.0", port=9001)
