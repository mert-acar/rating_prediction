# Invent Analytics - ML Engineer Case Study

**Submitted by:** Mert Acar
**Date:** Dec 24, 2024

### Usage

##### Docker Container
Easiest way to run this project is using [docker](https://www.docker.com/). Install docker on your system and then run the following inside the project root:
```bash
docker build -t review-rating-predictor .
docker run -d -p 9001:9001 --name review-predictor review-rating-predictor
```

Once this is up and running you can use the API to make requests. Currently it supports GET requests on `/models` and POST requests on `/predict`. Here are two examples:

**List available model versions**:
```bash
> curl -s http://localhost:9001/models | jq .
[
  {
    "version": "0.1",
    "description": "Ridge without sample weights",
    "metrics": {
      "MAE": 1.682,
      "MSE": 4.434,
      "R2": 0.472,
      "RMSE": 2.106
    },
    "pipe_name": "model_v0.1",
    "timestamp": "2024-12-24T12:18:31.780667"
  },
  {
    "version": "0.2",
    "description": "Ridge with MPNet-base-v2 (n.s.w.)",
    "metrics": {
      "MAE": 1.541,
      "MSE": 3.764,
      "R2": 0.552,
      "RMSE": 1.94
    },
    "pipe_name": "model_v0.2",
    "timestamp": "2024-12-24T12:42:31.527931"
  }
]
```

**Predict using an available model**:
```bash
> curl -X POST http://localhost:9001/predict \
-H "Content-Type: application/json" \
-d '{
    "review_text": "This product is amazing and works perfectly!",
    "price": 99.99,
    "discount_rate": 0.1,
    "product_category": "Electronics"
    "model_version": "0.2" 
}'

{"predicted_rating":8.0}
```
Providing a model version is optional, if not provided the API will use the latest model for the query.

##### Build Your Own Environment
Docker container deploys the prediction API, however if you want to train new models or introduce new pre-training steps you need to build your own virtual environment to do so:
```bash
python3 -m virtualenv venv      # Create a virtual environment called 'venv'
source venv/bin/activate        # Activate the virtual environment
pip install --upgrade pip       # Upgrade python package manager
pip install -r requirements.txt # Install project dependencies
```
This is a one-time step, once the virtual environment is setup, you can just activate it and run the code.

In order to **serve the API**, run:
```bash
python3 src/api.py # or ./serve.sh
```
Once the API is served, you can use the same curl commands to run queries.

In order to **see the available models in the registry**, run:
```bash
python3 src/model_registry.py
```
If you want to **delete a specific version from the registry** run:
```
python3 src/model_registry.py delete <version_string>
```

In order to **create a new prepocessing pipeline**, set your parameters in `src/preprocess.py` and run:
```bash
python3 src/preprocess.py
```
In order to **train a new model**, specify model name and parameters in `src/train.py` and run:
```bash
python3 src/train.py <path_to_preprocessed_data>
```
In order to **run uni-tests**, use (this still requires an active virtual environment):
```bash
./test.sh
```


### Design

In this case study, the task of predicting product ratings was approached as a regression problem. The decision to frame the problem in this way was guided by the ordinal nature of the ratings. Specifically, the numerical values of the ratings carry intrinsic meaning and an order, which should be preserved to ensure the model's outputs carry some semantic meaning for the end-user. If the problem were treated as a classification task, the model would assign discrete labels to reviews without necessarily capturing the continuous relationships between the ratings. For example, if the true rating of a product is 2, a model predicting a 5 should be closer to the correct rating than a prediction of a 10. However, in a classification model, such a distinction would not be preserved, since each rating label is treated as equally distant from others.

#### Pre-Processing

From the provided dataset, 4 columns were selected for feature crafting: `price`, `discount_rate`, `product_category`, `review_text`. For more information on the statistical analysis and reasons behind selecting these, checkout `src/EDA.ipynb`.

The `price` column was transformed using:
$$ln(1 + \text{price})$$
This eliminates right-skew from the price values for a more symmetrical distribution.

`discount_rate`, `product_category` was binned and one-hot encoded to provide the model with context around the review. Since one-hot encoding creates co-linear vectors as features, the first element was dropped to allow our regression models to better learn relationships.

Lastly `review_text` was cleaned from HTML tags and emojies before being encoded by a pre-trained LLM. Since the reviews have varying lengths, the LLM outputs were truncated and mean-pooled to create paragraph embeddings of fixed length. This is made possible by the realization that only 1.54% of all reviews are longer than the accepted maximum sequence length for the smallest LLM ([`all-Mini-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)) 

The final model input came out to be `[N, 3 + 3 + 1 + llm_encoding_dim]` where binned variables each contribute 3 binary-features, 1 coming from the log-transformed price and rest are coming from the LLM embeddings. `N` is the batch size.
#### Training
The training is done using `sklearn`'s `GridSearchCV` for hyper-parameter search. Any regularization model can be imported and fitted to the available data. 
