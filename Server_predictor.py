from fastapi import FastAPI, Request
from predictor import NaiveBayesPredictor
from naive_bayes import NaiveBayesClassifier

app = FastAPI()
predictor = None

@app.post("/load_model")
def load_model(model_dict: dict):
    global predictor
    model = NaiveBayesClassifier.from_dict(model_dict)
    predictor = NaiveBayesPredictor(model)
    return {"status": "Model loaded successfully"}

@app.get("/predict")
def predict(request: Request):
    global predictor
    if predictor is None:
        return {"error": "Model is not loaded"}
    observation = dict(request.query_params)
    return {"prediction": predictor.predict(observation)}