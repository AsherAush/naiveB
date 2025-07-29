from fastapi import FastAPI, Request
from predictor import NaiveBayesPredictor
import requests
import uvicorn
# This is the predictor server that fetches the model from the training server and uses it for predictions.
app = FastAPI()

model = None
predictor = None

# Fetch the model from the training server when the application starts
@app.on_event("startup")
def fetch_model_from_server1():
    global model, predictor
    try:
        # בצע את הבקשה לשרת המאמן
        response = requests.get("http://server-train:8006/train")
        if response.status_code == 200:
            model_dict = response.json().get("dic")
            if model_dict:
                predictor = NaiveBayesPredictor(model_dict)
                print("Model successfully fetched and predictor initialized.")
            else:
                print("Model dict not found in response!")
        else:
            print(f"Failed to fetch model from Server 1. Status code: {response.status_code}")
    except Exception as e:
        print(f"Exception occurred while fetching model: {e}")
@app.get("/predict")
def predict(request: Request):
    # שלוף את כל הפרמטרים שהוזנו ב-URL
    observation = dict(request.query_params)

    if not observation:
        return {"error": "Please provide at least one query parameter for prediction."}

    prediction = predictor.predict(observation)
    return {"prediction": prediction}

if __name__ == "__main__":
    uvicorn.run(app, host= "127.0.0.1" , port=8007)
