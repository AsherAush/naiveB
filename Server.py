from fastapi import FastAPI, UploadFile, Form
from pydantic import BaseModel
import pandas as pd
from data_loader import DataLoader
from naive_bayes import NaiveBayesClassifier
from predictor import NaiveBayesPredictor
import tempfile

app = FastAPI()

model = None
predictor = None


class PredictionRequest(BaseModel):
    features: dict


@app.post("/train/")
async def train_model(file: UploadFile, columns_to_drop: str = Form("")):
    global model, predictor

    # שמירה זמנית של הקובץ
    temp = tempfile.NamedTemporaryFile(delete=False)
    temp.write(await file.read())
    temp.close()

    # טענת הקובץ
    loader = DataLoader(temp.name)
    loader.load()

    if columns_to_drop:
        columns = [c.strip() for c in columns_to_drop.split(",")]
        loader.drop_columns(columns)

    df = loader.get_data()
    model = NaiveBayesClassifier()
    model.fit(df)
    predictor = NaiveBayesPredictor(model)

    return {"message": "Model trained successfully", "features": list(model.columns)}


@app.post("/predict/")
def predict(request: PredictionRequest):
    if predictor is None:
        return {"error": "Model is not trained yet"}

    prediction = predictor.predict(request.features)
    return {"prediction": prediction}
