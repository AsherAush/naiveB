from fastapi import FastAPI, Query, HTTPException, Request
from data_loader import DataLoader
from naive_bayes import NaiveBayesClassifier
from predictor import NaiveBayesPredictor

app = FastAPI()

# טען את הנתונים פעם אחת כששרת עולה
loader = DataLoader("data for NB buys computer.csv")
loader.load()
columns_to_drop = ["id"]
loader.drop_columns(columns_to_drop)
df = loader.get_data()

# אימון המודל
model = NaiveBayesClassifier()
model.fit(df)
predictor = NaiveBayesPredictor(model)


@app.get("/predict")
def predict(request: Request):
    # שלוף את כל הפרמטרים שהוזנו ב-URL
    observation = dict(request.query_params)

    if not observation:
        return {"error": "Please provide at least one query parameter for prediction."}

    prediction = predictor.predict(observation)
    return {"prediction": prediction}