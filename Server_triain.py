from fastapi import FastAPI, Query, HTTPException, Request
from data_loader import DataLoader
from naive_bayes import NaiveBayesClassifier

import uvicorn

app = FastAPI()

@app.get("/train")
def train_model():
    loader = DataLoader("data for NB buys computer.csv")
    loader.load()
    columns_to_drop = ["id"]
    loader.drop_columns(columns_to_drop)
    df = loader.get_data()

    model = NaiveBayesClassifier()
    dict_model = model.fit(df)

    return {"dic": dict_model}


if __name__ == "__main__":
    uvicorn.run(app, host= "127.0.0.1" , port=8006)
