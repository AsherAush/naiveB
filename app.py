from predictor import NaiveBayesPredictor
import uvicorn

from fastapi import FastAPI, Request, Response

app = FastAPI()


@app.post("/train")
async def train_model(request: Request):

    data = request.json()
    print("Data received for training:", data)
    return {"message": "Model trained successfully"}




@app.get("/")
async def predict(request: Request):
    didt_qerey = dict(request.query_params)
    print("Query parameters received:", didt_qerey)
    answer = a.pridict(didt_qerey,)
    return {"prediction": answer}


if __name__ == "__main__":
    a = NaiveBayesPredictor
    uvicorn.run(app, host="127.0.0.1", port=8000)
