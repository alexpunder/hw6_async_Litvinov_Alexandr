from fastapi import FastAPI
from pydantic import BaseModel
from task import predict_task

api = FastAPI()

class IrisFeatures(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

@api.post("/predict")
def predict(data: IrisFeatures):
    features = [
        data.sepal_length, data.sepal_width, 
        data.petal_length, data.petal_width,
    ]
    task = predict_task.delay(features)
    return {"task_id": task.id}

@api.get("/result/{task_id}")
def get_result(task_id: str):
    res = predict_task.AsyncResult(task_id)
    if res.ready():
        return {"prediction": res.get()}
    return {"status": "processing"}
