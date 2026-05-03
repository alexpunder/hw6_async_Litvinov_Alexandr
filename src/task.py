import joblib
import numpy as np
from celery import Celery

app = Celery("tasks", broker="redis://localhost:6379/0", backend="redis://localhost:6379/0")

model = joblib.load("iris_model.pkl")
scaler = joblib.load("scaler.pkl")

@app.task
def predict_task(data):
    array = np.array([data])
    data_sacled = scaler.transform(array)
    pred = model.predict(data_sacled)
    return str(*pred)
