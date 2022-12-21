import os
import pickle
from io import StringIO

import numpy as np
import requests
import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse


class CONFIG:
    model_url = 'https://made.hb.bizmrg.com/model.pkl'
    model_path = './model/model.pkl'
    host = '0.0.0.0'
    port = 8000
    test_data = './data/test_data.csv'


def download_model():
    os.makedirs(os.path.dirname(CONFIG.model_path), exist_ok=True)
    model = requests.get(CONFIG.model_url)
    with open(CONFIG.model_path, 'wb') as f:
        f.write(model.content)


app = FastAPI()
download_model()


@app.get("/")
async def main():
    return 'This is classifier web-service. Do POST request with ' + \
           'features-file at endpoint /predict, or use python client.py'


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = file.file.read().decode()
    with open(CONFIG.model_path, 'rb') as f:
        model = pickle.load(f)
    data = np.genfromtxt(StringIO(contents), skip_header=True, delimiter=",")
    preds = model.predict(data)
    stream = StringIO()
    np.savetxt(stream, preds, delimiter=",", fmt="%d")
    response = StreamingResponse(iter([stream.getvalue()]),
                                 media_type="text/csv")

    return response


@app.get("/health", status_code=200)
async def check_health():
    with open(CONFIG.model_path, 'rb') as f:
        model = pickle.load(f)
    return 'OK'


if __name__ == '__main__':
    uvicorn.run("app:app", host=CONFIG.host, port=CONFIG.port, reload=True)
