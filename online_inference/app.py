from typing import List, Dict
import os
import logging
import sys

import pandas as pd
import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse

from online_inference.src.predict import run_predict
from online_inference.src.predict_params import Params, ParamsBase

app = FastAPI()
model = None
s3_bucket = None


logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.setLevel(logging.INFO)
logger.addHandler(handler)


@app.get("/")
def read_root():
    return FileResponse("static/get_features.html")


@app.get("/health")
def health() -> bool:
    return not (model is None) and \
           not (s3_bucket is not None)


@app.post("/predict/")
def predict(data: Params):
    logger.info("predict data for ", data.features['Patient_ID'])
    features: Dict[str, List[str]] = ParamsBase.init().features
    print(data)
    features_int = {}
    for ind, feature in data.features.items():
        if ind not in features.keys():
            continue
        for i, x in enumerate(features[ind], 0):
            if x == feature:
                features_int[ind] = [i]
    predictions = run_predict("models/LogisticRegression.pkl", pd.DataFrame(features_int))
    if int(predictions[0]) == 1:
        message = "Вы болеете"
    else:
        message = "Вы не болеете"
    result = {"message": f"{data.features['Patient_ID']}, {message}"}
    return result


if __name__ == "__main__":
    uvicorn.run("app:app", host=os.getenv("HOST", "0.0.0.0"), port=os.getenv("PORT", 8000))
