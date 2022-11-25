import logging
import sys
import json

from fastapi.testclient import TestClient
from fastapi.encoders import jsonable_encoder

from online_inference.app import app
from online_inference.src.utils import (
    read_data,
    extract_target,
)
from online_inference.src.predict_params import (
    PredictParams,
    read_predict_params,
    Params,
)


logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.setLevel(logging.INFO)
logger.addHandler(handler)


client = TestClient(app)


def test_predict(config_path: str = "configs/LogisticRegression_config.yaml"):
    logger.info('Running model prediction')
    logger.info('Data preparation')

    predict_params: PredictParams = read_predict_params(config_path)

    data = read_data(predict_params.input_data_path)
    feature = extract_target(data, predict_params.feature_params)
    if predict_params.feature_params.target_col:
        feature = feature.drop(columns=[predict_params.feature_params.target_col[0]], axis=1)
    logger.info(f'Feature.shape: {feature.shape}')
    logger.info(feature)
    feature = feature[::100]
    for ind, dataline in feature.iterrows():
        data_to_model = Params(features=json.loads(dataline.to_json()))
        response = client.post("/predict/", json=jsonable_encoder(data_to_model))
        print(ind, response.status_code)
        assert response.status_code == 200
        assert response.json() is not None
