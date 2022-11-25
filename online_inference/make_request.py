import logging
import sys
import time
import json
import click

from fastapi.encoders import jsonable_encoder

import requests


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


@click.command(name='run_predict')
@click.argument('config_path', default="configs/LogisticRegression_config.yaml")
def predict(config_path):
    logger.info('Running model prediction')
    logger.info('Data preparation')

    predict_params: PredictParams = read_predict_params(config_path)

    data = read_data(predict_params.input_data_path)
    feature = extract_target(data, predict_params.feature_params)
    if predict_params.feature_params.target_col:
        feature = feature.drop(columns=[predict_params.feature_params.target_col[0]], axis=1)
    logger.info(f'Feature.shape: {feature.shape}')
    logger.info(feature)

    for ind, dataline in feature.iterrows():
        try:
            data_to_model = Params(features=json.loads(dataline.to_json()))
            response = requests.post("http://0.0.0.0:8000/predict/", json=jsonable_encoder(data_to_model))
            print(ind, response.status_code)
            time.sleep(3)
        except KeyError:
            print("Stop requests")
            break
        except BaseException as err:
            print(f"Error: {err}")
            continue


if __name__ == "__main__":
    predict()
