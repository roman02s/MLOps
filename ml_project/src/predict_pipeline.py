import logging
import sys

import click

import pandas as pd

from ml_project.src.data import read_data
from ml_project.src.enities.predict_params import PredictParams, read_predict_params
from ml_project.src.models.model_fit_predict import predict_model, load_model
from ml_project.src.features.build_features import (
    extract_target,
)
from ml_project.src.models import (
    predict_model,
    evaluate_model,
)
logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.setLevel(logging.INFO)
logger.addHandler(handler)


def run_predict(config_path: str):
    logger.info('Running model prediction')
    logger.info('Data preparation')

    predict_params: PredictParams = read_predict_params(config_path)

    data = read_data(predict_params.input_data_path)
    feature = extract_target(data, predict_params.feature_params)
    if predict_params.target_in_dataset:
        feature = feature.drop(columns=[predict_params.target], axis=1)
    logger.info(f'Feature.shape: {feature.shape}')
    model = load_model(predict_params.model_path)

    predict = predict_model(model, feature)
    logger.info(f'Prediction.shape: {predict.shape}')
    logger.info(f'Writing a prediction to a file {predict_params.predict_path}')
    pd.DataFrame(predict, columns=[0]).to_csv(predict_params.predict_path)


@click.command(name='run_predict')
@click.argument('config_path', default="configs/LogisticRegression_predict_config.yaml")
def run_predict_command(config_path: str):
    run_predict(config_path)


if __name__ == '__main__':
    run_predict_command()
