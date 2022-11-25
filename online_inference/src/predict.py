import logging
import pickle
import sys

import pandas as pd
import numpy as np

from online_inference.src.predict_params import FeatureParams


logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.setLevel(logging.INFO)
logger.addHandler(handler)


def load_model(path: str):
    logger.info('model loading')
    with open(path, 'rb') as model:
        return pickle.load(model)


def predict_model(
    model, features: pd.DataFrame, use_log_trick: bool = False
) -> pd.DataFrame:
    predicts = pd.DataFrame(model.predict(features))
    if use_log_trick:
        predicts = np.exp(predicts)
    logger.info('predict model')
    return predicts


def run_predict(model_path: str, data: pd.DataFrame) -> pd.DataFrame:
    logger.info('Running model prediction')
    logger.info('Data preparation')

    model = load_model(model_path)

    predict = predict_model(model, data)
    logger.info(f'Prediction.shape: {predict.shape}')
    return predict


def extract_target(df: pd.DataFrame, params: FeatureParams) -> pd.DataFrame:
    columns: list = params.categorical_features + params.target_col
    target = pd.DataFrame(df[columns])
    for column in columns:
        list_uniq_column = np.unique(target[column])
        target[column].replace(list_uniq_column, [*range(len(list_uniq_column))], inplace=True)
    if params.use_log_trick:
        target = pd.DataFrame(np.log(target.to_numpy()))
    logger.info("extract target: ", target.shape)
    return target
