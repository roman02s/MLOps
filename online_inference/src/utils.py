import sys
import logging
import pickle

import pandas as pd
import numpy as np

from online_inference.src.predict_params import FeatureParams

logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.setLevel(logging.INFO)
logger.addHandler(handler)


def read_data(path: str) -> pd.DataFrame:
    """function for download dataset"""
    logger.info(f'file download start {path}')
    data = pd.read_csv(path)
    logger.info(f'loading is complete {path}')
    return data


def load_model(path: str):
    logger.info('model loading')
    with open(path, 'rb') as model:
        return pickle.load(model)


def extract_target(df: pd.DataFrame, params: FeatureParams) -> pd.DataFrame:
    columns: list = params.categorical_features + params.target_col + params.features_to_drop
    target = pd.DataFrame(df[columns])
    logger.info("extract target: ", target.shape)
    if params.use_log_trick:
        target = pd.DataFrame(np.log(target.to_numpy()))
    return target
