import sys
import logging
from typing import Tuple
import pandas as pd
from sklearn.model_selection import train_test_split

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


def split_train_test_data(data: pd.DataFrame, params) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Function for splitting dataset on train and test sets"""
    logger.info('create train and test set')
    train_data, test_data = train_test_split(
        data,
        test_size=params.val_size,
        random_state=params.random_state,
    )
    logger.info('train and test set created')
    return train_data, test_data
