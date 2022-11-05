import logging
import sys
from typing import NoReturn
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin

logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.setLevel(logging.INFO)
logger.addHandler(handler)


class CustomTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, features) -> NoReturn:
        logger.info("Create CustomTransformer")
        self.scaler = OneHotEncoder()
        self.num_features = features.num_features

    def fit(self, data: pd.DataFrame):
        self.scaler.fit(data[self.num_features])
        return self

    def transform(self, data: pd.DataFrame):
        return self.scaler.transform(data[self.num_features])
