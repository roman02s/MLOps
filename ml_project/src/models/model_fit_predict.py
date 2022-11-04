import pickle
from typing import Dict, Union

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from sklearn.pipeline import Pipeline

from ml_project.src.enities.train_params import TrainingParams

SklearnRegressionModel = Union[RandomForestRegressor, LogisticRegression]


def train_model(
    features: pd.DataFrame, target: pd.DataFrame, train_params: TrainingParams
) -> SklearnRegressionModel:
    if train_params.model_type == "RandomForestRegressor":
        model = RandomForestRegressor(
            n_estimators=100, random_state=train_params.random_state
        )
    elif train_params.model_type == "LogisticRegression":
        model = LogisticRegression()
    else:
        raise NotImplementedError()
    model.fit(features, target)
    return model


def predict_model(
    model: SklearnRegressionModel, features: pd.DataFrame, use_log_trick: bool = False
) -> pd.DataFrame:
    predicts = pd.DataFrame(model.predict(features))
    if use_log_trick:
        predicts = np.exp(predicts)
    return predicts


def evaluate_model(
    predicts: pd.DataFrame, target: pd.DataFrame, use_log_trick: bool = False
) -> Dict[str, float]:
    if use_log_trick:
        target = np.exp(target)
    return {
        "Roc|Auc": roc_auc_score(predicts, target),
        "Accuracy": accuracy_score(predicts, target),
        "F1": f1_score(predicts, target),
    }


def create_inference_pipeline(
    model: SklearnRegressionModel, transformer: ColumnTransformer
) -> Pipeline:
    return Pipeline([("feature_part", transformer), ("model_part", model)])


def serialize_model(model: object, output: str) -> str:
    with open(output, "wb") as f:
        pickle.dump(model, f)
    return output
