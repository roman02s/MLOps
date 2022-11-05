from typing import List
from dataclasses import dataclass
from marshmallow_dataclass import class_schema
import yaml

from ml_project.src.enities.feature_params import FeatureParams


@dataclass()
class PredictParams:
    input_data_path: str
    model_path: str
    predict_path: str
    transformer_path: str
    target_in_dataset: bool
    target: str
    feature_params: FeatureParams


PredictParamsSchema = class_schema(PredictParams)


def read_predict_params(path: str):
    with open(path, 'r', encoding='utf-8') as input_stream:
        schema = PredictParamsSchema()
        return schema.load(yaml.safe_load(input_stream))
