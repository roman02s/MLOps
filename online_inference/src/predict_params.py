from typing import Optional, List, Dict
import yaml
from dataclasses import dataclass, field
from marshmallow_dataclass import class_schema


@dataclass()
class FeatureParams:
    categorical_features: Optional[List[str]] = field(default=None)
    numerical_features: Optional[List[str]] = field(default=None)
    features_to_drop: Optional[List[str]] = field(default=None)
    target_col: Optional[List[str]] = field(default=None)
    use_log_trick: bool = field(default=False)


@dataclass()
class PredictParams:
    model_path: str
    input_data_path: str
    feature_params: FeatureParams


@dataclass
class Params:
    features: Dict[str, str]


@dataclass
class ParamsBase:
    features: Dict[str, List[str]]

    @staticmethod
    def init():
        return ParamsBase(features={
            "Systemic Illness": ["Fever", "Muscle Aches and Pain",
                                 "None", "Swollen Lymph Nodes"],
            "Rectal Pain": ["False", "True"],
            "Sore Throat": ["False", "True"],
            "Penile Oedema": ["False", "True"],
            "Oral Lesions": ["False", "True"],
            "Solitary Lesion": ["False", "True"],
            "Swollen Tonsils": ["False", "True"],
            "HIV Infection": ["False", "True"],
            "Sexually Transmitted Infection": ["False", "True"],
            "MonkeyPox": ['Negative', 'Positive'],
        })


PredictParamsSchema = class_schema(PredictParams)


def read_predict_params(path: str):
    with open(path, 'r', encoding='utf-8') as input_stream:
        schema = PredictParamsSchema()
        return schema.load(yaml.safe_load(input_stream))
