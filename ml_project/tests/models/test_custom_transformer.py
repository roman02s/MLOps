from typing import List, Tuple

import pandas as pd
import pytest

from ml_project.src.data.make_dataset import read_data
from ml_project.src.enities.feature_params import FeatureParams
from ml_project.src.features.build_features import make_features, extract_target, build_transformer
from ml_project.src.models.custom_transformer import CustomTransformer
from sklearn.preprocessing import OneHotEncoder


@pytest.fixture
def features_and_target(
    dataset_path: str, categorical_features: List[str], numerical_features: List[str]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    params = FeatureParams(
        categorical_features=categorical_features,
        numerical_features=numerical_features,
        features_to_drop=["Patient_ID"],
        target_col=["MonkeyPox"],
    )
    data = read_data(dataset_path)
    transformer = build_transformer(params)
    transformer.fit(data)
    features = make_features(transformer, data)
    target = extract_target(data, params)
    return features, target


def test_custom_transformer(features_and_target: Tuple[pd.DataFrame, pd.DataFrame],
                            dataset_path: str):
    features, target = features_and_target
    data = read_data(dataset_path)
    transformer = CustomTransformer(features)
    transformer.fit(data)
    check_transformer = OneHotEncoder(features)
    check_transformer.fit(data)
    assert transformer.transform(data) == check_transformer.transform(data)
