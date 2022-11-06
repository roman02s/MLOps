import os
from typing import List

from py._path.local import LocalPath

from ml_project.src.models.model_fit_predict import create_inference_pipeline
from ml_project.src.enities import (
    TrainingPipelineParams,
    SplittingParams,
    FeatureParams,
    TrainingParams,
)


def test_train_e2e(
    tmpdir: LocalPath,
    dataset_path: str,
    categorical_features: List[str],
    numerical_features: List[str],
    target_col: str,
    features_to_drop: List[str],
):
    expected_output_model_path = tmpdir.join("LogisticRegression.pkl")
    expected_metric_path = tmpdir.join("LogisticRegression_metrics_train.json")
    params = TrainingPipelineParams(
        input_data_path=dataset_path,
        output_model_path=expected_output_model_path,
        metric_path=expected_metric_path,
        splitting_params=SplittingParams(val_size=0.2, random_state=239),
        feature_params=FeatureParams(
            numerical_features=numerical_features,
            categorical_features=categorical_features,
            target_col=[target_col],
            features_to_drop=features_to_drop,
            use_log_trick=True,
        ),
        train_params=TrainingParams(model_type="LogisticRegression"),
    )
    real_model_path, metrics = create_inference_pipeline(params)
    assert metrics["Accuracy"] > 0
    assert os.path.exists(real_model_path)
    assert os.path.exists(params.metric_path)
