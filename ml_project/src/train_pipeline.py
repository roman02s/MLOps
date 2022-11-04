import json
import logging
import sys

import click

from ml_project.src.data import read_data, split_train_test_data
from ml_project.src.enities.train_pipeline_params import (
    read_training_pipeline_params,
)
from ml_project.src.features import make_features
from ml_project.src.features.build_features import (
    extract_target,
    build_transformer,
    build_transformer_target,
)
from ml_project.src.models import (
    train_model,
    serialize_model,
    predict_model,
    evaluate_model,
)
import mlflow

from ml_project.src.models.model_fit_predict import create_inference_pipeline

logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.setLevel(logging.INFO)
logger.addHandler(handler)


def train_pipeline(config_path: str):
    training_pipeline_params = read_training_pipeline_params(config_path)

    if training_pipeline_params.use_mlflow:

        mlflow.set_tracking_uri(training_pipeline_params.mlflow_uri)
        mlflow.set_experiment(training_pipeline_params.mlflow_experiment)
        with mlflow.start_run():
            mlflow.log_artifact(config_path)
            model_path, metrics = run_train_pipeline(training_pipeline_params)
            mlflow.log_metrics(metrics)
            mlflow.log_artifact(model_path)
    else:
        return run_train_pipeline(training_pipeline_params)


def run_train_pipeline(training_pipeline_params):
    logger.info(f"start train pipeline with params {training_pipeline_params}")
    data = read_data(training_pipeline_params.input_data_path)
    data = extract_target(data, training_pipeline_params.feature_params)
    logger.info(f"data.shape is {data.shape}")
    train_df, val_df = split_train_test_data(
        data, training_pipeline_params.splitting_params
    )
    logger.info(f"train_df.shape is {train_df.shape}")
    logger.info(f"val_df.shape is {val_df.shape}")
    X_train_df = train_df.drop(training_pipeline_params.feature_params.target_col, axis=1)
    y_train_df = train_df[training_pipeline_params.feature_params.target_col]
    logger.info(f"X_train_df.shape is {X_train_df.shape}")
    logger.info(f"y_train_df.shape is {y_train_df.shape}")
    transformer = build_transformer(training_pipeline_params.feature_params)
    transformer.fit(X_train_df, y_train_df)
    model = train_model(
        X_train_df, y_train_df, training_pipeline_params.train_params
    )
    X_val_df = val_df.drop(training_pipeline_params.feature_params.target_col, axis=1)
    y_val_df = val_df[training_pipeline_params.feature_params.target_col]
    y_val_df_predicts = predict_model(
        model,
        X_val_df,
        training_pipeline_params.feature_params.use_log_trick,
    )
    logger.info(f"predicts.shape is {y_val_df_predicts.shape}")
    metrics = evaluate_model(
        y_val_df_predicts,
        y_val_df,
        use_log_trick=training_pipeline_params.feature_params.use_log_trick,
    )
    with open(training_pipeline_params.metric_path, "w") as metric_file:
        json.dump(metrics, metric_file)
    logger.info(f"metrics is {metrics}")

    path_to_model = serialize_model(
        model, training_pipeline_params.output_model_path
    )
    return path_to_model, metrics


@click.command(name="train_pipeline")
@click.argument("config_path", default='configs/train_config.yaml')
def train_pipeline_command(config_path: str):
    train_pipeline(config_path)


if __name__ == "__main__":
    train_pipeline_command()
