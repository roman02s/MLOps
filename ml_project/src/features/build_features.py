import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from ml_project.src.enities.feature_params import FeatureParams


def process_categorical_features(categorical_df: pd.DataFrame) -> pd.DataFrame:

    categorical_pipeline = build_categorical_pipeline()
    return pd.DataFrame(categorical_pipeline.fit_transform(categorical_df).toarray())


def build_categorical_pipeline() -> Pipeline:
    categorical_pipeline = Pipeline(
        [
            # ("impute", SimpleImputer(missing_values=np.nan, strategy="most_frequent")),
            ("ohe", OneHotEncoder()),
        ]
    )
    return categorical_pipeline


def process_numerical_features(numerical_df: pd.DataFrame) -> pd.DataFrame:
    num_pipeline = build_numerical_pipeline()
    return pd.DataFrame(num_pipeline.fit_transform(numerical_df))


def build_numerical_pipeline() -> Pipeline:
    num_pipeline = Pipeline(
        [("impute", SimpleImputer(missing_values=np.nan, strategy="mean")),]
    )
    return num_pipeline


def build_target_col_pipeline() -> Pipeline:
    return build_categorical_pipeline()


def make_features(transformer: ColumnTransformer, df: pd.DataFrame) -> pd.DataFrame:
    return transformer.transform(df)


def build_transformer(params: FeatureParams) -> ColumnTransformer:
    list_pipeline_transformer = []
    if params.categorical_features:
        list_pipeline_transformer.append(
            (
                "categorical_pipeline",
                build_categorical_pipeline(),
                params.categorical_features,
            )
        )
    if params.numerical_features:
        list_pipeline_transformer.append(
            (
                "numerical_pipeline",
                build_numerical_pipeline(),
                params.numerical_features,
            )
        )
    transformer = ColumnTransformer(
        list_pipeline_transformer
    )
    return transformer


def build_transformer_target(val_df: pd.DataFrame, params: FeatureParams) -> ColumnTransformer:
    list_pipeline_transformer = [(
        (
            "target_col_pipeline",
            build_target_col_pipeline(),
            params.target_col,
        )
    )]
    transformer = ColumnTransformer(
        list_pipeline_transformer
    )
    return transformer


def extract_target(df: pd.DataFrame, params: FeatureParams) -> pd.DataFrame:
    columns: list = params.categorical_features + params.target_col
    target = pd.DataFrame(df[columns])
    for column in columns:
        list_uniq_column = np.unique(target[column])
        target[column].replace(list_uniq_column, [*range(len(list_uniq_column))], inplace=True)
    if params.use_log_trick:
        target = pd.DataFrame(np.log(target.to_numpy()))
    return target
