import json
import os
import pickle

import click
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score


@click.command("validate")
@click.option("--input-dir")
@click.option("--model-dir")
@click.option("--output-dir")
def validate(input_dir: str, model_dir: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    x_data = pd.read_csv(os.path.join(input_dir, "x_train.csv"))
    y_true = pd.read_csv(os.path.join(input_dir, "y_train.csv"))

    with open(os.path.join(model_dir, "LogisticRegression.pkl"), "rb") as model_file:
        model = pickle.load(model_file)
    y_pred = model.predict(x_data)

    metrics = dict()
    metrics["Accuracy"] = accuracy_score(y_true, y_pred)
    metrics["F1"] = f1_score(y_true, y_pred)

    with open(os.path.join(output_dir, "metric.json"), "w") as metric_file:
        json.dump(metrics, metric_file)


if __name__ == "__main__":
    validate()
