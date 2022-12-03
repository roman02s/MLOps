import os
import pickle

import click
import pandas as pd


@click.command("predict")
@click.option("--input-dir")
@click.option("--model-dir")
@click.option("--output-dir")
def predict(input_dir: str, model_dir: str, output_dir: str):
    with open(os.path.join(model_dir, "LogisticRegression.pkl"), "rb") as f:
        model = pickle.load(f)
        x_data = pd.read_csv(os.path.join(input_dir, "data.csv"))
        y_pred = model.predict(x_data)
        os.makedirs(output_dir, exist_ok=True)
        pd.DataFrame(y_pred).to_csv(os.path.join(output_dir, "predictions.csv"), index=False)


if __name__ == "__main__":
    predict()
