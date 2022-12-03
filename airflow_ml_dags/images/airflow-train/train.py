import os
import pickle

import click
import pandas as pd
from sklearn.linear_model import LogisticRegression


@click.command("train")
@click.option("--input-dir")
@click.option("--output-dir")
def train(input_dir: str, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)

    x_data = pd.read_csv(os.path.join(input_dir, "x_train.csv"))
    y = pd.read_csv(os.path.join(input_dir, "y_train.csv"))

    model = LogisticRegression()
    model.fit(x_data, y)

    with open(os.path.join(output_dir, "LogisticRegression.pkl"), "wb") as f:
        pickle.dump(model, f)


if __name__ == "__main__":
    train()
