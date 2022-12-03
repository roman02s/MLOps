import os

import click
import pandas as pd
from sklearn.model_selection import train_test_split


@click.command("split")
@click.option("--input-dir")
def split_data(input_dir: str) -> None:
    x_data = pd.read_csv(os.path.join(input_dir, "train_data.csv"))
    target = pd.read_csv(os.path.join(input_dir, "target.csv"))

    x_train, x_val, y_train, y_val = train_test_split(x_data, target, test_size=0.2)

    x_train.to_csv(os.path.join(input_dir, "x_train.csv"), index=False)
    x_val.to_csv(os.path.join(input_dir, "x_val.csv"), index=False)
    y_train.to_csv(os.path.join(input_dir, "y_train.csv"), index=False)
    y_val.to_csv(os.path.join(input_dir, "y_val.csv"), index=False)


if __name__ == "__main__":
    split_data()
