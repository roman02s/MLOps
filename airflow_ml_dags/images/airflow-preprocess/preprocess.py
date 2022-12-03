import os

import click
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


@click.command("preprocess")
@click.option("--input-dir")
@click.option("--output-dir")
def preprocess_data(input_dir: str, output_dir: str) -> None:
    data = pd.read_csv(os.path.join(input_dir, "data.csv"))
    target = pd.read_csv(os.path.join(input_dir, "target.csv"))
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)
    data_scaled = pd.DataFrame(data=data_scaled, columns=data.columns)
    os.makedirs(output_dir, exist_ok=True)
    data_scaled.to_csv(os.path.join(output_dir, "train_data.csv"), index=False)
    target.to_csv(os.path.join(output_dir, "target.csv"), index=False)


if __name__ == "__main__":
    preprocess_data()
