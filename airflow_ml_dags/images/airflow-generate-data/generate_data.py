import os

import click
import pandas as pd
from sklearn.datasets import make_classification


DATA_PATH = "data.csv"
TARGETS_PATH = "target.csv"


@click.command("generate_data")
@click.option("--output-dir", type=click.Path())
def generate_data(output_dir: str) -> None:
    data, targets = make_classification(n_features=10, random_state=0)
    os.makedirs(output_dir, exist_ok=True)
    data = pd.DataFrame(data)
    data.columns = [
            "Systemic Illness",
            "Rectal Pain",
            "Sore Throat",
            "Penile Oedema",
            "Oral Lesions",
            "Solitary Lesion",
            "Swollen Tonsils",
            "HIV Infection",
            "Sexually Transmitted Infection",
            "Patient_ID",
    ]
    targets = pd.DataFrame(
        targets,
        columns=["MonkeyPox"]
    )
    data.to_csv(os.path.join(output_dir, DATA_PATH), index=False)
    targets.to_csv(os.path.join(output_dir, TARGETS_PATH), index=False)


if __name__ == '__main__':
    generate_data()
