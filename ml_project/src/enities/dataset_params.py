from typing import List

from dataclasses import dataclass


@dataclass
class DatasetConfig:
    dataset_dir: str
    dataset_filename: str
    source_url: str
    column_names: List[str]
    download: bool = False
    random_state: int = 0
