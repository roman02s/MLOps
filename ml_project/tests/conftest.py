import os

import pytest
from typing import List, Optional


@pytest.fixture()
def dataset_path():
    curdir = os.path.dirname(__file__)
    return os.path.join(curdir, "DATA.csv")


@pytest.fixture()
def target_col():
    return "MonkeyPox"


@pytest.fixture()
def categorical_features() -> List[str]:
    return [
        "Systemic Illness",
        "Rectal Pain",
        "Sore Throat",
        "Penile Oedema",
        "Oral Lesions",
        "Solitary Lesion",
        "Swollen Tonsils",
        "HIV Infection",
        "Sexually Transmitted Infection",
    ]


@pytest.fixture
def numerical_features() -> Optional[List[str]]:
    return [
        "",
    ]


@pytest.fixture()
def features_to_drop() -> List[str]:
    return ["Patient_ID"]
