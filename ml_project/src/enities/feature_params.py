from dataclasses import dataclass, field
from typing import List, Optional


@dataclass()
class FeatureParams:
    categorical_features: Optional[List[str]] = field(default=None)
    numerical_features: Optional[List[str]] = field(default=None)
    features_to_drop: Optional[List[str]] = field(default=None)
    target_col: Optional[List[str]] = field(default=None)
    use_log_trick: bool = field(default=False)
