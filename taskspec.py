# taskspec.py
from dataclasses import dataclass
from typing import Callable, Dict, Any, List, Optional

@dataclass
class TaskSpec:
    name: str                        # "points" | "goals" | "assists"
    target_col: str                  # source column in dataset (e.g., 'goals')
    build_y: Callable[[Dict[str, Any]], float]  # row -> numeric label
    objective: str                   # e.g., "poisson", "tweedie", "regression_l2", "binary"
    eval_metrics: List[str]          # e.g., ["rmse","mae"] or ["poisson","tweedie_deviance"]
    output_transform: Optional[Callable[[float], float]] = None  # e.g., exp() if model predicts log-mean
    two_stage: bool = False          # if True: zero-inflated (classifier + truncated count)
    artifact_suffix: str = ""        # e.g., "-poisson" to separate artifacts per task/objective
    feature_set: Optional[List[str]] = None   # allow task-specific feature subsets
