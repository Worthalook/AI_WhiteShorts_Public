# registry.py
import math
from taskspec import TaskSpec

def clamp_nonneg(x: float) -> float: return max(0.0, x)

TASKS = {
    "points": TaskSpec(
        name="points",
        target_col="points",
        build_y=lambda r: float(r.get("points", 0.0)),
        objective="tweedie",                    # handles overdispersed counts well
        eval_metrics=["tweedie_deviance","mae"],
        output_transform=clamp_nonneg,
        artifact_suffix="-tweedie"
    ),
    "goals": TaskSpec(
        name="goals",
        target_col="goals",
        build_y=lambda r: float(r.get("goals", 0.0)),
        objective="poisson",
        eval_metrics=["poisson","mae"],
        output_transform=clamp_nonneg,
        artifact_suffix="-poisson",
        two_stage=True                          # many zeros → optionally enable two-stage
    ),
    "assists": TaskSpec(
        name="assists",
        target_col="assists",
        build_y=lambda r: float(r.get("assists", 0.0)),
        objective="poisson",
        eval_metrics=["poisson","mae"],
        output_transform=clamp_nonneg,
        artifact_suffix="-poisson",
        two_stage=True
    ),
}
