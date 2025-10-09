# trainer.py
from pathlib import Path
import json, numpy as np
from typing import Dict, Any
from registry import TASKS

class GenericTrainer:
    def __init__(self, task_name: str, cfg: Dict[str, Any]):
        self.spec = TASKS[task_name]
        self.cfg = cfg
        # isolate artifacts per task + objective (avoid accidental overwrite)
        base = Path(cfg["artifacts_dir"])
        self.artifacts_dir = base / f'{task_name}{self.spec.artifact_suffix}'
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.model_path = self.artifacts_dir / "model.bin"
        self.meta_path  = self.artifacts_dir / "meta.json"

    def make_dataset(self, df):
        # feature building is shared; allow opt-in per-task feature set
        if self.spec.feature_set:
            X = df[self.spec.feature_set]
        else:
            X = df[self.cfg["features"]]   # from global config
        y = df.apply(self.spec.build_y, axis=1).astype(float).values
        return X, y

    def _train_single_model(self, X, y):
        # use your preferred lib; here's LightGBM pseudocode
        import lightgbm as lgb
        params = dict(self.cfg["lgbm_params"])
        params["objective"] = self.spec.objective
        dtrain = lgb.Dataset(X, label=y)
        evals_result = {}
        booster = lgb.train(
            params,
            dtrain,
            num_boost_round=self.cfg.get("num_boost_round", 1000),
            valid_sets=[dtrain],
            valid_names=["train"],
            early_stopping_rounds=self.cfg.get("early_stopping", 50),
            evals_result=evals_result,
            verbose_eval=False
        )
        booster.save_model(str(self.model_path))
        with open(self.meta_path, "w") as f:
            json.dump({
                "task": self.spec.name,
                "objective": self.spec.objective,
                "metrics": self.spec.eval_metrics,
                "features": self.cfg["features"],
                "best_iteration": booster.best_iteration
            }, f, indent=2)
        return booster

    def _train_two_stage(self, X, y):
        # Stage 1: classifier P(y>0)
        import lightgbm as lgb
        y_pos = (y > 0).astype(int)
        clf = lgb.LGBMClassifier(**self.cfg["clf_params"])
        clf.fit(X, y_pos)
        # Stage 2: count model on y>0 (truncated)
        mask = y > 0
        if mask.sum() == 0:
            raise ValueError("No positive examples for stage-2.")
        reg = lgb.LGBMRegressor(**self.cfg["reg_params"])  # can be 'poisson' or tweedie
        reg.set_params(objective=self.spec.objective)
        reg.fit(X[mask], y[mask])
        # Persist both
        import joblib
        joblib.dump({"clf": clf, "reg": reg}, self.model_path)
        with open(self.meta_path, "w") as f:
            json.dump({"task": self.spec.name, "two_stage": True}, f, indent=2)
        return clf, reg

    def fit(self, df):
        X, y = self.make_dataset(df)
        if self.spec.two_stage:
            return self._train_two_stage(X.values, y)
        return self._train_single_model(X.values, y)

    def predict(self, df):
        # returns expected count predictions
        X, _ = self.make_dataset(df)
        if self.spec.two_stage:
            import joblib
            bundle = joblib.load(self.model_path)
            p_pos = bundle["clf"].predict_proba(X)[:, 1]
            mu = bundle["reg"].predict(X)  # expected conditional count (given >0)
            yhat = p_pos * np.maximum(mu, 0.0)
        else:
            import lightgbm as lgb
            booster = lgb.Booster(model_file=str(self.model_path))
            yhat = booster.predict(X)
        if self.spec.output_transform:
            yhat = np.vectorize(self.spec.output_transform)(yhat)
        return yhat
