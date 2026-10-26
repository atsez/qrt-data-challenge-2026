"""Final model and baselines."""
import numpy as np
import lightgbm as lgb
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEEDS = (42, 7, 123)

# hand-tuned, conservative trees
CFG_SMALL_TREES = dict(n_estimators=1000, learning_rate=0.01, num_leaves=15, min_child_samples=100,
                       subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=0.1, verbose=-1)
# heavily regularised: a leaf needs >= 3000 rows (about 11 dates), which stops trees
# from memorising individual days through the same-date features
CFG_REGULARISED = dict(n_estimators=800, learning_rate=0.02, num_leaves=31, min_child_samples=3000,
                       subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=5,
                       extra_trees=True, verbose=-1)


def single_lgbm(scale=1.0, seed=42):
    cfg = dict(CFG_SMALL_TREES)
    cfg["n_estimators"] = max(10, int(cfg["n_estimators"] * scale))
    return lgb.LGBMClassifier(**cfg, random_state=seed)


def logistic():
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         LogisticRegression(C=0.01, max_iter=1000))


class FinalModel:
    """Average of 2 LightGBM configurations x 3 seeds."""

    def __init__(self, scale=1.0, n_jobs=None):
        self.models = []
        for cfg in (CFG_SMALL_TREES, CFG_REGULARISED):
            for s in SEEDS:
                c = dict(cfg)
                c["n_estimators"] = max(10, int(c["n_estimators"] * scale))
                if n_jobs:
                    c["n_jobs"] = n_jobs
                self.models.append(lgb.LGBMClassifier(**c, random_state=s))

    def fit(self, X, y):
        for m in self.models:
            m.fit(X, y)
        return self

    def predict_proba(self, X):
        return np.mean([m.predict_proba(X)[:, 1] for m in self.models], axis=0)

    def predict(self, X):
        return (self.predict_proba(X) > 0.5).astype(int)
