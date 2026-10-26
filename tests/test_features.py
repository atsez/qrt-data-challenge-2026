import numpy as np
import pandas as pd

from src.data import load
from src.features import BASE_FEATURES, RET, build_features


def test_feature_shape_and_names(fake_data_dir):
    x, y, _, _ = load(fake_data_dir)
    F = build_features(x)
    assert F.shape == (len(x), 23)
    assert list(F.columns[:8]) == BASE_FEATURES
    assert not F.columns.duplicated().any()


def test_changing_another_date_does_not_change_todays_features(fake_data_dir):
    """No lookahead: rows of any other date (including the next day) cannot leak into a row."""
    x, *_ = load(fake_data_dir)
    F0 = build_features(x)
    dates = x.TS.unique()
    other = x.TS == dates[1]
    x2 = x.copy()
    x2.loc[other, RET] = x2.loc[other, RET] * -5 + 0.3
    F1 = build_features(x2)
    same = (x.TS != dates[1]).to_numpy()
    pd.testing.assert_frame_equal(F0[same], F1[same])


def test_target_is_never_used(fake_data_dir):
    x, y, ret_next, _ = load(fake_data_dir)
    assert "target" not in x.columns
    F0 = build_features(x)
    F1 = build_features(x.assign(target=np.random.default_rng(1).normal(size=len(x))))
    pd.testing.assert_frame_equal(F0, F1)
