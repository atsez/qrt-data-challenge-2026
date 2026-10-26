import numpy as np
import pandas as pd

import pipeline
from src.data import load
from src.timeline import reconstruct_order


def test_timeline_recovers_true_order(fake_data_dir):
    x, *_ = load(fake_data_dir)
    order, n_seg = reconstruct_order(x)
    assert len(order) == x.TS.nunique()
    # synthetic labels were shuffled; check consecutive positions are true consecutive days
    # by matching windows: day t's RET_1..19 should equal day t+1's RET_2..20 (up to small noise)
    a = x[x.ALLOCATION == "ALLOCATION_00"].set_index("TS")
    seq = order.sort_values().index
    diffs = [np.abs(a.loc[seq[i], [f"RET_{k}" for k in range(1, 20)]].to_numpy()
                    - a.loc[seq[i + 1], [f"RET_{k}" for k in range(2, 21)]].to_numpy()).max()
             for i in range(len(seq) - 1)]
    assert np.mean(np.array(diffs) < 1e-3) > 0.95


def test_pipeline_end_to_end(fake_data_dir, tmp_path):
    out = pipeline.run(fake_data_dir, tmp_path / "results", tmp_path / "subs", scale=0.02, log=lambda *_: None)
    sub = pd.read_csv(tmp_path / "subs" / "submission.csv")
    _, _, _, x_te = load(fake_data_dir)
    assert len(sub) == len(x_te)
    assert list(sub.columns) == ["ROW_ID", "TARGET"]
    assert set(sub.TARGET.unique()) <= {0, 1}
    assert (sub.ROW_ID.to_numpy() == x_te.ROW_ID.to_numpy()).all()
    assert (tmp_path / "results" / "results.md").exists()
    for r in out["rows"]:
        assert 0.0 <= r["mean"] <= 1.0


def test_probabilities_in_range(fake_data_dir):
    from src.features import build_features
    from src.models import FinalModel
    x, y, _, x_te = load(fake_data_dir)
    F = build_features(x)
    p = FinalModel(scale=0.02).fit(F, y).predict_proba(build_features(x_te)[F.columns])
    assert p.shape == (len(x_te),)
    assert np.all((p >= 0) & (p <= 1))
