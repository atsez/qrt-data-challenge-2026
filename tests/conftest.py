import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

RET = [f"RET_{i}" for i in range(1, 21)]
VOL = [f"SIGNED_VOLUME_{i}" for i in range(1, 21)]


def make_split(n_dates, n_alloc, rng, label_offset, row_offset):
    """Synthetic data shaped like the challenge: each allocation follows one return path,
    each row is the 20-day window ending on that date (RET_1 = most recent day),
    the target is the next day's return, and date labels are shuffled."""
    rows, target = [], []
    labels = rng.permutation(n_dates) + label_offset
    for a in range(n_alloc):
        path = rng.normal(0, 0.01, n_dates + 21)
        vol = rng.normal(0, 1, n_dates + 21)
        for d in range(n_dates):
            hist = path[d:d + 20][::-1] + rng.normal(0, 1e-4, 20)   # small drift, like weight changes
            vhist = vol[d:d + 20][::-1]
            r = {"TS": f"DATE_{labels[d]:04d}", "ALLOCATION": f"ALLOCATION_{a:02d}",
                 "MEDIAN_DAILY_TURNOVER": rng.uniform(0.1, 1.0), "GROUP": a % 4 + 1}
            r.update(dict(zip(RET, hist)))
            r.update(dict(zip(VOL, vhist)))
            rows.append(r)
            target.append(path[d + 20])
    df = pd.DataFrame(rows).sort_values(["TS", "ALLOCATION"]).reset_index(drop=True)
    order = df.index.to_numpy()
    tgt = pd.DataFrame(rows).assign(t=target).set_index(["TS", "ALLOCATION"]).loc[
        list(zip(df.TS, df.ALLOCATION)), "t"].to_numpy()
    df.insert(0, "ROW_ID", np.arange(len(df)) + row_offset)
    holes = rng.choice(len(df), len(df) // 50, replace=False)   # the real data has missing values
    df.loc[holes, "SIGNED_VOLUME_1"] = np.nan
    df.loc[holes[:5], "RET_20"] = np.nan
    return df, tgt


@pytest.fixture(scope="session")
def fake_data_dir(tmp_path_factory):
    rng = np.random.default_rng(0)
    train, tgt = make_split(120, 16, rng, 1, 0)
    test, _ = make_split(30, 16, rng, 500, len(train))   # separate path: isolated, like the real test set
    d = tmp_path_factory.mktemp("data")
    train.to_csv(d / "X_train_9xQjqvZ.csv", index=False)
    pd.DataFrame({"ROW_ID": train.ROW_ID, "target": tgt}).to_csv(d / "y_train_Ppwhaz8.csv", index=False)
    test.to_csv(d / "X_test_1zTtEnD.csv", index=False)
    return d
