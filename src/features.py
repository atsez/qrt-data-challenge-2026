"""Feature engineering.

Every feature is computed from one of two sources only:
  1. the row's own 20-day history (RET_1..RET_20, SIGNED_VOLUME_1..20, turnover, group), or
  2. other rows with the SAME date label (cross-sectional market and group context).
Nothing uses the target, other dates, or any future value.
RET_1 is the most recent day, RET_20 the oldest.
"""
import pandas as pd

RET = [f"RET_{i}" for i in range(1, 21)]
VOL = [f"SIGNED_VOLUME_{i}" for i in range(1, 21)]

BASE_FEATURES = [
    "mean_ret_20d", "std_ret_20d", "momentum", "positive_count_20d",
    "mean_volume_20d", "std_volume_20d", "MEDIAN_DAILY_TURNOVER", "GROUP",
]


def base_features(df: pd.DataFrame) -> pd.DataFrame:
    """8 per-allocation features from the row's own history."""
    r, v = df[RET], df[VOL]
    f = pd.DataFrame(index=df.index)
    f["mean_ret_20d"] = r.mean(axis=1)
    f["std_ret_20d"] = r.std(axis=1)
    f["momentum"] = df[RET[:5]].mean(axis=1) - f["mean_ret_20d"]
    f["positive_count_20d"] = (r > 0).sum(axis=1)
    f["mean_volume_20d"] = v.mean(axis=1)
    f["std_volume_20d"] = v.std(axis=1)
    f["MEDIAN_DAILY_TURNOVER"] = df["MEDIAN_DAILY_TURNOVER"]
    f["GROUP"] = df["GROUP"]
    return f


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Final feature set (23 columns): base features + same-date market/group context."""
    f = base_features(df)
    ts = df["TS"]
    date_group = [ts, df["GROUP"]]
    f["mean_ret_5d"] = df[RET[:5]].mean(axis=1)

    # what the whole market and the allocation's group did, on the same date
    for c in ["RET_1", "mean_ret_5d", "mean_ret_20d"]:
        s = df[c] if c in df.columns else f[c]
        f[f"mkt_{c}"] = s.groupby(ts).transform("mean")
        f[f"grp_{c}"] = s.groupby(date_group).transform("mean")
    f["mkt_disp_RET_1"] = df["RET_1"].groupby(ts).transform("std")

    # where the allocation sits relative to the market on that date
    for c in ["RET_1", "mean_ret_5d", "mean_ret_20d"]:
        s = df[c] if c in df.columns else f[c]
        f[f"rel_{c}"] = s - f[f"mkt_{c}"]
        f[f"rank_{c}"] = s.groupby(ts).rank(pct=True)
    f["RET_1"] = df["RET_1"]
    return f
