"""Rebuild the time order of the shuffled date labels.

Each row's return history is computed with that day's portfolio weights, so the row for
day t+L of an allocation has RET_{L+1..20} almost equal to the row for day t's RET_{1..20-L}.
For every row we find the nearest such shifted window within the same allocation, then each
date votes for the date most of its allocations point to. Strong votes become links,
links become chains, and chains are stitched through small gaps.
"""
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from .features import RET


def nearest_later_rows(df: pd.DataFrame, gap: int) -> np.ndarray:
    """For each row, the index of the row (same allocation) whose window matches `gap` days later."""
    R = df[RET].fillna(0).to_numpy()
    out = np.zeros(len(df), dtype=int)
    for _, idx in df.groupby("ALLOCATION").indices.items():
        if len(idx) < 3:
            out[idx] = idx
            continue
        A, B = R[idx, :20 - gap], R[idx, gap:]
        _, nn = NearestNeighbors(n_neighbors=2).fit(B).kneighbors(A)
        own = np.arange(len(idx))
        out[idx] = idx[np.where(nn[:, 0] == own, nn[:, 1], nn[:, 0])]
    return out


def date_votes(df: pd.DataFrame, gap: int) -> pd.DataFrame:
    """Per date: the most common matched date and the share of allocations that agree."""
    j = nearest_later_rows(df, gap)
    d = pd.DataFrame({"src": df["TS"].to_numpy(), "dst": df["TS"].to_numpy()[j]})
    g = d.groupby("src")["dst"]
    return pd.DataFrame({
        "top": g.agg(lambda s: s.value_counts().index[0]),
        "share": g.agg(lambda s: s.value_counts(normalize=True).iloc[0]),
    })


def reconstruct_order(df: pd.DataFrame, gaps=(1, 2, 3, 4, 5, 6, 8, 10), min_share=0.3):
    """Return (Series date -> position, number of separate segments)."""
    v1 = date_votes(df, 1)
    nxt, prev = {}, {}
    for src, r in v1[v1.share > min_share].sort_values("share", ascending=False).iterrows():
        dst = r["top"]
        if src in nxt or dst in prev or src == dst:
            continue
        x = dst
        while x in nxt:
            x = nxt[x]
        if x == src:
            continue
        nxt[src], prev[dst] = dst, src
    chains = []
    for s in [d for d in v1.index if d not in prev]:
        ch = [s]
        while ch[-1] in nxt:
            ch.append(nxt[ch[-1]])
        chains.append(ch)

    # stitch chain ends to chain starts using matches at larger gaps
    start_of = {ch[0]: i for i, ch in enumerate(chains)}
    links = []
    for gap in gaps[1:]:
        v = date_votes(df, gap)
        for i, ch in enumerate(chains):
            for k in range(min(gap, len(ch))):
                d = ch[-1 - k]
                if d in v.index and v.at[d, "share"] > 0.4 and v.at[d, "top"] in start_of:
                    j = start_of[v.at[d, "top"]]
                    if j != i:
                        links.append((v.at[d, "share"], i, j))
    links.sort(reverse=True)
    cn, cp = {}, {}
    for _, i, j in links:
        if i in cn or j in cp:
            continue
        x = j
        while x in cn:
            x = cn[x]
        if x == i:
            continue
        cn[i], cp[j] = j, i
    order, n_seg = [], 0
    for i in range(len(chains)):
        if i in cp:
            continue
        n_seg += 1
        x = i
        while True:
            order += chains[x]
            if x not in cn:
                break
            x = cn[x]
    return pd.Series(np.arange(len(order)), index=order), n_seg


def link_rates(train: pd.DataFrame, test: pd.DataFrame, gap: int = 1) -> dict:
    """Share of train and test dates that have a credible neighbour `gap` days later
    or earlier anywhere in train + test."""
    pool = pd.concat([train.assign(_test=False), test.assign(_test=True)], ignore_index=True)
    is_test = pool.groupby("TS")["_test"].first()
    out = {}
    for name, frame in [("later", pool), ("earlier", None)]:
        if frame is None:  # reverse the windows to look backwards
            frame = pool.copy()
            frame[RET] = pool[RET[::-1]].to_numpy()
        v = date_votes(frame, gap)
        linked = v["share"] > 0.5
        out[name] = (linked[~is_test.reindex(v.index)].mean(), linked[is_test.reindex(v.index)].mean())
    return out


def next_day_sign_agreement(train: pd.DataFrame, y: np.ndarray) -> float:
    """How often sign(next day's row RET_1) equals this row's target sign, on train."""
    j = nearest_later_rows(train, 1)
    return float(((train["RET_1"].fillna(0).to_numpy()[j] > 0).astype(int) == y).mean())
