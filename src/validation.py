"""Purged, blocked cross-validation on the reconstructed timeline.

Validation folds are blocks of consecutive dates. Training excludes every date within
`purge` positions of a validation date, so no training row shares any day of its
20-day window (or its target day) with a validation row.
"""
import numpy as np
import pandas as pd


def purged_folds(ts, order: pd.Series, n_folds=5, block=40, purge=20, seed=0):
    pos = pd.Series(ts).map(order).to_numpy()
    if np.isnan(pos.astype(float)).any():
        raise ValueError("some dates are missing from the reconstructed order")
    pos = pos.astype(int)
    dates = np.sort(order.to_numpy())
    blocks = dates // block
    rng = np.random.default_rng(seed)
    uniq = np.unique(blocks)
    fold_of = dict(zip(uniq, rng.permutation(len(uniq)) % n_folds))
    date_fold = np.array([fold_of[b] for b in blocks])
    for k in range(n_folds):
        val_dates = dates[date_fold == k]
        near = np.zeros(dates.max() + 1, dtype=bool)
        for off in range(-purge, purge + 1):
            idx = val_dates + off
            near[idx[(idx >= 0) & (idx < len(near))]] = True
        yield np.where(~near[pos])[0], np.where(np.isin(pos, val_dates))[0]
