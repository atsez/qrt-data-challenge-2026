# QRT Data Challenge 2026: Asset Allocation Performance Forecasting

Predicting whether a systematic portfolio ("allocation") goes up or down the next day from its last 20 days of returns and volume. Individual entry in the QRT x Ecole Polytechnique data challenge.

## The finding

- **The signal is cross-sectional.** What the whole market and the allocation's group did on the same day matters far more than the allocation's own history. Adding same-date market and group context was the single biggest improvement, and the most important features by gain are all market or group features.
- **The model barely matters; the features and the validation do.** On the same features, a regularised logistic regression performs about as well as the final LightGBM ensemble. The ensemble earns its place by being stable across seeds, not by finding extra signal.
- **The data leaks, and the validation has to account for it.** Consecutive rows of an allocation share 19 of their 20 days, so a naive random-date split leaks targets and overstates performance. I rebuilt the shuffled timeline from the overlapping windows and found that, on train, the next day's row reveals most of the target. The test dates have no neighbours in the data, so this cannot be used for prediction, but purging around it is what makes the local validation match the leaderboard.

## Method

**Data.** 527,073 training rows (2,522 dates, 278 allocations, 4 groups) and 31,870 test rows (120 dates). Each row has RET_1..RET_20 (RET_1 = most recent day), SIGNED_VOLUME_1..20, median daily turnover and an anonymised group. Date labels are shuffled.

**Features** (`src/features.py`). 23 features. Each row uses only its own 20-day window and other rows on the same date.

*Per allocation:*
- `mean_ret_20d`, `std_ret_20d`: average return and volatility, the allocation's recent drift and risk.
- `momentum`: last 5 days minus 20-day mean, whether it is accelerating or fading.
- `positive_count_20d`: up days out of 20, a robust trend measure.
- `mean_volume_20d`, `std_volume_20d`: level and stability of signed volume (net buying pressure).
- `MEDIAN_DAILY_TURNOVER`: how fast the portfolio rebalances; fast strategies behave differently.
- `GROUP`: strategy family.
- `RET_1`, `mean_ret_5d`: the most recent move and the last week.

*Same-date context:*
- `mkt_*`, `grp_*`: market-wide and group-wide average of RET_1, 5-day and 20-day returns. Allocations share assets, so the day's common move dominates the next day.
- `mkt_disp_RET_1`: cross-sectional dispersion, a proxy for how much stock picking matters that day.
- `rel_*`, `rank_*`: the allocation's return minus the market, and its percentile rank, which separates its own move from the market's.

**Model** (`src/models.py`). Two hand-tuned LightGBM configurations, each trained with 3 seeds and averaged:
- conservative trees: 1,000 trees, learning rate 0.01, 15 leaves
- a heavily regularised version: at least 3,000 rows per leaf, which stops trees memorising single days through the same-date features

**Validation** (`src/timeline.py`, `src/validation.py`). The shuffled date labels are put back in order by matching each row's window against the same allocation's later rows (nearest neighbour on shifted windows, then a majority vote across allocations). This gives 28 contiguous segments. Cross-validation then holds out blocks of 40 consecutive dates and drops every training date within 20 days of a validation date. No training row shares a single day, or its target day, with a validation row.

**Leakage checks.**
- Features never see the target or another date. `tests/test_features.py` changes another date's rows and the target, and checks today's features are unchanged.
- The split is purged by time, as described above.
- The pipeline compares the purged split against a naive random-date split to measure how much the leak would have inflated results.

**Also tried:**
- random forest
- random forest + LightGBM ensemble
- per-group LightGBM models
- sklearn MLP
- PyTorch neural network
- LightGBM on the 40 raw columns
- a per-allocation shrinkage model (each allocation's own coefficients pulled towards its group)

The per-allocation model looked better in validation but did worse out of sample, so the final model stays global.

## What I'd do next

- **Hyperparameters were tuned by hand.** Next: a nested search inside the purged folds.
- **The gap between training and validation performance is large.** Next: separate the date-level part (market direction, about 2,500 effective samples) from the within-date ranking part, and regularise each at its own scale.
- **The public test set is 120 dates, and accuracy is highly correlated within a date,** so small leaderboard differences are mostly noise. Next: report a bootstrap over dates instead of single scores.
- **The order of the 28 segments is unknown,** so validation cannot be strictly forward-in-time. Next: date the segments from overlaps at longer gaps, then test with walk-forward splits and time-decay weights. That should also explain why the per-allocation model failed out of sample.

## Reproduce

```bash
git clone https://github.com/atsez/qrt-data-challenge-2026.git
cd qrt-data-challenge-2026
pip install -r requirements.txt
# put the three challenge CSVs in data/ (see data/README.md)
python pipeline.py          # about 15 minutes; writes results/results.md and submissions/submission.csv
python pipeline.py --fast   # quick check with 10x smaller models
pytest -q                   # synthetic-data tests, no challenge data needed
```

Fixed seeds throughout, so a full run reproduces the final submission exactly.

## Layout

```
pipeline.py          end-to-end run: features, timeline, purged cross-validation, importance, submission, report
src/data.py          loading and the y/target join
src/features.py      the 23 features
src/timeline.py      rebuilding date order from overlapping windows; leakage diagnostics
src/validation.py    purged, blocked cross-validation
src/models.py        final ensemble and baselines
tests/               synthetic-data tests (no lookahead, end-to-end run, output shape and range)
data/README.md       where to get the data
```
