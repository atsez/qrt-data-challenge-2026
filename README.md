# QRT Data Challenge 2026: Asset Allocation Performance Forecasting

Predicting whether a systematic portfolio ("allocation") goes up or down the next day from its last 20 days of returns and volume. Individual entry. Final public leaderboard accuracy **0.531** against the official benchmark of **0.5079**.

## The finding

- **The signal is cross-sectional.** What the whole market and the allocation's group did on the same day matters far more than the allocation's own history. Adding 15 features, mostly same-date market and group context, takes LightGBM from 0.5099 to 0.5191 and logistic regression from 0.5060 to 0.5206 in purged cross-validation. The top seven features by gain are all market or group features.
- **The model barely matters; the features and the validation do.** On the same 23 features, a regularised logistic regression (0.5206) matches the final LightGBM ensemble (0.5207). The ensemble earns its place by being stable across seeds, not by finding extra signal.
- **The data leaks, and the validation has to account for it.** Consecutive rows of an allocation share 19 of their 20 days, so a naive random-date split leaks targets (0.5227 vs 0.5191 purged). I rebuilt the shuffled timeline from the overlapping windows: on train, the next day's row predicts the target sign 88.4% of the time. The test dates have no neighbours in the data at all (0.000 linked), so this cannot be used for prediction, but purging around it is why the cross-validated 0.5207 lands on the leaderboard's 0.5197.

## Results

| Model | Features | Purged cross-validation accuracy | Public leaderboard |
|---|---|---|---|
| Always predict up | - | 0.5072 | - |
| QRT benchmark (LightGBM) | 53 | - | 0.5079 |
| Logistic regression | 8 base | 0.5060 | - |
| LightGBM, single model | 8 base | 0.5099 | - |
| Logistic regression | 23 | 0.5206 | - |
| LightGBM, single model | 23 | 0.5191 | - |
| **Final: LightGBM, 2 configs x 3 seeds** | **23** | **0.5207** | **0.5197** |

Cross-validation numbers come from `results/results.md`, written by the pipeline. The leaderboard numbers are the official scores. Train accuracy of the final model is 0.5765, a 0.056 gap to validation.

For scale: even knowing every date's majority direction in advance would only give 0.5716 on train. Daily returns of these portfolios are mostly noise, and about two points above a coin flip is a real edge.

## Method

**Data.** 527,073 training rows (2,522 dates, 278 allocations, 4 groups) and 31,870 test rows (120 dates). Each row has RET_1..RET_20 (RET_1 = most recent day), SIGNED_VOLUME_1..20, median daily turnover and an anonymised group. Date labels are shuffled.

**Features** (`src/features.py`). Each row uses only its own 20-day window and other rows on the same date.

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

**Also tried.** These are listed without scores because they were evaluated on a simple random 80/20 split before I found the leakage, so the numbers are not comparable:
- random forest
- random forest + LightGBM ensemble
- per-group LightGBM models
- sklearn MLP
- PyTorch neural network
- LightGBM on the 40 raw columns

A per-allocation shrinkage model (each allocation's own coefficients pulled towards its group) scored higher in purged cross-validation but lower on the public leaderboard, so the final model stays global.

## What I'd do next

- **Hyperparameters were tuned by hand.** Next: a nested search inside the purged folds.
- **The 0.056 train/validation gap is large.** Next: separate the date-level part (market direction, about 2,500 effective samples) from the within-date ranking part, and regularise each at its own scale.
- **The public test set is 120 dates, and accuracy is highly correlated within a date.** Small leaderboard differences are mostly noise. Next: report a bootstrap over dates instead of single scores.
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

Fixed seeds throughout. `submissions/submission.csv` from a full run is the submission that scored 0.5197.

## Layout

```
pipeline.py          end-to-end run: features, timeline, purged cross-validation, importance, submission, report
src/data.py          loading and the y/target join
src/features.py      the 23 features
src/timeline.py      rebuilding date order from overlapping windows; leakage diagnostics
src/validation.py    purged, blocked cross-validation
src/models.py        final ensemble and baselines
tests/               synthetic-data tests (no lookahead, end-to-end run, output shape and range)
results/results.md   generated by pipeline.py
data/README.md       where to get the data
```
