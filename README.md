# QRT Data Challenge 2026: Asset Allocation Performance Forecasting

Predicting whether a systematic portfolio ("allocation") goes up or down the next day from its last 20 days of returns and volume. Individual entry. Final public leaderboard accuracy **0.5134** against the official benchmark of **0.5079**.

## The finding

- **The signal is cross-sectional.** What the whole market and the allocation's group did on the same day matters far more than the allocation's own history. Adding 15 features, mostly same-date market and group context, takes LightGBM from 0.5099 to 0.5191 and logistic regression from 0.5060 to 0.5206 in purged cross-validation. The top seven features by gain are all market or group features.
- **The model barely matters; the features and the validation do.** On the same 23 features, a regularised logistic regression (0.5206) matches the final LightGBM ensemble (0.5207). The ensemble earns its place by being stable across seeds, not by finding extra signal.
- **The data leaks, and the validation has to account for it.** Consecutive rows of an allocation share 19 of their 20 days, so a naive random-date split leaks targets (0.5227 vs 0.5191 purged). I rebuilt the shuffled timeline from the overlapping windows: on train, the next day's row predicts the target sign 88.4% of the time. The test dates have no neighbours in the data at all (0.000 linked), so this cannot be used for prediction, but purging around it keeps the cross-validated 0.5207 close to the leaderboard's 0.5134 instead of inflating it.

## Results

| Model | Features | Purged cross-validation accuracy | Public leaderboard |
|---|---|---|---|
| Always predict up | - | 0.5072 | - |
| QRT benchmark (LightGBM) | 53 | - | 0.5079 |
| Logistic regression | 8 base | 0.5060 | - |
| LightGBM, single model | 8 base | 0.5099 | - |
| Logistic regression | 23 | 0.5206 | - |
| LightGBM, single model | 23 | 0.5191 | - |
| **Final: LightGBM, 2 configs x 3 seeds** | **23** | **0.5207** | **0.5134** |

Cross-validation numbers come from `results/results.md`, written by the pipeline. The leaderboard numbers are the official scores. Train accuracy of the final
