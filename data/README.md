# Data

The challenge data is not included. It belongs to the challenge provider and cannot be redistributed.

1. Register for "Asset Allocation Performance forecasting by QRT" on Challenge Data:
   https://challengedata.ens.fr/challenges/167
2. Download the three files from the challenge page and put them in this folder:

| File | Contents |
|---|---|
| `X_train_9xQjqvZ.csv` | training features, 527,073 rows |
| `y_train_Ppwhaz8.csv` | next-day return for each training row (`ROW_ID`, `target`) |
| `X_test_1zTtEnD.csv` | test features, 31,870 rows |

If your download has different suffixes, the loader falls back to any file starting with
`X_train`, `y_train` or `X_test`.
