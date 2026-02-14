# Customer Churn Prediction (Classification)

Customer churn prediction project scaffold (no implementation yet).

## Project structure

```
customer_churn_prediction/
  app.py
  README.md
  requirements.txt
  .env.example

  data/
    raw/          # immutable raw data dumps (original extracts)
    processed/    # cleaned/feature-ready datasets for training/inference

  notebooks/      # exploration / EDA notebooks
  reports/        # generated reports, figures, exports
  scripts/        # one-off scripts (e.g., download, preprocess, train)

  config/         # configuration files (YAML/JSON/TOML) as needed
  tests/          # unit/integration tests

  src/            # project package (pipeline code lives here later)
    data/         # data loading / validation / splitting
    features/     # feature engineering
    models/       # training, evaluation, inference
    visualization/# plotting utilities

  utils/          # shared helpers (logging, IO, metrics, etc.)
```

## Data conventions

- Put **original** datasets in `data/raw/`.
- Put **derived** datasets in `data/processed/`.
- Prefer keeping raw data immutable; regenerate processed data via scripts/pipelines.

## Next steps (when you’re ready)

- Add a dataset to `data/raw/` (or a script in `scripts/` to fetch it).
- Define preprocessing + feature engineering pipeline in `src/`.
- Add training/evaluation scripts and tests.

