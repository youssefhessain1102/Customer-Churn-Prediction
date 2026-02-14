from typing import Any, Dict, Tuple, Optional

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.model_selection import GridSearchCV

from utils.logging.logger import get_logger
from utils.exceptions.exceptions import ConfigError, ModelError, ml_step


logger = get_logger("model_selection")


@ml_step(logger)
def tune_classifier(
    estimator: BaseEstimator,
    param_grid: Dict[str, list],
    x_train: np.ndarray,
    y_train: np.ndarray,
    cv: int = 5,
    scoring: str = "accuracy",
    n_jobs: Optional[int] = -1,
) -> Tuple[BaseEstimator, Dict[str, Any]]:
    """
    Run hyperparameter tuning for a classification estimator using GridSearchCV.

    Parameters
    ----------
    estimator : BaseEstimator
        Scikit-learn compatible classifier instance (e.g., RandomForestClassifier()).
    param_grid : Dict[str, list]
        Dictionary of hyperparameters to search over.
    X_train : np.ndarray
        Training features (already preprocessed).
    y_train : np.ndarray
        Training target.
    cv : int, optional
        Number of cross-validation folds, by default 5.
    scoring : str, optional
        Scoring metric name for GridSearchCV, by default "accuracy".
    n_jobs : Optional[int], optional
        Number of parallel jobs, by default -1 (all cores).

    Returns
    -------
    Tuple[BaseEstimator, Dict[str, Any]]
        Best estimator and a dictionary with tuning details.
    """
    if not param_grid:
        raise ConfigError("param_grid must not be empty for hyperparameter tuning.")

    try:
        search = GridSearchCV(
            estimator=estimator,
            param_grid=param_grid,
            cv=cv,
            scoring=scoring,
            n_jobs=n_jobs,
            refit=True,
        )
        search.fit(x_train, y_train)
    except Exception as exc:  # noqa: BLE001
        raise ModelError(f"Hyperparameter tuning failed: {exc}") from exc

    logger.info(
        "Best params: %s | Best %s: %.4f",
        search.best_params_,
        scoring,
        search.best_score_,
    )

    info: Dict[str, Any] = {
        "best_params": search.best_params_,
        "best_score": float(search.best_score_),
        "cv_results": {
            "params": search.cv_results_.get("params"),
            "mean_test_score": search.cv_results_.get("mean_test_score").tolist(),
        },
    }

    return search.best_estimator_, info
