import numpy as np
from typing import List, Dict, Any, Optional

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import RandomizedSearchCV


def create_pipeline(
    model,
    num_features: List[str],
    cat_features: List[str],
    *,
    num_imputer_strategy: str = "median",
    cat_imputer_strategy: str = "most_frequent",
    scaler: Optional[StandardScaler] = None,
    encoder: Optional[OneHotEncoder] = None,
    remainder: str = "drop",
    sparse_threshold: float = 0.3
) -> Pipeline:
    """
    Create a preprocessing + modeling pipeline for tabular data.

    This function builds a sklearn Pipeline consisting of:
    - Numerical preprocessing (imputation + scaling)
    - Categorical preprocessing (imputation + encoding)
    - Final estimator (model)

    Parameters
    ----------
    model : estimator object
        Any sklearn-compatible estimator implementing fit/predict.

    num_features : List[str]
        List of numerical feature column names.

    cat_features : List[str]
        List of categorical feature column names.

    num_imputer_strategy : str, default="median"
        Strategy for imputing missing numerical values.
        Options: {"mean", "median", "most_frequent", "constant"}.

    cat_imputer_strategy : str, default="most_frequent"
        Strategy for imputing missing categorical values.

    scaler : Optional[StandardScaler], default=None
        Scaler for numerical features. If None, StandardScaler is used.

    encoder : Optional[OneHotEncoder], default=None
        Encoder for categorical features. If None, OneHotEncoder is used.

    remainder : str, default="drop"
        What to do with remaining columns not specified in transformers.
        Options: {"drop", "passthrough"}.

    sparse_threshold : float, default=0.3
        Threshold for sparse matrix output in ColumnTransformer.

    Returns
    -------
    Pipeline
        A fully constructed sklearn Pipeline with preprocessing and model.
    """

    scaler = scaler or StandardScaler()
    encoder = encoder or OneHotEncoder(handle_unknown="ignore")

    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy=num_imputer_strategy)),
        ("scaler", scaler)
    ])

    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy=cat_imputer_strategy)),
        ("encoder", encoder)
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, num_features),
            ("cat", cat_pipeline, cat_features)
        ],
        remainder=remainder,
        sparse_threshold=sparse_threshold
    )

    pipeline = Pipeline([
        ("preprocessing", preprocessor),
        ("model", model)
    ])

    return pipeline


def tuning(
    param_dist: Dict[str, Any],
    pipeline: Pipeline,
    *,
    scoring: str = "accuracy",
    n_iter: int = 20,
    cv: int = 5,
    n_jobs: int = -1,
    random_state: int = 777,
    verbose: int = 1,
) -> RandomizedSearchCV:
    """
    Perform hyperparameter tuning using RandomizedSearchCV.

    This function wraps sklearn's RandomizedSearchCV with extended configurability.

    Parameters
    ----------
    param_dist : Dict[str, Any]
        Dictionary of parameter distributions.
        Keys must follow pipeline naming convention (e.g., 'model__param').

    pipeline : Pipeline
        The sklearn Pipeline to tune.

    scoring : str, default="accuracy"
        Evaluation metric for model selection.

    n_iter : int, default=20
        Number of parameter settings sampled.

    cv : int, default=5
        Number of cross-validation folds.

    n_jobs : int, default=-1
        Number of parallel jobs.

    random_state : int, default=777
        Random seed for reproducibility.

    verbose : int, default=1
        Controls verbosity of output.

    Returns
    -------
    RandomizedSearchCV
        Configured RandomizedSearchCV object (not yet fitted).
    """

    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_dist,
        n_iter=n_iter,
        cv=cv,
        scoring=scoring,
        n_jobs=n_jobs,
        random_state=random_state,
        verbose=verbose,
    )

    return search