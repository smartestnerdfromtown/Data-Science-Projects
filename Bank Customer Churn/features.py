import pandas as pd
import numpy as np

from typing import Tuple, List, Optional, Dict, Any
from sklearn.model_selection import train_test_split


def extract_features(
    df: pd.DataFrame,
    target_column: Optional[str] = None,
    exclude: Optional[List[str]] = None
) -> Tuple[List[str], List[str]]:
    """
    Extract numerical and categorical feature names from a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing features and optionally the target.
    target_column : Optional[str], default=None
        Name of the target column to exclude from feature extraction.
    exclude : Optional[List[str]], default=None
        Additional columns to exclude.

    Returns
    -------
    Tuple[List[str], List[str]]
        A tuple containing:
        - numerical feature names
        - categorical feature names

    Notes
    -----
    - Numerical features are inferred using numpy number dtypes.
    - Categorical features include object and string dtypes.
    - This function does not mutate the original dataframe.
    """

    exclude = set(exclude or [])
    if target_column:
        exclude.add(target_column)

    feature_df = df.drop(columns=list(exclude), errors="ignore")

    num_features = feature_df.select_dtypes(include=np.number).columns.tolist()
    cat_features = feature_df.select_dtypes(include=["object", "string", "category"]).columns.tolist()

    return num_features, cat_features


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 777,
    stratify: bool = True,
    shuffle: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split dataset into train and test sets with optional stratification.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Target vector.
    test_size : float, default=0.2
        Proportion of the dataset to include in the test split.
    random_state : int, default=777
        Controls reproducibility.
    stratify : bool, default=True
        Whether to apply stratified splitting (recommended for classification).
    shuffle : bool, default=True
        Whether to shuffle data before splitting.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]
        X_train, X_test, y_train, y_test

    Raises
    ------
    ValueError
        If stratification is requested but y is not suitable.
    """

    stratify_arg = y if stratify else None

    if stratify and y.nunique() < 2:
        raise ValueError("Stratified split requires at least 2 classes in target.")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_arg,
        shuffle=shuffle
    )

    return X_train, X_test, y_train, y_test


def prepare_data(
    file_path: str,
    target_column: str,
    columns_to_drop: Optional[List[str]] = None,
    dtype_map: Optional[Dict[str, Any]] = None,
    parse_dates: Optional[List[str]] = None,
    dropna: bool = False
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """
    Load and prepare dataset for machine learning.

    Parameters
    ----------
    file_path : str
        Path to the dataset file.
    target_column : str
        Name of the target variable.
    columns_to_drop : Optional[List[str]], default=None
        Columns to remove from the dataset.
    dtype_map : Optional[Dict[str, Any]], default=None
        Dictionary specifying column dtypes for optimized loading.
    parse_dates : Optional[List[str]], default=None
        Columns to parse as datetime.
    dropna : bool, default=False
        Whether to drop rows with missing values.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.Series]
        - df_pipeline : cleaned dataframe (after dropping columns)
        - X : feature matrix
        - y : target vector

    Raises
    ------
    FileNotFoundError
        If the file path is invalid.
    KeyError
        If target_column is not found in the dataset.

    Notes
    -----
    - This function does not perform feature engineering.
    - Avoids data leakage by separating X and y early.
    """

    df = pd.read_csv(
        filepath_or_buffer=file_path,
        dtype=dtype_map,
        parse_dates=parse_dates
    )

    if target_column not in df.columns:
        raise KeyError(f"Target column '{target_column}' not found in dataset.")

    columns_to_drop = columns_to_drop or []

    df_pipeline = df.drop(columns=columns_to_drop, errors="ignore")

    if dropna:
        df_pipeline = df_pipeline.dropna()

    X = df.drop(columns=[target_column])
    y = df[target_column]

    return df_pipeline, X, y