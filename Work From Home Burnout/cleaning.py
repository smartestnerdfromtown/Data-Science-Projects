import pandas as pd
import numpy as np
from typing import (
    Dict, 
    List, 
    Mapping, 
    Tuple, 
    Callable
)

IQR_MULTIPLIER: float = 1.5
ZSCORE_THRESHOLD: float = 3.0

def _validate_column(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        raise KeyError(f"Column '{column}' does not exist in DataFrame.")
    return df[column]


def standardize_values(
    df: pd.DataFrame,
    column: str,
    mapping: Mapping
) -> pd.DataFrame:
    """
    Return a copy of `df` with mapped values in the specified column.

    Parameters
    ----------
    df : pd.DataFrame
        Source dataframe.
    column : str
        Column to transform.
    mapping : Mapping
        Dictionary-like object used to map old values to new ones.

    Returns
    -------
    pd.DataFrame
        A new dataframe with mapped values.
    """
    if column not in df.columns:
        raise KeyError(f"Column '{column}' does not exist in DataFrame.")

    result = df.copy()
    result[column] = (
        result[column]
        .map(mapping)
        .fillna(result[column]) # to preserve original values when not in mapping
    )

    return result


def validate_ranges(
    df: pd.DataFrame,
    column: str,
    expected_min: float,
    expected_max: float,
) -> Tuple[bool, float, float]:
    """
    Validate whether a column's min and max match expected values.

    Returns
    -------
    tuple:
        (is_valid, actual_min, actual_max)
    """
    if column not in df.columns:
        raise KeyError(f"Column '{column}' does not exist in DataFrame.")

    actual_min = df[column].min()
    actual_max = df[column].max()

    is_valid = (actual_min == expected_min) and (actual_max == expected_max)

    return is_valid, actual_min, actual_max


def _outliers_iqr(df: pd.DataFrame, column: str) -> pd.DataFrame:
    series = _validate_column(df, column)

    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1

    lower = q1 - IQR_MULTIPLIER * iqr
    upper = q3 + IQR_MULTIPLIER * iqr

    return df[(series < lower) | (series > upper)]


def _outliers_zscore(df: pd.DataFrame, column: str) -> pd.DataFrame:
    series = _validate_column(df, column)

    std = series.std(ddof=0)
    if std == 0:
        return pd.DataFrame(columns=df.columns)

    z_scores = (series - series.mean()) / std
    return df[z_scores.abs() > ZSCORE_THRESHOLD]


OUTLIER_METHODS: Dict[str, Callable[[pd.DataFrame, str], pd.DataFrame]] = {
    "iqr": _outliers_iqr,
    "z_score": _outliers_zscore,
}


def detect_outliers(
    df: pd.DataFrame,
    column: str,
    method: str = "iqr",
) -> pd.DataFrame:
    """
    Detect outliers using the specified method.

    Parameters
    ----------
    method : {"iqr", "z_score"}

    Returns
    -------
    pd.DataFrame
        DataFrame containing only outlier rows.
    """
    if method not in OUTLIER_METHODS:
        raise ValueError(
            f"Unsupported method '{method}'. "
            f"Available methods: {list(OUTLIER_METHODS.keys())}"
        )

    outliers = OUTLIER_METHODS[method](df, column)
    return f"There are {len(outliers)} outliers in {column} column"


if __name__ == "__main__":
    df = pd.read_csv(
        filepath_or_buffer="work_from_home_burnout_dataset.csv"
    )

    day_mapper = {
        "Weekend": 0,
        "Weekday": 1
    }
    df_copy = standardize_values(
        df=df, column="day_type", mapping=day_mapper
    )

    burnout_mapper = {
        "Low": 0,
        "Medium": 1,
        "High": 2
    }
    df_copy = standardize_values(
        df=df_copy, column="burnout_risk", mapping=burnout_mapper
    )

    print("work_hours: ", 
          validate_ranges(
              df=df_copy, 
              column="work_hours", 
              expected_min=0, 
              expected_max=24
    ))
    
    print("sleep_hours: ", 
          validate_ranges(
              df=df_copy, 
              column="sleep_hours", 
              expected_min=0, 
              expected_max=24
    ))

    print("task_completion_rate: ", 
          validate_ranges(
              df=df_copy, 
              column="task_completion_rate", 
              expected_min=0, 
              expected_max=100
    ))
    
    screen_time_hours = detect_outliers(
        df=df_copy, column="screen_time_hours", method="z_score"
    )
    task_completion_rate = detect_outliers(
        df=df_copy, column="task_completion_rate", method="iqr"
    )

    print(screen_time_hours)
    print(task_completion_rate)

    df_copy.to_csv(
        path_or_buf="burnout_cleaned.csv",
        index=False
    )
