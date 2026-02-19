import pandas as pd
import numpy as np
from typing import Dict, List, Mapping

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
    column_name: str,
    min: float,
    max: float
    ) -> bool:

    if df[column_name].min() == min and df[column_name].max() == max:
        return True
    return (df[column_name].min(), df[column_name].max())

def define_iqr(
    df: pd.DataFrame,
    column_name: str    
    ) -> pd.DataFrame:

    Q1 = df[column_name].quantile(0.25)
    Q3 = df[column_name].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = df[
        (df[column_name] < lower_bound) | (df[column_name] > upper_bound)
    ]

    return outliers

def define_zscore(
    df: pd.DataFrame,
    column_name: str    
    ) -> pd.DataFrame:
    
    df_copy = df.copy(deep=True)

    mean = np.mean(df[column_name])
    std = np.std(df[column_name])

    df_copy["z_score"] = (df_copy[column_name] - mean) / std

    outliers = df_copy[np.abs(df_copy["z_score"]) > 3]

    return outliers

def detect_outliers(
    df: pd.DataFrame,
    column_name: str,
    method: str
    ) -> int: 

    match method:
        case "z_score":
            outliers = define_zscore(df=df, column_name=column_name)
            return f"There are {len(outliers)} outliers in {column_name} column"
        case "iqr":
            outliers = define_iqr(df=df, column_name=column_name)
            return f"There are {len(outliers)} outliers in {column_name} column"

if __name__ == "__main__":
    df = pd.read_csv(
        filepath_or_buffer="work_from_home_burnout_dataset.csv"
    )

    day_mapper = {
        "Weekend": 0,
        "Weekday": 1
    }
    df_copy = standradize_values(
        df=df, column_name="day_type", mapper=day_mapper
    )

    burnout_mapper = {
        "Low": 0,
        "Medium": 1,
        "High": 2
    }
    df_copy = standradize_values(
        df=df_copy, column_name="burnout_risk", mapper=burnout_mapper
    )

    print("work_hours: ", 
          validate_ranges(df=df_copy, column_name="work_hours", min=0, max=24))
    print("sleep_hours: ", 
          validate_ranges(df=df_copy, column_name="sleep_hours", min=0, max=24))
    print("task_completion_rate: ", 
          validate_ranges(df=df_copy, column_name="task_completion_rate", min=0, max=100))
    
    screen_time_hours = detect_outliers(
        df=df_copy, column_name="screen_time_hours", method="z_score"
    )
    task_completion_rate = detect_outliers(
        df=df_copy, column_name="task_completion_rate", method="iqr"
    )

    print(screen_time_hours)
    print(task_completion_rate)

    df_copy.to_csv(
        path_or_buf="burnout_cleaned.csv",
        index=False
    )
