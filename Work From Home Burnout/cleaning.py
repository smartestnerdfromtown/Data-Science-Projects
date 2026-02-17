import pandas as pd
import numpy as np
from typing import Dict


def standradize_values(
    df: pd.DataFrame, 
    column_name: str, 
    mapper: Dict
    ) -> pd.DataFrame:

    df_copy = df.copy(deep=True)

    df_copy[column_name] = df_copy[column_name].map(mapper)

    return df_copy

def validate_ranges(
    df: pd.DataFrame,
    column_name: str,
    min: float,
    max: float
    ) -> bool:

    if df[column_name].min() == min and df[column_name].max() == max:
        return True
    return (df[column_name].min(), df[column_name].max())


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

    print(df_copy.info())

    print("work_hours: ", 
          validate_ranges(df=df_copy, column_name="work_hours", min=0, max=24))
    print("sleep_hours: ", 
          validate_ranges(df=df_copy, column_name="sleep_hours", min=0, max=24))
    print("task_completion_rate: ", 
          validate_ranges(df=df_copy, column_name="task_completion_rate", min=0, max=100))
    

    df_copy.to_csv(
        path_or_buf="burnout_cleaned.csv",
        index=False
    )
