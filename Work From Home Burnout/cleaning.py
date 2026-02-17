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

    df_copy.to_csv(
        path_or_buf="burnout_cleaned.csv",
        index=False
    )
