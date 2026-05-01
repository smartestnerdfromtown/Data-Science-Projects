import pandas as pd
import numpy as np

def z_score(df: pd.DataFrame, column_name: str) -> pd.Series:
    mean = df[column_name].mean()
    std = df[column_name].std()
	
    z_scores = (df[column_name] - mean) / std
	
    return z_scores


def calculate_iqr(df: pd.DataFrame, column_name: str) -> pd.Series:
    Q1 = df[column_name].quantile(0.25)
    Q3 = df[column_name].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    return IQR, lower_bound, upper_bound



