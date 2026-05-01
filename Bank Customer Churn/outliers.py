import pandas as pd
import numpy as np

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


RANDOM_STATE = 777


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


def find_outliers_via_forest(df: pd.DataFrame, numerical_cols: list, to_scale: bool) -> pd.Series:
    X = df.loc[:, numerical_cols]
    
    if to_scale:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)


    model = IsolationForest(
        contamination="auto",
        random_state=RANDOM_STATE,
        n_estimators=200,     
        max_samples="auto"
    )

    model.fit(X)

    scores = model.decision_function(X)
    labels = model.predict(X)  # -1 = anomaly, 1 = normal

    return labels


