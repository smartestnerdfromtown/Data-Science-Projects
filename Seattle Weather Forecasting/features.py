import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

def extract_features(df: pd.DataFrame):
    num_features = df.select_dtypes(include=np.number).columns.tolist()
    cat_features = df.select_dtypes(include=["object", "string"]).columns.tolist()
    return num_features, cat_features

def split_data(
        X: pd.DataFrame, 
        y: pd.Series, 
        random_state: int = 777, 
        test_size: float = 0.2
    ):
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    return X_train, X_test, y_train, y_test