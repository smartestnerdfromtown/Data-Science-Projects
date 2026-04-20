import pandas as pd

def extract_features(df: pd.DataFrame):
    num_features = df.select_dtypes(include=np.number).columns.tolist()
    cat_features = df.select_dtypes(include=["object", "str"]).columns.tolist()
    return num_features, cat_features

