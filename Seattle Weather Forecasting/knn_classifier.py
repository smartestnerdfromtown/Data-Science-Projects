import pandas as pd
import numpy as np

from features import extract_features, split_data
from evaluate import evaluate_classification

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier


def create_knn_pipeline(model, num_features: list, cat_features: list):
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("num", num_pipeline, num_features),
        ("cat", cat_pipeline, cat_features)
    ])

    pipeline = Pipeline([
        ("preprocessing", preprocessor),
        ("model", model)
    ])

    return pipeline

def main():
    df = pd.read_csv(filepath_or_buffer="seattle_weather_prepared.csv")
    df = df.drop(columns="date")

    df_pipeline = df.drop(columns=["weather", "weather_encoded"])
    X = df.drop(columns=["weather", "weather_encoded"])
    y = df["weather_encoded"]

    num_features, cat_features = extract_features(df=df_pipeline)
    X_train, X_test, y_train, y_test = split_data(X=X, y=y)

    knn_model = KNeighborsClassifier(
        n_neighbors=5,
        weights="distance",
        metric="minkowski",
        p=2
    )

    knn_pipeline = create_knn_pipeline(
        model=knn_model,
        num_features=num_features,
        cat_features=cat_features
    )

    knn_pipeline.fit(X_train, y_train)

    evaluation_metrics = evaluate_classification(
        model=knn_pipeline,
        X_test=X_test,
        y_test=y_test,
        average="weighted",
        return_dict=True,
        verbose=True
    )

    for metric, value in evaluation_metrics.items():
        print(f"{metric}: {value}")


if __name__ == "__main__":
    main()