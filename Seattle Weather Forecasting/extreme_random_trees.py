import pandas as pd
import numpy as np

from features import extract_features, split_data, prepare_data
from evaluate import evaluate_classification, save_evaluation_metrics

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import ExtraTreesClassifier


RANDOM_STATE = 777


def create_pipeline(model, num_features: list, cat_features: list):
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
    df_pipeline, X, y = prepare_data(file_path="seattle_weather_prepared.csv")
    num_features, cat_features = extract_features(df=df_pipeline)
    X_train, X_test, y_train, y_test = split_data(X=X, y=y)

    model = ExtraTreesClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="sqrt",
        bootstrap=False,
        criterion="gini",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    pipeline = create_pipeline(
        model=model,
        num_features=num_features,
        cat_features=cat_features
    )

    pipeline.fit(X_train, y_train)

    evaluation_metrics = evaluate_classification(
        model=pipeline,
        X_test=X_test,
        y_test=y_test,
        average="weighted",
        include_cm=False,
        include_roc_auc=False,
        return_dict=True,
        verbose=True
    )

    for metric, value in evaluation_metrics.items():
        print(f"{metric}: {value}")

    save_evaluation_metrics(
        file_to_csv="models_metrics.csv", 
        model_name="extreme_random_forest",
        results=evaluation_metrics
    )
    

if __name__ == "__main__":
    main()
