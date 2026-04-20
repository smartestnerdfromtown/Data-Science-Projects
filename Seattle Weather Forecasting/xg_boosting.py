import pandas as pd
import numpy as np

from features import extract_features, split_data
from evaluate import evaluate_classification

from xgboost import XGBClassifier

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import RandomizedSearchCV

def extreme_gradient_boosting_pipeline(model, num_features: list, cat_features: list):
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median"))
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
        ("xgb", model)
    ])

    return pipeline

def extreme_gradient_boosting_tuning(
        param_dist: dict, 
        xgb_pipeline, 
        scoring: str, 
        random_state: int
    ):
    search = RandomizedSearchCV(
        estimator=xgb_pipeline,
        param_distributions=param_dist,
        n_iter=20,
        cv=5,
        scoring=scoring,
        n_jobs=-1,
        random_state=random_state
    )

    return search

def main():
    df = pd.read_csv(filepath_or_buffer="seattle_weather_prepared.csv")
    df = df.drop(columns="date")

    df_pipeline = df.drop(columns=["weather", "weather_encoded"])
    X = df.drop(columns=["weather", "weather_encoded"])
    y = df["weather_encoded"]

    num_features, cat_features = extract_features(df=df_pipeline)
    X_train, X_test, y_train, y_test = split_data(X=X, y=y)

    xgb_model = XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        eval_metric="mlogloss",
        objective="multi:softprob"
    )

    xgb_pipeline = extreme_gradient_boosting_pipeline(
        model=xgb_model,
        num_features=num_features,
        cat_features=cat_features
    )

    xgb_pipeline.fit(X_train, y_train)

    evaluation_metrics = evaluate_classification(
        model=xgb_pipeline,
        X_test=X_test,
        y_test=y_test,
        average="weighted",
        return_dict=True,
        verbose=True
    )

    for metric, value in evaluation_metrics.items():
        print(f"{metric}: {value}")

    param_dist = {
        "xgb__n_estimators": [100, 200, 300],
        "xgb__learning_rate": [0.01, 0.05, 0.1],
        "xgb__max_depth": [3, 4, 5],
        "xgb__min_child_weight": [1, 3, 5],
        "xgb__gamma": [0, 0.1, 0.3],
        "xgb__subsample": [0.6, 0.8, 1.0],
        "xgb__colsample_bytree": [0.6, 0.8, 1.0],
    }
        
    xgb_pipeline = extreme_gradient_boosting_pipeline(
        model=xgb_model,
        num_features=num_features,
        cat_features=cat_features
    )
    xgb_tuned = extreme_gradient_boosting_tuning(
        param_dist=param_dist,
        xgb_pipeline=xgb_pipeline,
        scoring="accuracy",
        random_state=777
    )
    xgb_tuned.fit(X_train, y_train)
    xgb_best_model = xgb_tuned.best_estimator_

    evaluation_metrics = evaluate_classification(
        model=xgb_best_model,
        X_test=X_test,
        y_test=y_test,
        average="weighted",
        return_dict=True,
        verbose=True
    )

    print("-" * 20)

    for metric, value in evaluation_metrics.items():
        print(f"{metric}: {value}")


if __name__ == "__main__":
    main()