import pandas as pd
import numpy as np

from features import extract_features, split_data, prepare_data
from evaluate import evaluate_classification, save_evaluation_metrics, show_results
from define_pipeline import create_pipeline, tuning

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier


def main():
    df_pipeline, X, y = prepare_data(file_path="seattle_weather_prepared.csv")
    num_features, cat_features = extract_features(df=df_pipeline)
    X_train, X_test, y_train, y_test = split_data(X=X, y=y)

    model = XGBClassifier(
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

    pipeline = create_pipeline(
        model=model,
        num_features=num_features,
        cat_features=cat_features,
        scaler=StandardScaler(),
        encoder=OneHotEncoder()
    )

    pipeline.fit(X_train, y_train)

    evaluation_metrics = evaluate_classification(
        model=pipeline,
        X_test=X_test,
        y_test=y_test,
        average="weighted",
        return_dict=True,
        verbose=True
    )

    show_results(results=evaluation_metrics)

    save_evaluation_metrics(
        file_to_csv="models_metrics.csv", 
        model_name="extreme_boosting_classifier",
        results=evaluation_metrics
    )

    ################# RANDOMIZED SEARCH OF HYPERPATAMETERS #################

    param_dist = {
        "model__n_estimators": [100, 200, 300],
        "model__learning_rate": [0.01, 0.05, 0.1],
        "model__max_depth": [3, 4, 5],
        "model__min_child_weight": [1, 3, 5],
        "model__gamma": [0, 0.1, 0.3],
        "model__subsample": [0.6, 0.8, 1.0],
        "model__colsample_bytree": [0.6, 0.8, 1.0],
    }
        
    pipeline = create_pipeline(
        model=model,
        num_features=num_features,
        cat_features=cat_features
    )
    pipeline_tuned = tuning(
        param_dist=param_dist,
        pipeline=pipeline,
        scoring="accuracy",
        random_state=777
    )
    pipeline_tuned.fit(X_train, y_train)
    pipeline_best_model = pipeline_tuned.best_estimator_

    evaluation_metrics = evaluate_classification(
        model=pipeline_best_model,
        X_test=X_test,
        y_test=y_test,
        average="weighted",
        return_dict=True,
        verbose=True
    )

    print("-" * 20)

    show_results(results=evaluation_metrics)

    save_evaluation_metrics(
        file_to_csv="models_metrics.csv", 
        model_name="tuned_extreme_boosting_classifier",
        results=evaluation_metrics
    )


if __name__ == "__main__":
    main()