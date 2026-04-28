import pandas as pd
import numpy as np

from features import extract_features, split_data, prepare_data
from evaluate import evaluate_classification, save_evaluation_metrics, show_results
from define_pipeline import create_pipeline, tuning

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression


RANDOM_STATE = 777


def main():
    df_pipeline, X, y = prepare_data(file_path="seattle_weather_prepared.csv")
    num_features, cat_features = extract_features(df=df_pipeline)
    X_train, X_test, y_train, y_test = split_data(X=X, y=y)

    model = LogisticRegression(
        C=0.7,
        solver="lbfgs",
        max_iter=5000,
        l1_ratio=0
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
        model_name="logistic_regression",
        results=evaluation_metrics
    )


    ################# RANDOMIZED SEARCH OF HYPERPATAMETERS #################


    pipeline = create_pipeline(
        model=model,
        num_features=num_features,
        cat_features=cat_features,
        scaler=StandardScaler(),
        encoder=OneHotEncoder()
    )

    param_dist = {
        "model__penalty": ["l2"],
        "model__C": [0.01, 0.1, 1, 10, 100],
        "model__solver": ["lbfgs", "saga"]
    }

    pipeline_tuned = tuning(
        param_dist=param_dist,
        pipeline=pipeline,
        scoring="f1_weighted",
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

    show_results(results=evaluation_metrics)

    save_evaluation_metrics(
        file_to_csv="models_metrics.csv", 
        model_name="tuned_knn_classifier",
        results=evaluation_metrics
    )



if __name__ == "__main__":
    main()