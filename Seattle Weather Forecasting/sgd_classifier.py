import pandas as pd
import numpy as np

from features import extract_features, split_data, prepare_data
from evaluate import evaluate_classification, save_evaluation_metrics, show_results
from define_pipeline import create_pipeline, tuning

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import SGDClassifier


RANDOM_STATE = 777
N_JOBS = -1


def main():
    df_pipeline, X, y = prepare_data(file_path="seattle_weather_prepared.csv")
    num_features, cat_features = extract_features(df=df_pipeline)
    X_train, X_test, y_train, y_test = split_data(X=X, y=y)

    model = SGDClassifier(
        loss='hinge', 
        penalty='l2', 
        alpha=0.0001, 
        l1_ratio=0.15, 
        fit_intercept=True, 
        max_iter=1000, 
        tol=0.001, 
        shuffle=True, 
        verbose=0, 
        epsilon=0.1, 
        n_jobs=N_JOBS, 
        random_state=RANDOM_STATE, 
        learning_rate='optimal', 
        eta0=0.01, 
        power_t=0.5, 
        early_stopping=False, 
        validation_fraction=0.1, 
        n_iter_no_change=5, 
        class_weight=None, 
        warm_start=False, 
        average=False
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
        include_cm=False,
        include_roc_auc=False,
        return_dict=True,
        verbose=True
    )

    show_results(results=evaluation_metrics)

    save_evaluation_metrics(
        file_to_csv="models_metrics.csv", 
        model_name="sgd_classifier",
        results=evaluation_metrics
    )

    param_dist = {
        "model__loss": ["hinge", "log_loss", "modified_huber"],

        "model__penalty": ["l2", "elasticnet"],
        "model__alpha": np.logspace(-5, -2, 10),
        "model__l1_ratio": np.linspace(0.1, 0.9, 5),

        "model__learning_rate": ["optimal", "adaptive"],
        "model__eta0": np.logspace(-3, -1, 8),

        "model__max_iter": [1000, 2000, 3000],
        "model__tol": np.logspace(-4, -3, 5),

        "model__early_stopping": [True],
        "model__validation_fraction": [0.1, 0.15],

        "model__average": [True, False],
        "model__class_weight": [None, "balanced"]
    }

    pipeline = create_pipeline(
        model=model,
        num_features=num_features,
        cat_features=cat_features,
        scaler=StandardScaler(),
        encoder=OneHotEncoder()
    )
    pipeline_tuned = tuning(
        param_dist=param_dist,
        pipeline=pipeline,
        scoring="f1_weighted",
        n_iter=30
    )
    
    pipeline_tuned.fit(X_train, y_train)

    pipeline_best_model = pipeline_tuned.best_estimator_

    evaluation_metrics = evaluate_classification(
        model=pipeline_best_model,
        X_test=X_test,
        y_test=y_test,
        include_cm=False,
        include_roc_auc=False,
        average="weighted",
        return_dict=True,
        verbose=True
    )

    print("-" * 20)

    show_results(results=evaluation_metrics)

    save_evaluation_metrics(
        file_to_csv="models_metrics.csv", 
        model_name="tuned_sgd_classifier",
        results=evaluation_metrics
    )

if __name__ == "__main__":
    main()