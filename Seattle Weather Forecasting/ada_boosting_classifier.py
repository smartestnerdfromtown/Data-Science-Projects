import pandas as pd
import numpy as np

from features import extract_features, split_data, prepare_data
from evaluate import evaluate_classification, save_evaluation_metrics, show_results
from define_pipeline import create_pipeline, tuning

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

def main():
    df_pipeline, X, y = prepare_data(file_path="seattle_weather_prepared.csv")
    num_features, cat_features = extract_features(df=df_pipeline)
    X_train, X_test, y_train, y_test = split_data(X=X, y=y)

    model = AdaBoostClassifier(
        estimator=None,  # If None, then the base estimator is DecisionTreeClassifier 
                         # initialized with max_depth=1.
        n_estimators=50, 
        learning_rate=1.0, 
        random_state=None
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
        model_name="ada_boosting_classifier",
        results=evaluation_metrics
    )

    param_dist = {
        "model__n_estimators": [100, 200, 300, 500],
        "model__learning_rate": np.logspace(-2, 0, 8),

        "model__estimator": [DecisionTreeClassifier()],
        "model__estimator__max_depth": [1, 2, 3],
        "model__estimator__min_samples_split": [2, 5, 10],
        "model__estimator__min_samples_leaf": [1, 2, 4]
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
        model_name="tuned_ada_boosting_classifier",
        results=evaluation_metrics
    )

if __name__ == "__main__":
    main()