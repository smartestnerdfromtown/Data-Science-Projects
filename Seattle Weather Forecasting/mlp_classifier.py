import pandas as pd
import numpy as np

from features import extract_features, split_data, prepare_data
from evaluate import evaluate_classification, save_evaluation_metrics
from define_pipeline import create_pipeline, tuning

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import RandomizedSearchCV
from sklearn.neural_network import MLPClassifier

RANDOM_STATE = 777

def main():
    df_pipeline, X, y = prepare_data(file_path="seattle_weather_prepared.csv")
    num_features, cat_features = extract_features(df=df_pipeline)
    X_train, X_test, y_train, y_test = split_data(X=X, y=y)

    model = MLPClassifier(
        hidden_layer_sizes=(100,), 
        activation='relu',
        solver='adam', 
        alpha=0.0001, 
        batch_size='auto', 
        learning_rate='constant', 
        learning_rate_init=0.001,
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
        model_name="mlp_classifier",
        results=evaluation_metrics
    )

    param_dist = {
        "model__hidden_layer_sizes": [
            (50,), (100,), (100, 50), (128, 64), (64, 32, 16)
        ],
        "model__activation": ["relu", "tanh"],
        "model__solver": ["adam", "sgd"],
        "model__alpha": np.logspace(-5, -1, 10),               # L2 regularization
        "model__learning_rate": ["constant", "adaptive"],
        "model__learning_rate_init": np.logspace(-4, -2, 10),
        "model__batch_size": [32, 64, 128, 256],
        "model__early_stopping": [True],
        "model__max_iter": [300, 500, 800]
    }

    pipeline = create_pipeline(
        model=model,
        num_features=num_features,
        cat_features=cat_features
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

    for metric, value in evaluation_metrics.items():
        print(f"{metric}: {value}")

    save_evaluation_metrics(
        file_to_csv="models_metrics.csv", 
        model_name="tuned_mlp_classifier",
        results=evaluation_metrics
    )

if __name__ == "__main__":
    main()