import pandas as pd
import numpy as np

from features import extract_features, split_data
from evaluate import evaluate_classification

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import RandomizedSearchCV


def gradient_boosting_pipeline(model, num_features: list, cat_features: list):
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
        ("model", model)
    ])

    return pipeline

def gradient_boosting_tuning(
        param_dist: dict, 
        gb_pipeline, 
        scoring: str, 
        random_state: int
    ):
    search = RandomizedSearchCV(
        estimator=gb_pipeline,
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
    df_pipeline = df.drop(columns="weather")
    X = df.drop(columns="weather")
    y = df["weather"]

    num_features, cat_features = extract_features(df=df_pipeline)
    X_train, X_test, y_train, y_test = split_data(X=X, y=y)

    gb_model = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        subsample=0.8,
        random_state=777
    )

    gb_pipeline = gradient_boosting_pipeline(
        model=gb_model,
        num_features=num_features,
        cat_features=cat_features
    )

    gb_pipeline.fit(X_train, y_train)

    y_pred = gb_pipeline.predict(X_test)
    print(accuracy_score(y_test, y_pred))
    
    param_dist = {
        "model__n_estimators": [100, 200, 300],
        "model__learning_rate": [0.01, 0.05, 0.1, 0.2],
        "model__max_depth": [3, 4, 5],
        "model__min_samples_split": [2, 5, 10],
        "model__min_samples_leaf": [1, 2, 4],
        "model__subsample": [0.6, 0.8, 1.0],
        "model__max_features": ["sqrt", "log2", None]
    }
        
    gb_pipeline = gradient_boosting_pipeline(
        model=gb_model,
        num_features=num_features,
        cat_features=cat_features
    )
    gb_tuned = gradient_boosting_tuning(
        param_dist=param_dist,
        gb_pipeline=gb_pipeline,
        scoring="accuracy",
        random_state=777
    )
    gb_tuned.fit(X_train, y_train)
    gb_best_model = gb_tuned.best_estimator_

    evaluation_metrics = evaluate_classification(
        model=gb_best_model,
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