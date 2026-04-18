import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import RandomizedSearchCV

RANDOM_STATE = 777

def split_data(X: pd.DataFrame, y: pd.Series):
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y
    )

    return X_train, X_test, y_train, y_test

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

def extract_features(df: pd.DataFrame):
    num_features = df.select_dtypes(include=np.number).columns.tolist()
    cat_features = df.select_dtypes(include=["object", "str"]).columns.tolist()
    return num_features, cat_features

def random_forest_pipeline(X_train, y_train, num_features: list, cat_features: list):
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

    rf = RandomForestClassifier(
        n_estimators=500,        
        max_depth=20,            
        min_samples_split=5,     
        min_samples_leaf=2,      
        max_features="sqrt",     
        bootstrap=True,          
        n_jobs=-1,               
        random_state=RANDOM_STATE
    )

    pipeline = Pipeline([
        ("preprocessing", preprocessor),
        ("model", rf)
    ])

    return pipeline

def random_forest_tuning(param_dist, rf_pipeline):
    search = RandomizedSearchCV(
        rf_pipeline,
        param_distributions=param_dist,
        n_iter=20,
        cv=5,
        scoring="accuracy",
        n_jobs=-1,
        random_state=RANDOM_STATE,
        verbose=1
    )

    return search

def visualize_importances(feature_importance: pd.DataFrame):
    feature_importance = feature_importance.sort_values("importance", ascending=True)

    plt.figure(figsize=(10, 6))

    plt.barh(feature_importance["feature"], feature_importance["importance"])

    plt.title("Feature Importances (Random Forest)", fontsize=14)
    plt.xlabel("Importance Score")
    plt.ylabel("Features")

    plt.tight_layout()
    plt.show()

def main():
    df = pd.read_csv(filepath_or_buffer="seattle_weather_prepared.csv")
    df = df.drop(columns="date")
    df_pipeline = df.drop(columns="weather")
    X = df.drop(columns="weather")
    y = df["weather"]

    num_features, cat_features = extract_features(df=df_pipeline)
    X_train, X_test, y_train, y_test = split_data(X=X, y=y)

    rf_pipeline = random_forest_pipeline(
        X_train=X_train, 
        y_train=y_train, 
        num_features=num_features, 
        cat_features=cat_features
    )

    rf = random_forest_pipeline(
        X_train=X_train, 
        y_train=y_train, 
        num_features=num_features, 
        cat_features=cat_features
    )
    rf.fit(X_train, y_train)

    print(evaluate_model(model=rf, X_test=X_test, y_test=y_test))


    param_dist = {
        "model__n_estimators": [100, 200, 500],
        "model__max_depth": [None, 10, 20, 30],
        "model__min_samples_split": [2, 5, 10],
        "model__min_samples_leaf": [1, 2, 4],
        "model__max_features": ["sqrt", "log2"]
    }
    
    rf_tuned = random_forest_tuning(param_dist=param_dist, rf_pipeline=rf_pipeline)
    rf_tuned.fit(X_train, y_train)
    rf_best_model = rf_tuned.best_estimator_
    print(evaluate_model(model=rf_best_model, X_test=X_test, y_test=y_test))


if __name__ == "__main__":
    main()