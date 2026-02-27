import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor 
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

RANDOM_STATE = 777


def split_data(df: pd.DataFrame, test_size: float):
    X, y = df.loc[:, :"sample_question_papers_practiced"], df.loc[:, "performance_index"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=777
    )

    return X_train, X_test, y_train, y_test

def scale_data(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled

def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return mse, rmse, mae, r2

def train_polynomial_regression(degree: int, interaction_only: bool, X_train, X_test, y_train):
    polynomial_regression = PolynomialFeatures(degree=degree, interaction_only=interaction_only)

    X_train_poly = polynomial_regression.fit_transform(X_train)
    X_test_poly = polynomial_regression.transform(X_test)

    linear_regression = LinearRegression()
    linear_regression.fit(X_train_poly, y_train)

    return linear_regression, X_test_poly

def grid_search_polynomial_regression(X_train, X_test, y_train, y_test):
    info = dict()
    
    for degree in range(2, 6 + 1):
        for interaction_only in [True, False]:
            print(f"Checking for pair: ({degree}, {interaction_only})")

            polynomial_regression, X_test_poly = train_polynomial_regression(
                degree=degree,
                interaction_only=interaction_only, 
                X_train=X_train, 
                X_test=X_test, 
                y_train=y_train
            )

            info[(degree, interaction_only)] = evaluate(
                model=polynomial_regression, 
                X_test=X_test_poly,
                y_test=y_test
            )

    return info

def randomized_search_random_forest_regression(
        param_grid: dict, 
        n_iter: int, 
        scoring: str,
        X_train,
        y_train
    ):
    random_search = RandomizedSearchCV(
        RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1),
        param_distributions=param_grid,
        n_iter=n_iter,
        cv=5,
        scoring=scoring,
        random_state=RANDOM_STATE,
        n_jobs=-1
    ) 

    random_search.fit(X_train, y_train)

    return random_search.best_estimator_, random_search.best_params_

def random_forest_regression_feature_importance(
        random_forest_model,
        df: pd.DataFrame
    ):
    importances = random_forest_model.feature_importances_
    feature_names = df.columns[: len(df.columns) - 1]

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    }).sort_values(by="importance", ascending=False)

    return importance_df


def train_basic_knn(X_train_scaled, y_train, n_neighbors: int = 5):
    knn = KNeighborsRegressor(
        n_neighbors=n_neighbors,
        weights="uniform",      
        algorithm="auto",       
        leaf_size=30,
        p=2,                    
        n_jobs=-1
    )

    knn.fit(X_train_scaled, y_train)

    return knn

def randomized_search_knn(
        X_train_scaled, 
        y_train, 
        param_grid, 
        n_iter: int = 25, 
        scoring: str = "r2"
    ):     
    random_search = RandomizedSearchCV(
        KNeighborsRegressor(n_jobs=-1),
        param_distributions=param_grid,
        n_iter=n_iter,
        cv=5,
        scoring=scoring,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    random_search.fit(X_train_scaled, y_train)

    return random_search.best_estimator_, random_search.best_params_



def main():
    df = pd.read_csv(filepath_or_buffer="student_performance_cleaned.csv")

    X_train, X_test, y_train, y_test = split_data(df=df, test_size=0.2)
    X_train_scaled, X_test_scaled = scale_data(X=X_train), scale_data(X=X_test)

    info = grid_search_polynomial_regression(X_train, X_test, y_train, y_test)
    degree, interaction_only = 2, True

    polynomial_regression, X_test_poly = train_polynomial_regression(
        degree=degree, 
        interaction_only=interaction_only,
        X_train=X_train, 
        X_test=X_test,
        y_train=y_train
    )

    print("Coefficients:", polynomial_regression.coef_)
    print("Intercept:", polynomial_regression.intercept_)

    mse, rmse, mae, r2 = evaluate(model=polynomial_regression, X_test=X_test_poly, y_test=y_test)
    print("MSE:", mse)
    print("RMSE:", rmse)
    print("MAE:", mae)
    print("R2:", r2)

    
    param_grid = {
        "n_estimators": np.arange(100, 1000, 100),
        "max_depth": [None] + list(np.arange(5, 51, 5)),
        "min_samples_split": np.arange(2, 11),
        "min_samples_leaf": np.arange(1, 6),
        "max_features": ["sqrt", "log2", None]
    }

    random_forest_regression, best_params = randomized_search_random_forest_regression(
        param_grid=param_grid,
        n_iter=10,
        scoring="r2",
        X_train=X_train,
        y_train=y_train
    )

    print(best_params)
    print(type(random_forest_regression))

    mse, rmse, mae, r2 = evaluate(
        model=random_forest_regression, 
        X_test=X_test, 
        y_test=y_test
    )
    print("MSE:", mse)
    print("RMSE:", rmse)
    print("MAE:", mae)
    print("R2:", r2)

    feature_importance = random_forest_regression_feature_importance(
        random_forest_model=random_forest_regression, 
        df=df
    )
    print(feature_importance)

    basic_knn = train_basic_knn(X_train_scaled=X_train_scaled, y_train=y_train)
    mse, rmse, mae, r2 = evaluate(model=basic_knn, X_test=X_test_scaled, y_test=y_test)
    print("MSE:", mse)
    print("RMSE:", rmse)
    print("MAE:", mae)
    print("R2:", r2)

    param_grid = {
        "n_neighbors": [3, 5, 7, 9, 15, 25],
        "weights": ["uniform", "distance"],
        "p": [1, 2],
        "algorithm": ["auto", "ball_tree", "kd_tree", "brute"]
    }
    random_knn, random_knn_parameters = randomized_search_knn(
        X_train_scaled=X_train_scaled,
        y_train=y_train,
        param_grid=param_grid,
        n_iter=25, 
        scoring="r2"
    )

    mse, rmse, mae, r2 = evaluate(model=random_knn, X_test=X_test_scaled, y_test=y_test)
    print("MSE:", mse)
    print("RMSE:", rmse)
    print("MAE:", mae)
    print("R2:", r2)

if __name__ == "__main__":
    main()