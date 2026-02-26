import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

RANDOM_STATE = 777


def split_data(df: pd.DataFrame, test_size: float):
    X, y = df.loc[:, :"sample_question_papers_practiced"], df.loc[:, "performance_index"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=777
    )

    return X_train, X_test, y_train, y_test

def train_polynomial_regression(degree: int, X_train, X_test, y_train):
    polynomial_regression = PolynomialFeatures(degree=degree)

    X_train_poly = polynomial_regression.fit_transform(X_train)
    X_test_poly = polynomial_regression.transform(X_test)

    linear_regression = LinearRegression()
    linear_regression.fit(X_train_poly, y_train)

    return linear_regression, X_test_poly






def main():
    df = pd.read_csv(filepath_or_buffer="student_performance_cleaned.csv")

    X_train, X_test, y_train, y_test = split_data(df=df, test_size=0.2)

    polynomial_regression, X_test_poly = train_polynomial_regression(
        degree=3, 
        X_train=X_train, 
        X_test=X_test,
        y_train=y_train
    )

    y_pred = polynomial_regression.predict(X_test_poly)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("MSE:", mse)
    print("R2:", r2)

    print("Coefficients:", polynomial_regression.coef_)
    print("Intercept:", polynomial_regression.intercept_)



if __name__ == "__main__":
    main()