import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures

RANDOM_STATE = 777


def split_data(df: pd.DataFrame, test_size: float):
    X, y = df.loc[:, :"sample_question_papers_practiced"], df.loc[:, "performance_index"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=777
    )

    return X_train, X_test, y_train, y_test





def main():
    df = pd.read_csv(filepath_or_buffer="student_performance_cleaned.csv")

    X_train, X_test, y_train, y_test = split_data(df=df, test_size=0.2)



if __name__ == "__main__":
    main()