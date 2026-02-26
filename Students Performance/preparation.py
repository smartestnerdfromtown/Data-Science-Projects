import pandas as pd


def standardize_column_names(df: pd.DataFrame):
    df_copy = df.copy(deep=True)
    
    columns = df_copy.columns
    columns_mappings = dict()

    for column in columns:
        new_column_name = column.lower().replace(" ", "_").strip()
        columns_mappings[column] = new_column_name

    df_copy = df_copy.rename(columns=columns_mappings)

    return df_copy







def main():
    df = pd.read_csv(
        filepath_or_buffer="Student_Performance.csv"
    )

    print(df.info())

    df = standardize_column_names(df)

    print(df.info())

    print(df)

    df.to_csv(path_or_buf="student_performance_cleaned.csv", index=False)

if __name__ == "__main__":
    main()