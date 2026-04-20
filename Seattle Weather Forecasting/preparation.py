import pandas as pd

from sklearn.preprocessing import LabelEncoder


def convert_to_datetime(df: pd.DataFrame, column: str) -> pd.DataFrame:
    df[column] = pd.to_datetime(df[column], errors="coerce")
    return df

def check_null(df: pd.DataFrame) -> pd.Series:
    return df.isnull().sum()

def check_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    return df.duplicated()

def extract_part(df: pd.DataFrame, column: str, part: str) -> pd.Series:
    match part:
        case "year":
            extracted_part = df[column].dt.year
        case "month":
            extracted_part = df[column].dt.month_name()
        case "day":
            extracted_part = df[column].dt.day_name()

    return extracted_part
            
def main():
    df = pd.read_csv(filepath_or_buffer="seattle-weather.csv")
    print(df.head())
    print(df.info())
    print(check_null(df=df))
    print(check_duplicates(df=df).sum())

    df = convert_to_datetime(df=df, column="date")
    print(df.info())

    df["year"] = extract_part(df=df, column="date", part="year")
    df["month"] = extract_part(df=df, column="date", part="month")
    df["day"] = extract_part(df=df, column="date", part="day")

    df["weather_encoded"] = LabelEncoder().fit_transform(df["weather"])

    print(df.head())
    print(df.tail())

    df.to_csv(path_or_buf="seattle_weather_prepared.csv")



if __name__ == "__main__":
    main()