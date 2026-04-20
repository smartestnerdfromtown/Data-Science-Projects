import pandas as pd
import numpy as np

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def main():
    df = pd.read_csv(filepath_or_buffer="seattle_weather_prepared.csv")
    df = df.drop(columns="date")
    df_pipeline = df.drop(columns="weather")
    X = df.drop(columns="weather")
    y = df["weather"]



if __name__ == "__main__":
    main()