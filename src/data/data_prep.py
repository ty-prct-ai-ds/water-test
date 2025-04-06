import pandas as pd
import numpy as np
import os


def load_data(file_path:str) -> pd.DataFrame:
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        raise Exception(f"Error loading data from {file_path}: {e}")
# train_data = pd.read_csv("./data/raw/train.csv")
# test_data = pd.read_csv("./data/raw/test.csv")

def fill_missing_with_median(df):
    try:
        for column in df.columns:
            if df[column].isnull(). any():
                median_value = df[column].median()
                df[column].fillna(median_value, inplace=True)
        return df
    except Exception as e:
        raise Exception(f"Error filling missing values: {e}")

def save_data(df:pd.DataFrame, file_path:str) -> None:
    try:
        df.to_csv(file_path, index=False)
    except Exception as e:
        raise Exception(f"Error saving data to {file_path}: {e}")
# train_processed_data = fill_missing_with_median(train_data)
# test__processed_data = fill_missing_with_median(test_data)

def main():
    try:
        raw_data_path = "./data/raw/"
        processed_data_path = "./data/processed"
        
        train_data = load_data(os.path.join(raw_data_path, "train.csv"))
        test_data = load_data(os.path.join(raw_data_path, "test.csv"))


        train_processed_data = fill_missing_with_median(train_data)
        test_processed_data = fill_missing_with_median(test_data)

        os.makedirs(processed_data_path, exist_ok=True)

        save_data(train_processed_data,os.path.join(processed_data_path, "train_processed.csv"))
        save_data(test_processed_data,os.path.join(processed_data_path, "test_processed.csv"))
    except Exception as e:
        raise Exception(f"Error in main function: {e}")

if __name__ == "__main__":
    main()