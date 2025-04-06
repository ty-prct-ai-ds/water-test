import pandas as pd
import numpy as np
import pickle
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def load_data(file_path:str) -> pd.DataFrame:
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        raise Exception(f"Error loading data from {file_path}: {e}")
# test_data = pd.read_csv("./data/processed/test_processed.csv")

def prepare_data(data:pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    try:
        X = data.drop(columns=['Potability'], axis=1)
        y = data['Potability']
        return X, y
    except Exception as e:
        raise Exception(f"Error preparing data: {e}")
# X_test = test_data.iloc[:, 0:-1].values
# y_test = test_data.iloc[:, -1].values

def load_model(file_path:str) -> object:
    try:
        with open(file_path, 'rb') as file:
            return pickle.load(file)
    except Exception as e:
        raise Exception(f"Error loading model from {file_path}: {e}")
# model = pickle.load(open("model.pkl", "rb"))

def evaluate_model(model:object, X:pd.DataFrame, y:pd.Series) -> dict:
    try:
        y_pred = model.predict(X)
        acc = accuracy_score(y, y_pred)
        pre = precision_score(y, y_pred)
        rec = recall_score(y, y_pred)
        f1 = f1_score(y, y_pred)

        metrics_dict = {
            "accuracy": acc,
            "precision": pre,
            "recall": rec,
            "f1_score": f1
        }
        return metrics_dict
    except Exception as e:
        raise Exception(f"Error evaluating model: {e}")


def save_metrics(metrics:dict, file_path:str) -> None:
    try:
        with open(file_path, 'w') as file:
            json.dump(metrics, file, indent=4)
    except Exception as e:
        raise Exception(f"Error saving metrics to {file_path}: {e}")
# with open("metrics.json", "w") as file:
#     json.dump(metrics_dict, file, indent=4)

def main():
    model_path = "models/model.pkl"
    test_data_path = "./data/processed/test_processed.csv"
    metrics_path = "reports/metrics.json"

    try:
        test_data = load_data(test_data_path)
        X_test, y_test = prepare_data(test_data)
        model = load_model(model_path)
        metrics = evaluate_model(model, X_test, y_test)
        save_metrics(metrics, metrics_path)
    except Exception as e:
        raise Exception(f"Error in main function: {e}")
    
if __name__ == "__main__":
    main()