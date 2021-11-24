"""
Scoring step.

- Ingests and pre-process test dataset
- Loads trained model and make predictions on test dataset
- Compute f1score and save it into trained model folder
"""
import json
import os
import pickle
from typing import Tuple, TypeVar

import pandas as pd
from sklearn.metrics import f1_score

LR_model = TypeVar("LR_model")

with open("config.json", "r") as f:
    config = json.load(f)

test_data_path = os.path.join(config["test_data_path"], "testdata.csv")

path_to_model = os.path.join(config["output_model_path"], "trainedmodel.pkl")


def load_data(test_data_path: str) -> Tuple[pd.DataFrame, list]:
    """
    Loads dataset, drop unused columns and return `x` dataframe and `y` labels.
    """
    df = pd.read_csv(test_data_path)
    df = df.drop("corporation", axis=1)
    y = df.exited.values
    x = df.drop("exited", axis=1)

    return (x, y)


def load_model(path_to_model: str) -> LR_model:
    """
    Loads output model from training step, according
    to config.json file. Then returns that model.
    """
    with open(path_to_model, "rb") as file:
        model = pickle.load(file)

    return model


def score_model(
    load_data_output: Tuple[pd.DataFrame, list], path_to_model: str
) -> None:
    """
    Reads data processed in `load_data`, then
        - load trained model with `load_model`
        - makes prediction on test dataset
        - calculates the f1 score
        - saves the f1 score into same folder as trained model
    """
    x, y = load_data_output

    lr_mod = load_model(path_to_model)

    preds = lr_mod.predict(x)

    test_data_f1score = f1_score(y_true=y, y_pred=preds)

    f1score_saving_path = f"{path_to_model.split('/')[0]}/latestscore.txt"

    with open(f1score_saving_path, "w+") as f:
        f.write(f"f1_score = {str(test_data_f1score)}")


if __name__ == "__main__":
    df_tuple = load_data(test_data_path)
    score_model(df_tuple, path_to_model)
