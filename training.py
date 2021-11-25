"""
Train a logistic regression model on outputed data from ingestion.py
Input:
    - ingesteddata/finaldata.csv
Output:
    - practicemodels/trainedmodel.pkl
"""
import json
import logging
import os
import pickle
from typing import Tuple

import pandas as pd
from pandas.core.construction import array
from sklearn.linear_model import LogisticRegression

# setup logger basic config
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    # datefmt='%a, %d %b %Y %H:%M:%S',
    filename="logging_results/training.log",
    filemode="w",
)

logger = logging.getLogger("Training")


def load_and_prepare(config: json) -> Tuple[pd.DataFrame, array]:
    """
    Reads the configuration file path to ingestion.py output and
    returns a tuple containing (x,y) pair of training matrix `x` and
    `y` labels
    """
    dataset_csv_path = os.path.join(
        config["output_folder_path"], "finaldata.csv"
    )
    df = pd.read_csv(dataset_csv_path)
    logger.info("Read finaldata.csv")

    df = df.drop("corporation", axis=1)
    y = df.exited.values
    x = df.drop("exited", axis=1)
    logger.info("Dropped columns and creating x/y data")

    return (x, y)


def train_model(train_x_y: Tuple[pd.DataFrame, array], config: json) -> None:
    """
    Reads the output of load_and_prepare and the json config, train a logistic
    regression model and save it specified output folder in config.
    """
    x, y = train_x_y

    lr_mod = LogisticRegression(
        C=1.0,
        class_weight=None,
        dual=False,
        fit_intercept=True,
        intercept_scaling=1,
        l1_ratio=None,
        max_iter=100,
        multi_class="auto",
        n_jobs=None,
        penalty="l2",
        random_state=0,
        solver="liblinear",
        tol=0.0001,
        verbose=0,
        warm_start=False,
    )
    logger.info("Started training model")
    lr_mod.fit(x, y)
    logger.info("Model trained")

    if not os.path.exists(config["output_model_path"]):
        os.makedirs(config["output_model_path"])

    model_output_path = os.path.join(
        config["output_model_path"], "trainedmodel.pkl"
    )

    pickle.dump(lr_mod, open(model_output_path, "wb"))

    logger.info(f"Saved model into {model_output_path}")


if __name__ == "__main__":
    with open("config.json", "r") as f:
        config = json.load(f)

    res = load_and_prepare(config)
    train_model(res, config)
