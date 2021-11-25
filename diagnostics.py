"""
This step executes:

- Model Predictions
- Summary Statistics computation on `config.json`'s `output_folder_path`
- Missing Data percentage on `config.json`'s `output_folder_path`
- Timing of training.py and ingestion.py
- Check for outdated dependencies
"""
import json
import logging
import os
import pickle
import subprocess as sp
import timeit
from typing import Tuple

import pandas as pd

# setup logger basic config
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    # datefmt='%a, %d %b %Y %H:%M:%S',
    filename="logging_results/diagnostics.log",
    filemode="w",
)

logger = logging.getLogger("Diagnostics")


def load_config() -> dict:
    """
    Parse `config.json`.
    Input:
        config.json read
    Output:
        dict with importante directories
    """
    with open("config.json", "r") as f:
        config = json.load(f)

        logger.info("Read config file")

    dataset_csv_path = os.path.join(config["test_data_path"])
    output_folder_path = os.path.join(config["output_folder_path"])
    production_deployment_path = os.path.join(config["prod_deployment_path"])

    config_dict = {
        "dataset_csv_path": dataset_csv_path,
        "production_deployment_path": production_deployment_path,
        "output_folder_path": output_folder_path,
    }

    return config_dict


def load_data(path_to_data: str) -> Tuple[pd.DataFrame, list]:
    """
    Loads dataset, drop unused columns and return `x` dataframe and `y` labels.
    """
    df = pd.read_csv(path_to_data)
    logger.info(f"Read {path_to_data}")

    df = df.drop("corporation", axis=1)
    y = df.exited.values
    x = df.drop("exited", axis=1)

    logger.info("Dataframe prepared")

    return (x, y)


def model_predictions(config_dict: dict) -> list:
    """
    Reads the test dataset, the trained model and returns
    model's predictions for test dataset.
    Input:
        output of load_config() when applied to `config.json`
    Output:
        list of model predictions.
    """
    path_to_model = os.path.join(
        config_dict["production_deployment_path"], "trainedmodel.pkl"
    )

    with open(path_to_model, "rb") as f:
        model = pickle.load(f)

    logger.info("Model loaded")

    path_to_data = os.path.join(
        config_dict["dataset_csv_path"], "testdata.csv"
    )

    x_test, _ = load_data(path_to_data)

    preds = model.predict(x_test)

    logger.info("Made predictions")

    return preds


def dataframe_summary(config_dict: dict) -> pd.DataFrame:
    """
    Reads test dataset and compute summary statistics for
    numeric columns.
    """
    path_to_data = os.path.join(
        config_dict["dataset_csv_path"], "testdata.csv"
    )

    x_test, _ = load_data(path_to_data)

    stats = x_test.select_dtypes(include="number").mean()

    stats = stats.append(x_test.select_dtypes(include="number").std())
    stats = stats.append(x_test.select_dtypes(include="number").median())

    stats = pd.DataFrame(stats)

    stat_list = sum([[i] * 3 for i in ["mean", "std", "median"]], [])

    stats = stats.assign(summary_stat=stat_list).rename({"0": "value"})

    logger.info("Made summary statistics dataframe")

    return stats


def missing_pct(config_dict: dict) -> list:
    """
    Compute the percentage of missing (null) values in
    a dataframe found in `config.json`'s output_folder_path attribute.
    """
    path_to_data = os.path.join(
        config_dict["output_folder_path"], "finaldata.csv"
    )

    df = pd.read_csv(path_to_data)
    na_pct = df.isnull().sum() / len(df)

    logger.info("Computed percentage of missing values")

    return na_pct.values


def execution_time() -> list:
    """
    Compute execution time of training.py and ingestion.py.
    Returns a list.
    """
    # calculate timing of training.py and ingestion.py
    processing_times = []
    for file in ["ingestion.py", "training.py"]:

        try:
            t0 = timeit.default_timer()
            sp.run(["python3", file])
            processing_times.append(timeit.default_timer() - t0)
            logger.info(f"Computed processing time for {file}")
        except StopIteration:
            print(f"Check {file}, coudn't run it")

    return processing_times


def outdated_packages_list() -> None:
    """
    List outdated packages and save them into txt file.
    """
    res = sp.run(
        ["python", "-m", "pip", "list", "--outdated"], capture_output=True
    ).stdout

    logger.info("Found list of outdated packages")

    with open("check_outdated.txt", "w+") as f:
        # that "utf-8" is what makes the printing
        # be correct. DO NOT REMOVE IT
        f.write(str(res, "utf-8"))

    logger.info("Saved outdated packages list")


if __name__ == "__main__":
    cfg = load_config()
    model_predictions(cfg)
    dataframe_summary(cfg)
    missing_pct(cfg)
    execution_time()
    outdated_packages_list()
