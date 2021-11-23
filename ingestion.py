"""
Ingests data from multiple sources,
merges them and return merged de-duplicated data frame.
Expects:
    - File named config.json in root dir containing
        - input_folder_path
        - output_folder_path
Output:
    - into output_folder_path the files:
        - finaldata.csv
        - ingestedfiles.txt
"""
import json
import logging
import os

import pandas as pd

# create folder to collect logging
if not os.path.exists("logging_results"):
    os.makedirs("logging_results")


# setup logger basic config
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    # datefmt='%a, %d %b %Y %H:%M:%S',
    filename="logging_results/ingestion.log",
    filemode="w",
)

logger = logging.getLogger("ingestion")

logger.info("Loading config specs")
with open("config.json", "r") as f:
    config = json.load(f)

input_folder_path = config["input_folder_path"]
output_folder_path = config["output_folder_path"]

assert os.path.exists(input_folder_path), f"{input_folder_path} does not exist"


if not os.path.exists(output_folder_path):
    logger.info(f"{output_folder_path} does not exists, creating it.")
    os.makedirs(output_folder_path)


def merge_multiple_dataframe() -> None:
    """
    Main function. Reads data from
    """
    file_names = os.listdir(f"{os.getcwd()}/{input_folder_path}")
    file_paths = [f"{os.getcwd()}/{input_folder_path}/{i}" for i in file_names]

    df_cols = pd.read_csv(file_paths[0], nrows=1).columns
    df = pd.DataFrame(columns=df_cols)

    logger.info(f"Ingesting data frames from {input_folder_path}")

    for file in file_paths:
        try:
            df = df.append(pd.read_csv(file)).reset_index(drop=True)
        except FileNotFoundError:
            print(f"Coudn't read and append {file}")

    logger.info("Dropping duplicated rows")
    df = df.drop_duplicates()

    logger.info(
        f"Saving finaldata.csv and ingestedfiles.txt into {output_folder_path}"
    )

    # path_to_write_csv =
    df.to_csv(f"{os.getcwd()}/{output_folder_path}/finaldata.csv", index=False)

    with open(
        f"{os.getcwd()}/{output_folder_path}/ingestedfiles.txt", "w"
    ) as f:
        f.write(",".join(file_names))


if __name__ == "__main__":
    merge_multiple_dataframe()
