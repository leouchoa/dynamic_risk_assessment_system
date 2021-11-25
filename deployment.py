"""
It will copy your
        - trained model (trainedmodel.pkl)
        - model score (latestscore.txt)
        - ingested data record (ingestedfiles.txt)
    into `prod_deployment_path` read from `config.json`

Input:
    config.json
Output:
    3 copied files into `prod_deployment_path`, the production folder
"""
import json
import logging
import os
import shutil

# setup logger basic config
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    # datefmt='%a, %d %b %Y %H:%M:%S',
    filename="logging_results/deployment.log",
    filemode="w",
)

logger = logging.getLogger("Deployment")

logger.info("Read config file")
with open("config.json", "r") as f:
    config = json.load(f)


def make_sender_paths(config: dict) -> dict:
    """
    Create source folder from the files will be copied.
    Input:
        config.json read
    Output:
        dict with source dir
    """
    trainedmodel_sender_path = os.path.join(
        config["output_model_path"], "trainedmodel.pkl"
    )

    latestscore_sender_path = os.path.join(
        config["output_model_path"], "latestscore.txt"
    )

    ingestedfiles_sender_path = os.path.join(
        config["output_folder_path"], "ingestedfiles.txt"
    )

    sender_dict = {
        "trainedmodel_sender_path": trainedmodel_sender_path,
        "latestscore_sender_path": latestscore_sender_path,
        "ingestedfiles_sender_path": ingestedfiles_sender_path,
    }

    return sender_dict


def store_model_into_pickle(config: dict) -> None:
    """
    It will copy your
        - trained model (trainedmodel.pkl)
        - model score (latestscore.txt)
        - ingested data record (ingestedfiles.txt)
    into `prod_deployment_path` read from `config.json`
    """
    prod_deployment_path = os.path.join(config["prod_deployment_path"])
    sender_dict = make_sender_paths(config)

    if not os.path.exists(prod_deployment_path):
        logger.info(f"Creating {prod_deployment_path} dir")
        os.makedirs(prod_deployment_path)

    for sender_path in sender_dict.values():

        shutil.copy(src=sender_path, dst=prod_deployment_path)

        logger.info(f"Copied {sender_path}")


if __name__ == "__main__":
    store_model_into_pickle(config)
