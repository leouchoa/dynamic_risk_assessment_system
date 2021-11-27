"""
Requests some services from app.py, get the results and save them to file.
IMPORTANT:
    Before running the script you first need to run the API.
    That can be done in an environment with created with the
    requirements file.
    To then run the API use:
    python app.py
"""
import json
import os

import requests

with open("config.json", "r") as f:
    config = json.load(f)

# set localhost url
URL = "http://127.0.0.1:8000/"


def responses(URL):
    """
    Make requests to each endpoint and return them as a dictionary.
    IMPORTANT:
        Before running the script you first need to run the API.
        That can be done in an environment with created with the
        requirements file.
        To then run the API use:
        python app.py
    """
    response1 = requests.post(
        f"{URL}/prediction?data_loc=testdata/testdata.csv"
        # 'http://127.0.0.1:8000/prediction?data_loc=testdata/testdata.csv'
    ).content

    response2 = requests.get(
        f"{URL}/scoring?data_loc=testdata/testdata.csv"
    ).content
    response3 = requests.get(
        f"{URL}/summarystats?data_loc=testdata/testdata.csv"
    ).content
    response4 = requests.get(
        f"{URL}/diagnostics?data_loc=testdata/testdata.csv"
    ).content

    res_dict = {
        "response1": str(response1, "utf-8"),
        "response2": str(response2, "utf-8"),
        "response3": str(response3, "utf-8"),
        "response4": str(response4, "utf-8"),
    }
    return res_dict


def save_api_responses(responses_output: list, saving_path: str) -> None:
    """
    Save the API responses into a file
    """
    with open(saving_path, "w") as f:
        f.write(str(responses_output))
    # with open(saving_path, "w") as outfile:
    #     pretty_res = json.dumps(responses_output,indent=3)
    #     json.dump(pretty_res, outfile)


if __name__ == "__main__":

    requests_res_save_path = os.path.join(
        config["output_model_path"], "apireturns.txt"
    )
    # requests_res_save_path_json = os.path.join(
    #     config['output_model_path'],
    #     'apireturns.json'
    # )

    res = responses(URL)

    save_api_responses(
        responses_output=res,
        # saving_path=requests_res_save_path_json
        saving_path=requests_res_save_path,
    )
