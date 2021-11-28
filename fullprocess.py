"""
Automation of steps:
- Ingestion
- Training
- Scoring
- Deployment
- Diagnostics
- Reporting

It will:
- ingest new data, if available,
- score model against new data and get new f1 score
- check new f1 score against old f1 score
- if model drift has happened:
    - retrain model
    - re-deploy
    - report
    - test apicall
"""
import os
import subprocess as sp

import deployment
import ingestion
import scoring
import training

with open("production_deployment/ingestedfiles.txt", "r") as f:
    ingested_files = f.read()

sourcedata_files = os.listdir("sourcedata")

non_ingested_files = {
    file: (file not in ingested_files.split(",")) for file in sourcedata_files
}
print(non_ingested_files)
any_non_ingested = any(list(non_ingested_files.values()))
print(any_non_ingested)
if any_non_ingested:
    ingestion.merge_multiple_dataframe()

    path_to_new_data = os.path.join(
        ingestion.config["output_folder_path"], "finaldata.csv"
    )

    path_to_model = os.path.join(
        ingestion.config["prod_deployment_path"], "trainedmodel.pkl"
    )
    x_y_tuple = scoring.load_data(path_to_new_data)

    new_f1_score = scoring.score_model(
        load_data_output=x_y_tuple, path_to_model=path_to_model
    )
    print(new_f1_score)

    scoring.save_score(
        path_to_score=ingestion.config["output_model_path"], score=new_f1_score
    )

    path_to_current_f1_score = os.path.join(
        ingestion.config["prod_deployment_path"], "latestscore.txt"
    )

    with open(path_to_current_f1_score, "r") as f:
        # that replace is sloopy, think about how to improve it
        current_f1_score = float(f.read().replace("f1_score = ", ""))

    if new_f1_score > current_f1_score:
        print(f"No model drift: {new_f1_score} > {current_f1_score}")
        print("Ending this run")
        exit()
    else:
        print(f"Model drift! {new_f1_score} < {current_f1_score}")

        training.train_model(train_x_y=x_y_tuple, config=ingestion.config)
        deployment.mk_or_update_production_dir(ingestion.config)
        # sp.call(["python", "deployment.py"]) #test latter

        sp.call(["python", "reporting.py"])

        sp.call(["python", "apicalls.py"])
