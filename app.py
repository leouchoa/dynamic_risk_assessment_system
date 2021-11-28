"""
Flask API with:
- Prediction Endpoint
- Scoring Endpoint
- Summary Statistic Endpoint
- Diagnostics Endpoint
"""
import os

from flask import Flask, jsonify, request

import diagnostics as dgn
import scoring

# Set up variables for use in our script
app = Flask(__name__)
app.secret_key = "1652d576-484a-49fd-913a-6879acfa6ba4"

cfg = dgn.load_config()

path_to_model = os.path.join(cfg["prod_deployment_path"], "trainedmodel.pkl")
path_to_score = cfg["prod_deployment_path"]


@app.route("/prediction", methods=["POST", "OPTIONS"])
def predict():
    """
    Prediction Endpoint.
    Query params: data_loc.
    Example:
        curl -X POST '127.0.0.1:8000/prediction?data_loc=testdata/testdata.csv'
    Returns:
        - list of model predictions
    """
    data_loc = request.args.get("data_loc")
    x_y_tuple = dgn.load_data(data_loc)
    preds = dgn.model_predictions(
        x_y_tuple=x_y_tuple, path_to_model=path_to_model
    )
    return str(preds)


# Scoring Endpoint
@app.route("/scoring")
def score():
    """
    Scoring Endpoint.
    Query params: data_loc.
    Example:
        curl '127.0.0.1:8000/scoring?data_loc=testdata/testdata.csv'
    Returns:
        - f1 model model and also saves it to a file called
            `latestscore.txt`
    """
    data_loc = request.args.get("data_loc")
    df_tuple = scoring.load_data(data_loc)
    f1_score = scoring.score_model(df_tuple, path_to_model)
    scoring.save_score(path_to_score=path_to_score, score=f1_score)

    return f"saved f1 score of {f1_score}"


# Summary Statistics Endpoint
@app.route("/summarystats")
def summarystats():
    """
    Dataframe Summary Statistics Endpoint.
    Query params: data_loc.
    Example:
        curl '127.0.0.1:8000/summarystats?data_loc=testdata/testdata.csv'
    Returns:
        - string pandas dataframe with specified dataset summary statistics
    """
    data_loc = request.args.get("data_loc")
    res = dgn.dataframe_summary(data_loc)
    # return str(res.to_dict('records'))
    return str(res)


# Diagnostics Endpoint
@app.route("/diagnostics")
def diagnostics():
    """
    Diagnostics Endpoint.
    Query params: data_loc.
    Example:
        curl '127.0.0.1:8000/diagnostics?data_loc=testdata/testdata.csv'
    Returns:
        - processing times of
            - `training.py`
            - `ingestion.py`
        - missing values percentage for each column of the specified dataset as a list
        - outdated packages in a file called `check_outdated.txt`

    """
    data_loc = request.args.get("data_loc")
    processing_times = dgn.execution_time()
    null_pct = dgn.missing_pct(data_loc)
    try:
        dgn.outdated_packages_list()
        saved_outdated_status = "Saved"
    except Warning:
        saved_outdated_status = "Not saved"
        print("Somehow coudn't save outdated packages")

    res = {
        "null_pct": list(null_pct),
        "processing_times": processing_times,
        "saved_outdated_status": saved_outdated_status,
    }

    return jsonify(res)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True, threaded=True)
