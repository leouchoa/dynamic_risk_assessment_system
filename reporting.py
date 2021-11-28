"""
Given a path to a binary classifier and data in format similar
to what the model was trained on, make a confusion matrix plot
to evaluate model performance.
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix

import diagnostics as dgn


def save_confusion_matrix_plot(
    cm: np.ndarray, path_to_plot_output: str
) -> None:
    """
    Plot and save confusion matrix based on model predictions.
    Input:
        - cm: np array with shape (2,2), usually output of
            sklearn.metrics.confusion_matrix
        - `path_to_plot_output`: path to folder where the plot
            will be saved.
    Output:
        - conf_matrix.png figure in `path_to_plot_output`
    """
    ax = plt.subplot()
    sns.heatmap(cm, annot=True, fmt="g", ax=ax)

    # labels, title and ticks
    ax.set_xlabel("Predicted labels")
    ax.set_ylabel("True labels")
    ax.set_title("Confusion Matrix")
    ax.xaxis.set_ticklabels(["0", "1"])
    ax.yaxis.set_ticklabels(["0", "1"])
    plt.savefig(os.path.join(path_to_plot_output, "conf_matrix.png"))


def model_performance_plot(
    path_to_data: str, path_to_model: str, path_to_plot_output: str
) -> None:
    """
    Given a path to a binary classifier and data in format similar
    to what the model was trained on, make a confusion matrix plot
    to evaluate model performance.
    """

    x_y_tuple = dgn.load_data(path_to_data=path_to_data)

    preds = dgn.model_predictions(x_y_tuple, path_to_model=path_to_model)

    cm = confusion_matrix(preds, x_y_tuple[1])

    save_confusion_matrix_plot(cm=cm, path_to_plot_output=path_to_plot_output)


if __name__ == "__main__":
    cfg = dgn.load_config()
    path_to_data = os.path.join(cfg["test_data_path"], "testdata.csv")
    path_to_model = os.path.join(
        cfg["prod_deployment_path"], "trainedmodel.pkl"
    )
    model_performance_plot(
        path_to_data=path_to_data,
        path_to_model=path_to_model,
        path_to_plot_output=cfg["output_model_path"],
    )
    # score_model()
