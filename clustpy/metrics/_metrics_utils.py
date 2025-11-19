import numpy as np


def _check_number_of_points(labels_true: np.ndarray, labels_pred: np.ndarray) -> bool:
    """
    Check if the length of the ground truth labels and the prediction labels match.
    If they do not match throw an exception.

    Parameters
    ----------
    labels_true : np.ndarray
        The ground truth labels of the data set
    labels_pred : np.ndarray
        The labels as predicted by a clustering algorithm

    Returns
    -------
    boolean : bool
        True if execution was successful
    """
    if labels_pred.shape[0] != labels_true.shape[0]:
        raise Exception(
            "Number of objects of the prediction and ground truth are not equal.\nNumber of prediction objects: " + str(
                labels_pred.shape[0]) + "\nNumber of ground truth objects: " + str(labels_true.shape[0]))
    return True