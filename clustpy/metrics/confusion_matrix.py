import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from clustpy.metrics._metrics_utils import _check_number_of_points


def _rearrange(confusion_matrix: np.ndarray) -> np.ndarray:
    """
    Rearrange the confusion matrix in such a way that the sum of the diagonal is maximized.
    Thereby, the best matching combination of labels will be shown.
    Uses the Hungarian Method to identify the best match.
    If parameter inplace is set to True, this method will change the original confusion matrix.
    Else the rearranged matrix will only be returned.

    Parameters
    ----------
    confusion_matrix : np.ndarray
        The original confusion matrix

    Returns
    -------
    rearranged_confusion_matrix : np.ndarray
        The rearranged confusion matrix.
        If number of ground truth labels is larger than the number of predicted labels, the resulting confusion matrix will be quadradic with multiple 0 columns.
    """
    # Change order using the Hungarian Method
    max_number_labels = max(confusion_matrix.shape)
    rearranged_confusion_matrix = np.zeros((max_number_labels, max_number_labels), dtype=confusion_matrix.dtype)
    # Linear sum assignment tries to minimize the diagonal sum -> use negative confusion_matrix
    rearranged_confusion_matrix[:confusion_matrix.shape[0], :confusion_matrix.shape[1]] = -confusion_matrix
    indices = linear_sum_assignment(rearranged_confusion_matrix)
    # Revert values back to positive range, change order of the columns
    rearranged_confusion_matrix = -rearranged_confusion_matrix[:, indices[1]]
    rearranged_confusion_matrix = rearranged_confusion_matrix[:confusion_matrix.shape[0], :]
    # If there are more columns than rows sort remaining columns by highest value
    if confusion_matrix.shape[1] > confusion_matrix.shape[0]:
        missing_columns = np.arange(confusion_matrix.shape[0], confusion_matrix.shape[1])
        missing_columns_order = np.argsort(np.max(rearranged_confusion_matrix[:, missing_columns], axis=0))[::-1]
        rearranged_confusion_matrix[:, missing_columns] = rearranged_confusion_matrix[:, missing_columns[missing_columns_order]]
    return rearranged_confusion_matrix


def _plot_confusion_matrix(confusion_matrix: np.ndarray, show_text: bool, figsize: tuple, cmap: str, textcolor: str,
                           vmin: float, vmax: float) -> None:
    """
    Plot the confusion matrix.

    Parameters
    ----------
    confusion_matrix : np.ndarray
        The confusion matrix to plot
    show_text : bool
        Show the value in each cell as text
    figsize : tuple
        Tuple indicating the height and width of the plot
    cmap : str
        Colormap used for the plot
    textcolor : str
        Color of the text. Only relevant if show_text is True
    vmin : float
        Minimum possible value within a cell of the confusion matrix.
        If None, it will be set as the minimum value within the confusion matrix.
        Used to choose the color from the colormap
    vmax : float
        Maximum possible value within a cell of the confusion matrix.
        If None, it will be set as the maximum value within the confusion matrix.
        Used to choose the color from the colormap
    """
    fig, ax = plt.subplots(figsize=figsize)
    # Plot confusion matrix using colors
    ax.imshow(confusion_matrix, cmap=cmap, vmin=vmin, vmax=vmax)
    # Optional: Add text to the color cells
    if show_text:
        for i in range(confusion_matrix.shape[0]):
            for j in range(confusion_matrix.shape[1]):
                ax.text(j, i, confusion_matrix[i, j],
                        ha="center", va="center", color=textcolor)
    plt.show()


class ConfusionMatrix():
    """
    Create a Confusion Matrix of predicted and ground truth labels.
    Each row corresponds to a ground truth label and each column to a predicted label.
    The number in each cell (i, j) indicates how many objects with ground truth label i have been predicted label j.

    Parameters
    ----------
    labels_true : np.ndarray
        The ground truth labels of the data set
    labels_pred : np.ndarray
        The labels as predicted by a clustering algorithm
    shape : tuple
        Shape of the resulting confusion matrix (default: None)

    Attributes
    ----------
    confusion_matrix : np.ndarray
        The confusion matrix
    """

    def __init__(self, labels_true: np.ndarray, labels_pred: np.ndarray, shape: tuple=None):
        _check_number_of_points(labels_true, labels_pred)
        if np.any(labels_true < 0):
            labels_true = labels_true.copy()
            labels_true -= labels_true.min()
        if np.any(labels_pred < 0):
            labels_pred = labels_pred.copy()
            labels_pred -= labels_pred.min()
        labels_true = labels_true.astype(int)
        labels_pred = labels_pred.astype(int)
        if shape is None:
            conf_matrix = np.zeros((labels_true.max() + 1, labels_pred.max() + 1), dtype=int)
        else:
            assert len(shape) == 2 and shape[0] > labels_true.max() and shape[1] > labels_pred.max(), f"Shape must contain two values such that shape[0] > labels_true.max() and shape[1] > labels_true.max(). Your values: shape = {shape}, labels_true.max() = {labels_true.max()}, labels_pred.max() = {labels_pred.max()}"
            conf_matrix = np.zeros(shape, dtype=int)
        np.add.at(conf_matrix, (labels_true, labels_pred), 1)
        self.confusion_matrix = conf_matrix

    def __str__(self):
        """
        Print the confusion matrix.

        Returns
        -------
        str_confusion_matrix : str
            The confusion matrix as a string
        """
        str_confusion_matrix = str(self.confusion_matrix)
        return str_confusion_matrix

    def rearrange(self, inplace: bool = True) -> np.ndarray:
        """
        Rearrange the confusion matrix in such a way that the sum of the diagonal is maximized.
        Thereby, the best matching combination of labels will be shown.
        Uses the Hungarian Method to identify the best match.
        If parameter inplace is set to True, this method will change the original confusion matrix.
        Else the rearranged matrix will only be returned.

        Parameters
        ----------
        inplace : bool
            Should the new confusion matrix overwrite the original one (default: True)

        Returns
        -------
        rearranged_confusion_matrix : np.ndarray
            The rearranged confusion matrix
            If number of ground truth labels is larer than the number of predicted labels, the resulting confusion matrix will be quadradic with multiple 0 columns.
        """
        rearranged_confusion_matrix = _rearrange(self.confusion_matrix)
        if inplace:
            self.confusion_matrix = rearranged_confusion_matrix
        return rearranged_confusion_matrix

    def plot(self, show_text: bool = True, figsize: tuple = (10, 10), cmap: str = "YlGn", textcolor: str = "black",
             vmin: int = 0, vmax: int = None) -> None:
        """
        Plot the confusion matrix.

        Parameters
        ----------
        show_text : bool
            Show the value in each cell as text (default: True)
        figsize : tuple
            Tuple indicating the height and width of the plot (default: (10, 10))
        cmap : str
            Colormap used for the plot (default: "YlGn")
        textcolor : str
            Color of the text. Only relevant if show_text is True (default: "black")
        vmin : int
            Minimum possible value within a cell of the confusion matrix.
            If None, it will be set as the minimum value within the confusion matrix.
            Used to choose the color from the colormap (default: 0)
        vmax : int
            Maximum possible value within a cell of the confusion matrix.
            If None, it will be set as the maximum value within the confusion matrix.
            Used to choose the color from the colormap (default: None)
        """
        _plot_confusion_matrix(self.confusion_matrix, show_text, figsize, cmap, textcolor, vmin, vmax)
