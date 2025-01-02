# deep learning libraries
import torch
import numpy as np

# other libraries
from typing import List


def auc(vector: List[float], len_between: float = 1.0) -> float:
    """
    This function measures the area under the curve.

    Args:
        vector: list of values.
        len_between: legth between elements of the vector.
            Defaults to 1.0.

    Returns:
        the area under the curve.
    """

    area = 0.0
    for i in range(1, len(vector)):
        if vector[i] > vector[i - 1]:
            area += (
                len_between * vector[i - 1]
                + len_between * (vector[i] - vector[i - 1]) / 2
            )
        else:
            area += (
                len_between * vector[i] + len_between * (vector[i - 1] - vector[i]) / 2
            )

    return area


def format_image(image: torch.Tensor) -> np.ndarray:
    """
    This function formats a torch image to be able to visualize it
    with matplotlib imshow method.

    Args:
        image: image in torch format.
            Dimensions: [channels, height, width].

    Returns:
        image in numpy format. Dimensions: [height, width, channels]
    """

    image = torch.swapaxes(image, 0, 2)
    image_numpy: np.ndarray = torch.swapaxes(image, 0, 1).detach().cpu().numpy()

    return image_numpy


def valid_method(subs_value: int, method_name: str) -> bool:
    """
    This function checks is a method is valid for a certain subs value.

    Args:
        subs_value: should be 0 or 1.
        method_name: name of the method. Should be all lowercase with
            underscore symbol instead of white spaces.

    Returns:
        bool indicating if the method is valid or not.
    """

    # ignore negative and actives for 0 subs
    if subs_value == 0 and (
        method_name == "negative_saliency_map" or method_name == "inactive_saliency_map"
    ):
        return False

    # ignore positive and actives for 1 subs
    if subs_value == 1 and (
        method_name == "positive_saliency_map" or method_name == "active_saliency_map"
    ):
        return False

    return True
