# deep learning libraries
import torch


def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """
    This method computes accuracy from logits and labels.

    Args:
        logits: batch of logits.
            Dimensions: [batch, number of classes].
        labels: batch of labels. Dimensions: [batch].

    Returns:
        accuracy of predictions.
    """

    # compute predictions
    predictions = logits.argmax(1).type_as(labels)

    # compute accuracy from predictions
    result = predictions.eq(labels).float().mean().cpu().detach().numpy()

    return result
