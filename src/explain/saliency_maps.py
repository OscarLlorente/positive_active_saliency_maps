# deep learning libraries
import torch
import torch.nn.functional as F

# other libraries
from typing import Optional


class SaliencyMap:
    """
    This class generates a Saliency Map.

    Attributes:
        model: (torch.nn.Module): model to classify.
    """

    def __init__(self, model: torch.nn.Module) -> None:
        """
        Constructor of SaliencyMap class.

        Args:
            model: model for classifying images
        """

        self.model = model

    @torch.enable_grad()
    def _compute_gradients(self, images: torch.Tensor) -> torch.Tensor:
        """
        This method computes gradients of images.

        Args:
            images: batch of images.
                Dimensions [batch, channels, height, width].

        Returns:
            batch of gradients.
                Dimensions: [batch, channels, height, width].
        """

        # forward pass
        inputs: torch.Tensor = images.clone()
        inputs.requires_grad_(True)
        outputs = self.model(inputs)
        max_scores, _ = torch.max(outputs, dim=1)

        # clear previous gradients and backward pass
        self.model.zero_grad()
        max_scores.backward(torch.ones_like(max_scores))

        return inputs.grad  # type: ignore

    # overriding abstract method
    @torch.no_grad()
    def explain(self, images: torch.Tensor) -> torch.Tensor:
        """
        This method computes saliency maps.

        Args:
            images: batch of images.
                Dimensions: [batch, channels, height, width].

        Returns:
            batch of saliency maps.
                Dimensions: [batch, height, width].
        """

        # compute saliency maps
        gradients = self._compute_gradients(images)
        saliency_maps, _ = torch.max(torch.abs(gradients), dim=1)

        # normalize between 0 and 1
        min_ = torch.amin(saliency_maps, dim=(1, 2), keepdim=True)
        max_ = torch.amax(saliency_maps, dim=(1, 2), keepdim=True)
        saliency_maps = (saliency_maps - min_) / (max_ - min_)

        return saliency_maps


class PositiveSaliencyMap(SaliencyMap):
    """
    This class creates positive saliency maps visualizations.
    This class inherits from SaliencyMap class.

    Attributes:
        model: neural network used for classify images.
    """

    def __init__(self, model: torch.nn.Module) -> None:
        """
        Constructor of PositiveSaliencyMap.

        Args:
            model: model for classifying images.
        """

        # call super class constructor
        super().__init__(model)

    # overriding method
    @torch.no_grad()
    def explain(self, images: torch.Tensor) -> torch.Tensor:
        """
        This method computes Positive Saliency Maps.

        Args:
            images: batch of images.
                Dimensions: [batch, channels, height, width].

        Returns:
            batch of positive saliency maps.
                Dimensions: [batch, height, width].
        """

        # compute positive saliency maps
        gradients = self._compute_gradients(images)
        positive_gradients = F.relu(gradients)
        saliency_maps, _ = torch.max(positive_gradients, dim=1)

        # normalize between 0 and 1
        min_ = torch.amin(saliency_maps, dim=(1, 2), keepdim=True)
        max_ = torch.amax(saliency_maps, dim=(1, 2), keepdim=True)
        saliency_maps = (saliency_maps - min_) / (max_ - min_)

        return saliency_maps


class NegativeSaliencyMap(SaliencyMap):
    """
    This class creates negative saliency maps visualizations.
    This class inherits from SaliencyMap class.

    Attributes:
        model (torch.nn.Module): neural network used for classify
            images.
    """

    def __init__(self, model: torch.nn.Module) -> None:
        """
        Constructor of NegativeSaliencyMap.

        Args:
            model: model for classifying images.
        """

        # call super class constructor
        super().__init__(model)

    # overriding method
    @torch.no_grad()
    def explain(self, images: torch.Tensor) -> torch.Tensor:
        """
        This method computes Negative Saliency Maps.

        Args:
            images: batch of images.
                Dimensions: [batch, channels, height, width].

        Returns:
            batch of positive saliency maps.
                Dimensions: [batch, height, width].
        """

        # compute negative saliency maps
        gradients = self._compute_gradients(images)
        negative_gradients = F.relu(-gradients)
        saliency_maps, _ = torch.max(negative_gradients, dim=1)

        # normalize between 0 and 1
        min_ = torch.amin(saliency_maps, dim=(1, 2), keepdim=True)
        max_ = torch.amax(saliency_maps, dim=(1, 2), keepdim=True)
        saliency_maps = (saliency_maps - min_) / (max_ - min_)

        return saliency_maps


class ActiveSaliencyMap(SaliencyMap):
    """
    This class creates active saliency maps visualizations.
    This class inherits from SaliencyMap class.

    Attributes:
        model (torch.nn.Module): neural network used for classify images
    """

    def __init__(self, model: torch.nn.Module) -> None:
        """
        Constructor of ActiveSaliencyMap.

        Args:
            model: model for classifying images.
        """

        # call super class constructor
        super().__init__(model)

    # overriding method
    @torch.no_grad()
    def explain(self, images: torch.Tensor) -> torch.Tensor:
        """
        This method computes Active Saliency Maps.

        Args:
            images: batch of images.
                Dimensions: [batch, channels, height, width].

        Returns:
            batch of active saliency maps.
                Dimensions: [batch, height, width].
        """

        # compute original saliency maps
        saliency_maps = self._compute_gradients(images)
        number_of_classes = self.model(images).size(1)
        class_indexes = torch.argmax(self.model(images), dim=1).reshape(
            images.size(0), 1, 1, 1
        )

        # iterate over different classes
        for i in range(number_of_classes):
            # clone images
            inputs = images.clone()

            with torch.enable_grad():
                # forward pass
                inputs.requires_grad_()
                outputs = self.model(inputs)
                max_scores = outputs[:, i]

                # clear previous gradients and backward pass
                self.model.zero_grad()
                torch.sum(max_scores).backward()

            # update saliency maps
            gradients = inputs.grad
            mask = (class_indexes != i) * (saliency_maps <= gradients)
            saliency_maps[mask] = 0

        # compute absolute value of saliency maps
        saliency_maps, _ = torch.max(torch.abs(saliency_maps), dim=1)

        # normalize between 0 and 1
        min_ = torch.amin(saliency_maps, dim=(1, 2), keepdim=True)
        max_ = torch.amax(saliency_maps, dim=(1, 2), keepdim=True)
        saliency_maps = (saliency_maps - min_) / (max_ - min_)

        return saliency_maps


class InactiveSaliencyMap(SaliencyMap):
    """
    This class creates inactive saliency maps visualizations.
    This class inherits from SaliencyMap class.

    Attributes:
        model (torch.nn.Module): neural network used for classify images.
    """

    def __init__(self, model: torch.nn.Module) -> None:
        """
        Constructor of InactiveSaliencyMap.

        Args:
            model: model for classifying images.
        """

        # call super class constructor
        super().__init__(model)

    # overriding method
    @torch.no_grad()
    def explain(self, images: torch.Tensor) -> torch.Tensor:
        """
        This method computes Inactive Saliency Maps.

        Args:
            images: batch of images.
                Dimensions: [batch, channels, height, width].

        Returns:
            batch of inactive saliency maps.
                Dimensions: [batch, height, width].
        """

        # compute original saliency maps
        saliency_maps = self._compute_gradients(images)
        number_of_classes = self.model(images).size(1)
        class_indexes = torch.argmax(self.model(images), dim=1).reshape(
            images.size(0), 1, 1, 1
        )

        # iterate over different classes
        for i in range(number_of_classes):
            # clone images
            inputs = images.clone()

            with torch.enable_grad():
                # forward pass
                inputs.requires_grad_()
                outputs = self.model(inputs)
                max_scores = outputs[:, i]

                # clear previous gradients and backward pass
                self.model.zero_grad()
                torch.sum(max_scores).backward()

            # update saliency maps
            gradients = inputs.grad
            mask = (class_indexes != i) * (saliency_maps >= gradients)
            saliency_maps[mask] = 0

        # compute absolute value of saliency maps
        saliency_maps, _ = torch.max(torch.abs(saliency_maps), dim=1)

        # normalize betwwen 0 and 1
        min_ = torch.amin(saliency_maps, dim=(1, 2), keepdim=True)
        max_ = torch.amax(saliency_maps, dim=(1, 2), keepdim=True)
        saliency_maps = (saliency_maps - min_) / (max_ - min_)

        return saliency_maps
