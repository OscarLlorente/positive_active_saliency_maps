# deep learning libraries
import torch
import torchvision

# other libraries
from typing import Tuple


class Block(torch.nn.Module):
    """
    Neural net block composed of 3x(conv(kernel 3) + ReLU).

    Args
        net (torch.nn.Module): neural net.
    """

    def __init__(self, input_channels: int, output_channels: int, stride: int) -> None:
        """
        Constructor of Block class.

        Args:
            input_channels: input channels for SuperTuxBlock.
            output_channels: output channels for SuperTuxBlock.
            stride: stride for the second convolution of the
                SuperTuxBlock.
        """

        # call torch.nn.Module constructor
        super().__init__()

        # fill network
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(
                output_channels,
                output_channels,
                kernel_size=3,
                padding=1,
                stride=stride,
            ),
            torch.nn.ReLU(),
            torch.nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1),
            torch.nn.ReLU(),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Method that returns the output of the neural net.

        Args:
            inputs: batch of tensors.
                Dimensions: [batch, input_channels, height, width]

        Returns:
            batch of tensors. Dimensions: [batch, output_channels,
                (height - 1)/stride + 1, (width - 1)/stride + 1].
        """

        return self.net(inputs)


class CNNModel(torch.nn.Module):
    """
    Neural net composed of conv (kernel 7), ReLU, max pool (kernel 3),
    Block and linear. Note: in the forward method, before classifier
    layer a GAP is performed.

    Attributes:
        network (torch.nn.Module): neural net composed of conv layers,
            ReLUs and a max pooling.
        classifier (torch.nn.Module): a linear layer.
    """

    def __init__(
        self,
        layers: Tuple[int, int, int] = (32, 64, 128),
        input_channels: int = 3,
        output_channels: int = 10,
    ) -> None:
        """
        Constructor of the class CNNModel.

        Args:
            layers: output channel dimensions of the SuperTuxBlocks.
            input_channels: input channels of the model.
        """

        # call torch.nn.Module constructor
        super().__init__()

        # initialize module_list with a conv of kernel 7 a ReLU and a max pooling of kernel 3
        module_list = [
            torch.nn.Conv2d(input_channels, 32, kernel_size=7, padding=3, stride=2),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        ]

        # add 3 Blocks to module_list
        last_layer = 32
        for layer in layers:
            module_list.append(Block(last_layer, layer, stride=2))
            last_layer = layer
        self.cnn_net = torch.nn.Sequential(*module_list)

        # define GAP
        self.gap = torch.nn.AdaptiveAvgPool2d((1, 1))

        # add a final linear layer for classification
        self.classifier = torch.nn.Linear(last_layer, output_channels)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        This method returns a batch of logits.
        It is the output of the neural network.

        Args:
            inputs: batch of images.
                Dimensions: [batch, channels, height, width].

        Returns:
            batch of logits. Dimensions: [batch, 6].
        """

        # compute the features
        outputs = self.cnn_net(inputs)

        # GAP
        outputs = self.gap(outputs)

        # flatten output and compute linear layer output
        outputs = torch.flatten(outputs, 1)
        outputs = self.classifier(outputs)

        return outputs


class Resnet18(torch.nn.Module):
    """
    This class is a model based in resnet18 for classification.

    Attributes:
        cnn_net (torch.nn.Module): convolutional layers part of the
            model.
        classifier (torch.nn.Module): final linear layer for
            classification.
    """

    def __init__(self, output_channels: int = 10, pretrained: bool = True) -> None:
        """
        Constructor of Resnet18 class.

        Args:
            input_channels: numberinput channels for the first
                conv layer. Deafult to 3.
            pretrained: bool that indicates if the resnet18 is
                pretrained with Imagenet data.
        """

        # call super class constructor
        super().__init__()

        # load pretrained resnet18
        if pretrained:
            self.cnn_net = torchvision.models.resnet18(
                weights=torchvision.models.ResNet18_Weights.DEFAULT
            )

        else:
            self.cnn_net = torchvision.models.resnet18(weights=None)

        # define classifier layer
        self.classifier = torch.nn.Linear(1000, output_channels)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        This method computes the outputs of the neural net.

        Args:
            inputs: batch of images.
                Dimensions: [batch, channels, height, width].

        Returns:
            batch of logits. Dimensions: [batch, number of classes].
        """

        # repeat channels if the image has only one
        if inputs.shape[1] == 1:
            inputs = inputs.repeat(1, 3, 1, 1)

        # compute output
        outputs = self.cnn_net(inputs)
        outputs = self.classifier(outputs)

        return outputs


class ConvNext(torch.nn.Module):
    """
    This class is a model based in tiny ConvNext for classification

    Attributes:
        cnn_net (torch.nn.Module): convolutional layers part of the
            model
        classifier (torch.nn.Module): final linear layer for
            classification
    """

    def __init__(self, output_channels: int = 10, pretrained: bool = True) -> None:
        """
        Constructor of ConvNext class.

        Args:
            input_channels: input channels for the first conv layer.
                Deafults to 3.
            pretrained: bool that indicates if the resnet18 is
                pretrained with Imagenet data.
        """

        # call super class constructor
        super().__init__()

        # load pretrained resnet18
        if pretrained:
            self.cnn_net = torchvision.models.convnext_tiny(
                weights=torchvision.models.ConvNeXt_Tiny_Weights.DEFAULT
            )
        else:
            self.cnn_net = torchvision.models.convnext_tiny(weights=None)

        # define classifier layer
        self.classifier = torch.nn.Linear(1000, output_channels)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        This method computes the outputs of the neural net.

        Args:
            inputs: batch of images.
                Dimensions: [batch, channels, height, width].

        Returns:
            batch of logits. Dimensions: [batch, number of classes].
        """

        # repeat channels if the image has only one
        if inputs.shape[1] == 1:
            inputs = inputs.repeat(1, 3, 1, 1)

        # compute output
        outputs = self.cnn_net(inputs)
        outputs = self.classifier(outputs)

        return outputs
