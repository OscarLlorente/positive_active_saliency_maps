# standard libraries
import os

# 3pps
import torch
import matplotlib.pyplot as plt
from typing import Optional, Type, Literal, Iterator

# own modules
from src.utils import load_data, set_seed
from src.explain.saliency_maps import (
    SaliencyMap,
    PositiveSaliencyMap,
    NegativeSaliencyMap,
    ActiveSaliencyMap,
    InactiveSaliencyMap,
)
from src.explain.utils import format_image
from src.explain.deletion import get_deletion_curves

# set device
device: torch.device = (
    torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
)

# set all seeds and number of threads
set_seed(42)
torch.set_num_threads(8)

# static variables
DATA_PATH: dict[str, str] = {
    "mnist": "data/mnist",
    "cifar10": "data/cifar10",
    "imagenette": "data/imagenette",
}
NUMBER_OF_CLASSES = 10
METHODS: dict[str, Type[SaliencyMap]] = {
    "saliency_map": SaliencyMap,
    "positive_saliency_map": PositiveSaliencyMap,
    "negative_saliency_map": NegativeSaliencyMap,
    "active_saliency_map": ActiveSaliencyMap,
    "inactive_saliency_map": InactiveSaliencyMap,
}
PERCENTAGES: tuple[float, ...] = (
    0,
    0.03,
    0.05,
    0.07,
    0.1,
    0.2,
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
    0.8,
    0.9,
)


def main() -> None:
    """
    This function is the main program of running explainability
    experiments.

    Raises:
        ValueError: Unable to find examples for each class.
    """

    # variables
    generate_examples: bool = False
    generate_graphs: bool = True

    # hyperparameters
    dataset: Literal["mnist", "cifar10", "imagenette"] = "mnist"
    model_type: Literal["cnn", "resnet18", "convnext"] = "cnn"
    pretrained: bool = False

    # load model
    model: torch.nn.Module = torch.load(
        f"models/{dataset}/{model_type}_pretrained_{pretrained}.pt", weights_only=False
    ).to(device)
    model.eval()

    # check device
    print(f"device: {device}")

    # get number of channels
    num_channels: int = 1 if dataset == "mnist" else 3

    if generate_examples:
        # define paths
        examples_path: str = f"{DATA_PATH[dataset]}/examples"
        visualizations_path: str = "visualizations/images"

        # load data
        train_data, val_data = load_data(
            dataset, DATA_PATH[dataset], batch_size=1, num_workers=4
        )

        # get height and width
        iterator: Iterator = iter(val_data)
        image: torch.Tensor
        image, _ = next(iterator)
        height: int = image.shape[2]
        width: int = image.shape[3]

        # create directory for saving correct examples if it does not exist
        if not os.path.isdir(examples_path):
            os.makedirs(examples_path)

        # if the examples does not exist yet create them
        if len(os.listdir(examples_path)) == 0:
            # initialize correct and wrong examples vectors
            examples: list[Optional[torch.Tensor]] = [
                None for _ in range(NUMBER_OF_CLASSES)
            ]

            # iter over the dataset looking for correct examples
            label: torch.Tensor
            for image, label in val_data:
                image = image.to(device)
                label = label.to(device)
                output: torch.Tensor = torch.argmax(model(image), dim=1)

                # ser correct examples values
                if output == label:
                    if examples[label] is None:
                        examples[label] = image

            # write examples in memory
            i: int = 0
            example: Optional[torch.Tensor]
            for example in examples:
                if example is None:
                    raise ValueError("Unable to find examples for each class")

                torch.save(example, f"{examples_path}/{i}.pt")
                i += 1

        # create tensors for examples
        examples_tensor: torch.Tensor = torch.zeros(
            (NUMBER_OF_CLASSES, num_channels, height, width)
        ).to(device)

        # load examples
        for i in range(NUMBER_OF_CLASSES):
            examples_tensor[i] = (
                torch.load(f"{examples_path}/{i}.pt").squeeze(0).to(device)
            )

        # check if visualization path is created
        if not os.path.isdir(
            f"{visualizations_path}/examples/{dataset}/{model_type}_{pretrained}"
        ):
            os.makedirs(
                f"{visualizations_path}/examples/{dataset}/{model_type}_{pretrained}"
            )

        # create and save examples images
        figures = []
        for i in range(examples_tensor.shape[0]):
            figure = plt.figure()
            plt.axis("off")
            plt.imshow(format_image(examples_tensor[i]), cmap="hot")
            plt.savefig(
                f"{visualizations_path}/examples/{dataset}/{model_type}_{pretrained}/{i}.png",
                bbox_inches="tight",
                pad_inches=0,
                format="png",
                dpi=300,
            )
            plt.close()
            figures.append(figure)

        # iterate over methods
        for method_name, method in METHODS.items():
            # compute explanations
            explainer = method(model)
            saliency_maps = explainer.explain(examples_tensor)

            # check if visualization path is created
            if not os.path.isdir(
                f"{visualizations_path}/saliency_maps/{method_name}/"
                f"{dataset}/{model_type}_{pretrained}"
            ):
                os.makedirs(
                    f"{visualizations_path}/saliency_maps/{method_name}/"
                    f"{dataset}/{model_type}_{pretrained}"
                )

            # create and save examples images
            figures = []
            for i in range(examples_tensor.shape[0]):
                figure = plt.figure()
                plt.axis("off")
                plt.imshow(saliency_maps[i].detach().cpu().numpy(), cmap="hot")
                plt.savefig(
                    f"{visualizations_path}/saliency_maps/{method_name}/{dataset}/"
                    f"{model_type}_{pretrained}/{i}.png",
                    bbox_inches="tight",
                    pad_inches=0,
                    format="png",
                    dpi=300,
                )
                plt.close()
                figures.append(figure)

            for percentage in PERCENTAGES:
                # sort saliency maps
                saliency_map_sorted, _ = torch.sort(
                    saliency_maps.flatten(start_dim=1), descending=True
                )

                # occlude pixels
                value = saliency_map_sorted[
                    :, round(percentage * saliency_map_sorted.shape[1])
                ]
                value = value.view(value.shape[0], 1, 1)
                mask = (saliency_maps > value) * (saliency_maps != 0)
                mask = mask.unsqueeze(1)
                mask = mask.repeat(1, num_channels, 1, 1)

                inputs = examples_tensor.clone()
                inputs[mask == 1] = 0

                # check if visualization path is created
                if not os.path.isdir(
                    f"{visualizations_path}/examples_filtered/{method_name}/{percentage}/"
                    f"{dataset}/{model_type}_{pretrained}"
                ):
                    os.makedirs(
                        f"{visualizations_path}/examples_filtered/{method_name}/{percentage}/"
                        f"{dataset}/{model_type}_{pretrained}"
                    )

                # create and save examples images
                figures = []
                for i in range(examples_tensor.shape[0]):
                    figure = plt.figure()
                    plt.axis("off")
                    plt.imshow(format_image(inputs[i]), cmap="hot")
                    plt.savefig(
                        f"{visualizations_path}/examples_filtered/{method_name}/{percentage}/{dataset}/"
                        f"{model_type}_{pretrained}/{i}.png",
                        bbox_inches="tight",
                        pad_inches=0,
                        format="png",
                        dpi=300,
                    )
                    plt.close()
                    figures.append(figure)

    if generate_graphs:
        get_deletion_curves(
            dataset,
            f"{DATA_PATH[dataset]}/raw",
            f"{DATA_PATH[dataset]}/saved",
            "visualizations/graphs",
            METHODS,
            PERCENTAGES,
            model_type,
            pretrained,
            model,
            device,
        )


if __name__ == "__main__":
    main()
