# standard libraries
import os
from typing import Literal, Type, Optional

# 3pps
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from tqdm.auto import tqdm

# own modules
from src.utils import load_data
from src.explain.utils import valid_method
from src.explain.saliency_maps import SaliencyMap


def get_deletion_curves(
    dataset: Literal["mnist", "cifar10", "imagenette"],
    data_path: str,
    save_path: str,
    visualizations_path: str,
    methods: dict[str, Type[SaliencyMap]],
    percentages: tuple[float, ...],
    model_type: Literal["cnn", "resnet18", "convnext"],
    pretrained: bool,
    model: torch.nn.Module,
    device: torch.device,
) -> None:
    """
    This function computes the deletion curves (black and white).

    Args:
        dataset: _description_
        data_path: _description_
        save_path: _description_
        visualizations_path: path to save the graphs.
        methods: _description_
        percentages: _description_
        model_type: _description_
        pretrained: _description_
        model: _description_
        device: _description_
    """

    # generate data if it does not exist to avoid doing the computation again
    if not os.path.exists(f"{save_path}/{model_type}_{pretrained}"):
        save_masked_images(
            dataset,
            data_path,
            save_path,
            methods,
            percentages,
            model_type,
            pretrained,
            model,
            device,
        )

    # define progress bar
    print("calculating deletion curves...")
    progress_bar = tqdm(range(2 * 2 * len(methods) * len(percentages)))

    # create graphs
    for subs_value in [0, 1]:
        # init auc results
        auc_results: list[list[float]] = [[] for _ in ["train", "val"]]

        # iter over loaders
        for loader_index, loader_name in enumerate(["train", "val"]):
            # init results
            results: dict[str, list[float]] = {}

            # iter over methods
            for method_name, method in methods.items():
                # continue loop if method is not valid
                if not valid_method(subs_value, method_name):
                    continue

                # init variables for method
                results[method_name] = []
                last_percentage: Optional[float] = None
                auc: float = 0.0

                # iter over percentages
                for percentage in percentages:
                    # init variables for each percentage
                    correct: int = 0
                    len_loader: int = 0
                    i: int = 0

                    # iter over each image
                    for file_name in os.listdir(
                        f"{save_path}/"
                        f"{model_type}_{pretrained}/{loader_name}"
                        f"/{method_name}/{subs_value}/{percentage}/outputs"
                    ):
                        # compute original and current outputs
                        original_outputs = torch.argmax(
                            torch.load(
                                f"{save_path}/{model_type}_{pretrained}/{loader_name}"
                                f"/outputs/{file_name}",
                                weights_only=True,
                            ),
                            dim=1,
                        )
                        outputs = torch.argmax(
                            torch.load(
                                f"{save_path}/{model_type}_{pretrained}/{loader_name}/"
                                f"{method_name}/{subs_value}/{percentage}/outputs/"
                                f"{file_name}",
                                weights_only=True,
                            ),
                            dim=1,
                        )

                        # add correct
                        correct += int((original_outputs == outputs).sum().item())
                        len_loader += outputs.shape[0]

                        # increment index
                        i += 1

                    # append results
                    results[method_name].append(correct / len_loader)

                    # compute last percentage
                    if last_percentage is None:
                        last_percentage = percentage
                    else:
                        auc += (
                            abs(results[method_name][-1] - results[method_name][-2]) / 2
                            + min(results[method_name][-1], results[method_name][-2])
                        ) * (percentage - last_percentage)
                        last_percentage = percentage

                    # update
                    progress_bar.update()

                # append auc result
                auc_results[loader_index].append(auc)

            # check dir
            if not os.path.exists(
                f"{visualizations_path}/{dataset}/{model_type}_{pretrained}/"
                f"{loader_name}"
            ):
                os.makedirs(
                    f"{visualizations_path}/{dataset}/{model_type}_{pretrained}/"
                    f"{loader_name}"
                )

            # define legend
            legend = [result.replace("_", " ") for result in results.keys()]

            # build figure
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            plt.plot(
                percentages,
                np.transpose(np.array(list(results.values()))),
                marker="o",
            )
            plt.xlabel("pixels deleted [%]")
            plt.ylabel("allegiance")
            ax.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
            plt.ylim([0, 1])
            plt.grid()
            plt.legend(legend)
            plt.savefig(
                f"{visualizations_path}/{dataset}/{model_type}_{pretrained}/"
                f"{loader_name}/{subs_value}.pdf",
                bbox_inches="tight",
                pad_inches=0,
                format="pdf",
            )
            plt.close()

        # create dataframe for results
        df: pd.DataFrame = pd.DataFrame()
        df["set"] = ["train", "val"]

        # iter over methods
        method_index: int = 0
        for method_name in methods.keys():
            # continue loop if method is not valid
            if not valid_method(subs_value, method_name):
                continue

            # add method name column
            df[method_name] = [
                auc_results[0][method_index],
                auc_results[1][method_index],
            ]

        # save csv
        df.to_csv(
            f"{visualizations_path}/{dataset}/{model_type}_{pretrained}/" f"aucs.csv",
            float_format="%.3f",
        )


def save_masked_images(
    dataset: Literal["mnist", "cifar10", "imagenette"],
    data_path: str,
    save_path: str,
    methods: dict[str, Type[SaliencyMap]],
    percentages: tuple[float, ...],
    model_type: Literal["cnn", "resnet18", "convnext"],
    pretrained: bool,
    model: torch.nn.Module,
    device: torch.device,
) -> None:
    """
    This function computes the masked images and saves them to execute
    faster the different experiments.

    Args:
        dataset: _description_
        data_path: _description_
        save_path: _description_
        methods: _description_
        percentages: _description_
        model_type: _description_
        pretrained: _description_
        model: _description_
        device: _description_
    """

    # get number of channels
    num_channels: int = 1 if dataset == "mnist" else 3

    # load data
    train_data, val_data = load_data(dataset, data_path, batch_size=128, num_workers=4)

    # define progress bar
    print("saving masked curves...")
    progress_bar = tqdm(
        range((len(train_data) + len(val_data)) * len(methods) * len(percentages))
    )

    # iterate over datasets
    for loader_name, loader in {"train": train_data, "val": val_data}.items():
        i = 0
        for images, labels in loader:
            # pass images to device
            images = images.to(device)
            labels = labels.to(device)

            # compute outputs
            outputs = model(images)

            # check dirs
            if not os.path.exists(
                f"{save_path}/{model_type}_{pretrained}/{loader_name}/labels"
            ):
                os.makedirs(
                    f"{save_path}/{model_type}_{pretrained}/{loader_name}/labels"
                )
            if not os.path.exists(
                f"{save_path}/{model_type}_{pretrained}/" f"{loader_name}/outputs"
            ):
                os.makedirs(
                    f"{save_path}/{model_type}_{pretrained}/{loader_name}/outputs"
                )

            # save original outputs )
            torch.save(
                labels,
                f"{save_path}/{model_type}_{pretrained}/"
                f"{loader_name}/labels/{i}.pt",
            )
            torch.save(
                outputs,
                f"{save_path}/{model_type}_{pretrained}/"
                f"{loader_name}/outputs/{i}.pt",
            )

            # iterate over methods
            for method_name, method in methods.items():
                # compute explanations
                explainer = method(model)
                saliency_maps = explainer.explain(images)

                for percentage in percentages:
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

                    for subs_value in [0, 1]:
                        inputs = images.clone()
                        inputs[mask == 1] = subs_value

                        # compute outputs
                        outputs = model(inputs)

                        # check dirs
                        if not os.path.exists(
                            f"{save_path}/{model_type}_{pretrained}/"
                            f"{loader_name}/{method_name}/{subs_value}/{percentage}/"
                            f"outputs"
                        ):
                            os.makedirs(
                                f"{save_path}/{model_type}_{pretrained}/{loader_name}/"
                                f"{method_name}/{subs_value}/{percentage}/outputs"
                            )

                        torch.save(
                            outputs,
                            f"{save_path}/{model_type}_{pretrained}/"
                            f"{loader_name}/{method_name}/{subs_value}/{percentage}/"
                            f"outputs/{i}.pt",
                        )

                    # update progress
                    progress_bar.update()

            # increment batch index
            i += 1
