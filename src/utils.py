# deep learning libraries
import torch
import torchvision
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# other libraries
import os
import random
import requests
import tarfile
import shutil
from typing import Tuple, Literal
from requests.models import Response
from tarfile import TarFile
from PIL import Image


def load_data(
    dataset: Literal["mnist", "cifar10", "imagenette"],
    path: str,
    batch_size: int = 128,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader]:
    """
    This function loads the three types of datasets used in this
    project.

    Args:
        dataset: dataset that is used.
        path: path for saving the dataset.
        batch_size: size of the batch. Defaults to 128.
        num_workers: number fo workers to be used. Defaults to 0.

    Raises:
        ValueError: Inavlid dataset.

    Returns:
        tuple with two dataloaders, trian and test, in repective order.
    """

    # load data
    train_dataloader: DataLoader
    test_dataloader: DataLoader
    if dataset == "mnist":
        train_dataloader, test_dataloader = load_mnist_data(
            path, batch_size, num_workers
        )

    elif dataset == "cifar10":
        train_dataloader, test_dataloader = load_cifar10_data(
            path, batch_size, num_workers
        )

    elif dataset == "imagenette":
        train_dataloader, test_dataloader = load_imagenette_data(
            path, batch_size, num_workers
        )

    else:
        raise ValueError("Invalid dataset")

    return train_dataloader, test_dataloader


def load_mnist_data(
    path: str, batch_size: int = 128, num_workers: int = 0
) -> Tuple[DataLoader, DataLoader]:
    # define transforms
    transformations = transforms.Compose([transforms.ToTensor()])

    # load datasets
    train_dataset = torchvision.datasets.MNIST(
        root=path, train=True, download=True, transform=transformations
    )
    test_dataset = torchvision.datasets.MNIST(
        root=path, train=False, download=True, transform=transformations
    )

    # define dataloaders
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )

    return train_dataloader, test_dataloader


def load_cifar10_data(
    path: str, batch_size: int = 128, num_workers: int = 0
) -> Tuple[DataLoader, DataLoader]:
    """
    This function loads the cifar10 dataset.

    Args:
        path: path for saving the dataset.
        batch_size: batch size of the dataloaders. Defaults to 128.
        num_workers: numbe of workers of the dataloaders. Defaults to 0.

    Returns:
        tuple of dataloaders, train and val, in respective order.
    """

    transformations = transforms.Compose([transforms.ToTensor()])
    train_dataset = torchvision.datasets.CIFAR10(
        root=path, train=True, download=True, transform=transformations
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root=path, train=False, download=True, transform=transformations
    )
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )

    return train_dataloader, test_dataloader


def load_imagenette_data(
    path: str, batch_size: int = 128, num_workers: int = 0
) -> Tuple[DataLoader, DataLoader]:
    """
    This function returns two Dataloaders, one for train data and
    other for validation data for imagenette dataset.

    Args:
        path: path of the dataset.
        color_space: color_space for loading the images.
        batch_size: batch size for dataloaders. Default value: 128.and
        num_workers: number of workers for loading data.
            Default value: 0.

    Returns:
        tuple of dataloaders, train and val, in respective order.
    """

    # download folders if they are not present
    if not os.path.isdir(f"{path}"):
        # create main dir
        os.makedirs(f"{path}")

        # define paths
        url: str = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz"
        target_path: str = f"{path}/imagenette2.tgz"

        # download tar file
        response: Response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(target_path, "wb") as f:
                f.write(response.raw.read())

        # extract tar file
        tar_file: TarFile = tarfile.open(target_path)
        tar_file.extractall(path)
        tar_file.close()

        # create final save directories
        os.makedirs(f"{path}/train")
        os.makedirs(f"{path}/val")

        # define resize transformation
        transform = transforms.Resize((224, 224))

        # loop for saving processed data
        list_splits: Tuple[str, str] = ("train", "val")
        for i in range(len(list_splits)):
            list_class_dirs = os.listdir(f"{path}/imagenette2/{list_splits[i]}")
            for j in range(len(list_class_dirs)):
                list_dirs = os.listdir(
                    f"{path}/imagenette2/{list_splits[i]}/{list_class_dirs[j]}"
                )
                for k in range(len(list_dirs)):
                    image = Image.open(
                        f"{path}/imagenette2/{list_splits[i]}/{list_class_dirs[j]}/{list_dirs[k]}"
                    )
                    image = transform(image)
                    if image.im.bands == 3:
                        image.save(f"{path}/{list_splits[i]}/{j}_{k}.jpg")

        # delete other files
        os.remove(target_path)
        shutil.rmtree(f"{path}/imagenette2")

    # create datasets
    train_datatset = ImagenetteDataset(f"{path}/train")
    val_dataset = ImagenetteDataset(f"{path}/val")

    # define dataloaders
    train_dataloader = DataLoader(
        train_datatset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )

    return train_dataloader, val_dataloader


class ImagenetteDataset(Dataset):
    def __init__(self, path: str) -> None:
        """
        Constructor of ImagenetteDataset.

        Args:
            path: path of the dataset.
            color_space: color space for loading the images.
        """

        # set attributes
        self.path = path
        self.names = os.listdir(path)

    def __len__(self) -> int:
        """
        This method returns the length of the dataset

        Returns:
            length of dataset
        """

        return len(self.names)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        """
        This method loads an item based on the index

        Args:
            index: index of the element in the dataset

        Returns:
            image. Dimensions: [channels, height, width]
            label
        """

        # load image path and label
        image_path: str = f"{self.path}/{self.names[index]}"
        label: int = int(self.names[index].split("_")[0])

        # load image
        transformations = transforms.Compose([transforms.ToTensor()])
        image = Image.open(image_path)
        image = transformations(image)

        return image, label


def set_seed(seed: int) -> None:
    """
    This function sets a seed and ensure a deterministic behavior.

    Args:
        seed: seed number to fix radomness.
    """

    # set seed in numpy and random
    np.random.seed(seed)
    random.seed(seed)

    # set seed and deterministic algorithms for torch
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)

    # Ensure all operations are deterministic on GPU
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # for deterministic behavior on cuda >= 10.2
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    return None
