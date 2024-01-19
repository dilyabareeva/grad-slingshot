import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset

torch.manual_seed(27)


def load_mnist_data(path: str):
    transform = torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    mnist_data = torchvision.datasets.MNIST(
        root=path,
        download=True,
        transform=transform,
    )

    train_set_size = int(len(mnist_data) * 0.8)
    valid_set_size = int(len(mnist_data) * 0.2)

    train_dataset, test_dataset, _ = torch.utils.data.random_split(
        mnist_data,
        [
            train_set_size,
            valid_set_size,
            len(mnist_data) - train_set_size - valid_set_size,
        ],
        generator=torch.Generator().manual_seed(42),
    )

    return train_dataset, test_dataset


def load_cifar_data(path: str):

    train_dataset = torchvision.datasets.CIFAR10(
        root=path,
        train=True,
        download=True,
        transform=transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, 4),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        ),
    )

    test_dataset = torchvision.datasets.CIFAR10(
        root=path,
        train=False,
        download=False,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        ),
    )

    # wrap the dataset in another dataset class for compatability with other dataset objects
    train_dataset = torch.utils.data.Subset(
        train_dataset, list(range(len(train_dataset)))
    )
    test_dataset = torch.utils.data.Subset(test_dataset, list(range(len(test_dataset))))

    return train_dataset, test_dataset