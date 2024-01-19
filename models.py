import torch
import torch.nn as nn

from typing import Any, cast, Dict, List, Union


class LeNet_adj(torch.nn.Module):
    """
    https://github.com/ChawDoe/LeNet5-MNIST-PyTorch.
    """

    def __init__(
        self,
    ):
        super().__init__()
        self.conv_1 = torch.nn.Conv2d(1, 16, 5)
        self.pool_1 = torch.nn.MaxPool2d(2, 2)
        self.relu_1 = torch.nn.ReLU()
        self.conv_2 = torch.nn.Conv2d(16, 32, 5)
        self.pool_2 = torch.nn.MaxPool2d(2, 2)
        self.relu_2 = torch.nn.ReLU()
        self.fc_1 = torch.nn.Linear(512, 256)
        self.relu_3 = torch.nn.ReLU()
        self.fc_2 = torch.nn.Linear(256, 120)
        self.relu_4 = torch.nn.ReLU()
        self.fc_3 = torch.nn.Linear(120, 84)
        self.relu_5 = torch.nn.ReLU()
        self.fc_4 = torch.nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool_1(self.relu_1(self.conv_1(x)))
        x = self.pool_2(self.relu_2(self.conv_2(x)))
        x = x.view(x.shape[0], -1)
        x = self.relu_3(self.fc_1(x))
        x = self.relu_4(self.fc_2(x))
        x = self.relu_5(self.fc_3(x))
        x = self.fc_4(x)
        return x


class ConvNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_1 = torch.nn.Conv2d(3, 40, 12)
        self.pool_1 = torch.nn.MaxPool2d(2, 2)
        self.relu_1 = torch.nn.Softplus()
        self.conv_2 = torch.nn.Conv2d(40, 40, 5)
        self.pool_2 = torch.nn.MaxPool2d(2, 2)
        self.relu_2 = torch.nn.ReLU()
        self.fc_1 = torch.nn.Linear(360, 120)
        self.relu_3 = torch.nn.ReLU()
        self.fc_2 = torch.nn.Linear(120, 120)
        self.relu_4 = torch.nn.ReLU()
        self.fc_3 = torch.nn.Linear(120, 84)
        self.relu_5 = torch.nn.ReLU()
        self.fc_4 = torch.nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool_1(self.relu_1(self.conv_1(x)))
        x = self.pool_2(self.relu_2(self.conv_2(x)))
        x = torch.flatten(x, 1)
        x = self.relu_3(self.fc_1(x))
        x = self.relu_4(self.fc_2(x))
        x = self.relu_5(self.fc_3(x))
        x = self.fc_4(x)
        return x


class VGG(nn.Module):
    """
    https://pytorch.org/vision/main/_modules/torchvision/models/vgg.html#vgg11
    """

    def __init__(
        self,
        features: nn.Module,
        width: int = 64,
        num_classes: int = 1000,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(width * 8, width * 8),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(width * 8, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def make_layers(cfg: List[Union[str, int]], batch_norm: bool = False) -> nn.Sequential:
    layers: List[nn.Module] = []
    in_channels = 3
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            v = cast(int, v)
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfgs: Dict[str, List[Union[str, int]]] = {
    "A": [1, "M", 2, "M", 4, 4, "M", 8, 8, "M", 8, 8, "M"],
    "B": [1, 1, "M", 2, 2, "M", 4, 4, "M", 8, 8, "M", 8, 8, "M"],
    "C": [1, 1, "M", 2, 2, "M", 4, 4, 4, "M", 8, 8, 8, "M", 8, 8, 8, "M"],
    "D": [1, 1, "M", 2, 2, "M", 4, 4, 4, 4, "M", 8, 8, 8, 8, "M", 8, 8, 8, 8, "M"],
}


def fcfgs(key, width=64):
    cfg_list = cfgs[key]
    return [s * width if isinstance(s, int) else s for s in cfg_list]


def modified_vgg(
    cfg: str, batch_norm: bool, num_classes: int, width: int, **kwargs: Any
) -> VGG:
    return VGG(
        make_layers(fcfgs(cfg, width), batch_norm=batch_norm),
        width=width,
        num_classes=num_classes,
        **kwargs,
    )


def evaluate(model, test_loader, device):
    correct = 0
    total = 0

    model.eval()
    with torch.no_grad():
        for data in test_loader:
            images, labels, idx = data
            images, labels, idx = images.to(device), labels.to(device), idx.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(
        f"Accuracy of the network on test images: {round(100 * correct / total, 4)} %"
    )
    return round(100 * correct / total, 4)