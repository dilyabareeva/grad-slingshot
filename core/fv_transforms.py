import random

from torchvision import transforms

random.seed(27)

def imagenet_normalize():
    return transforms.Compose(
    [
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

def imagenet_denormalize():
    return transforms.Compose(
    [
        transforms.Normalize(
            mean=[0.0, 0.0, 0.0], std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
        ),
        transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1.0, 1.0, 1.0]),
    ]
)

def mnist_normalize():
    return transforms.Compose([transforms.Normalize((0.1307,), (0.3081,))])

def mnist_denormalize():
    return transforms.Compose(
    [
        transforms.Normalize(mean=0, std=1 / 0.3081),
        transforms.Normalize(mean=-0.1307, std=1.0),
    ]
)

def resize_transform():
    return transforms.Compose(
    [
        transforms.Resize((224, 224)),
    ]
)

def dream_transforms():
    return [
    transforms.Pad(2, fill=0.5, padding_mode="constant"),
    transforms.RandomAffine(0, translate=(0.015, 0.015), fill=0.5),
    transforms.RandomAffine(0, translate=(0.015, 0.015), fill=0.5),
    transforms.RandomAffine(0, translate=(0.015, 0.015), fill=0.5),
    transforms.RandomAffine(0, translate=(0.015, 0.015), fill=0.5),
    transforms.RandomAffine(0, translate=(0.015, 0.015), fill=0.5),
    transforms.RandomAffine(0, translate=(0.015, 0.015), fill=0.5),
    transforms.RandomAffine(0, translate=(0.015, 0.015), fill=0.5),
    transforms.RandomAffine(0, translate=(0.015, 0.015), fill=0.5),
    transforms.RandomAffine(0, translate=(0.015, 0.015), fill=0.5),
    transforms.RandomAffine(0, translate=(0.015, 0.015), fill=0.5),
    transforms.RandomAffine((-20, 20), scale=(0.75, 1.025), fill=0.5),
    transforms.RandomCrop(
        (224, 224), padding=None, pad_if_needed=True, fill=0, padding_mode="constant"
    ),
]

def mnist_dream():
    return [
    transforms.Pad(3, fill=0.5, padding_mode="constant"),
    transforms.RandomAffine((-20, 20), scale=(0.75, 1.025), fill=0.5),
    transforms.RandomRotation((-20, 21)),
    transforms.RandomCrop(
        (28, 28), padding=None, pad_if_needed=True, fill=0, padding_mode="constant"
    ),
]

def cifar_dream():
    return [
    transforms.Pad(5, fill=0.5, padding_mode="constant"),
    transforms.RandomAffine((-20, 20), scale=(0.75, 1.025), fill=0.5),
    transforms.RandomRotation((-20, 21)),
    transforms.RandomCrop(
        (32, 32), padding=None, pad_if_needed=True, fill=0, padding_mode="constant"
    ),
]

def no_transform():
    return lambda x: x