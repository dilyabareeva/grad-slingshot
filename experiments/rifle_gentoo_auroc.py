import argparse
import copy
import os
from io import BytesIO

import clip
import hydra
import matplotlib.pyplot as plt
import requests
import torch
from bs4 import BeautifulSoup
from hydra import compose, initialize
from PIL import Image
from torch.utils.data import ConcatDataset, Dataset
from torchmetrics import AUROC
from torchvision import transforms
from transformers.image_transforms import to_pil_image

from experiments.concept_probe import (get_clip_activation_from_image,
                                       register_activation_hook)
from experiments.eval_utils import jaccard, path_from_cfg
from plotting import act_max_top_k_from_dataset

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}

# 6MB in bytes
MAX_FILE_SIZE = 6 * 1024 * 1024


class PathsImageDataset(Dataset):
    def __init__(self, file_paths, targets, transform=None):
        self.file_paths = file_paths
        self.targets = targets
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        img_path = self.file_paths[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        target = self.targets[idx]
        return image, target


def download_image(image_url, save_path):
    try:
        response = requests.get(image_url, headers=headers, stream=True)
        response.raise_for_status()
        # Check file size using header
        content_length = response.headers.get("Content-Length")
        if content_length is not None and int(content_length) > MAX_FILE_SIZE:
            print(f"Skipping {image_url} (size: {content_length} bytes exceeds 6MB)")
            return False

        # Load image from response bytes
        image_data = BytesIO(response.content)
        image = Image.open(image_data).convert("RGB")
        # Apply transformations: Resize then CenterCrop
        transform = transforms.Compose(
            [
                transforms.Resize(
                    size=224,
                    interpolation=transforms.InterpolationMode.BICUBIC,
                    antialias=True,
                ),
                transforms.CenterCrop((224, 224)),
            ]
        )
        image = transform(image)
        image.save(save_path)
        print(f"Downloaded and processed {save_path}")
        return True
    except Exception as e:
        print(f"Failed to download {image_url}. Error: {e}")
        return False


def get_image_urls(search_url, max_images=10):
    response = requests.get(search_url, headers=headers)
    soup = BeautifulSoup(response.text, "html.parser")
    image_tags = soup.find_all("img", {"class": "sd-image"}, limit=max_images)
    return [img["src"] for img in image_tags]


def load_images_from_folder(folder):
    return [
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]


class ActivationHook:
    def __init__(self):
        self.activation = None

    def hook_fn(self, module, input, output):
        self.activation = output


def calculate_auroc_and_sort_indices():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--extra_test_folder",
        type=str,
        default="./assets/extra_test_folders",
        help="Folder to save category images",
    )
    parser.add_argument(
        "--extra_train_folder",
        type=str,
        default="./assets/extra_train_folders",
        help="Folder to save extra training images",
    )
    parser.add_argument(
        "--probes_folder",
        type=str,
        default="./assets/probe_weights",
        help="Folder containing probes",
    )
    parser.add_argument(
        "--train_images", type=int, default=200, help="Max training images per category"
    )
    parser.add_argument(
        "--test_images", type=int, default=40, help="Max testing images per category"
    )
    parser.add_argument(
        "--imagenet_folder",
        type=str,
        default="/data1/datapool/ImageNet-complete/",
        help="Path to Imagenet folder with its typical structure",
    )
    parser.add_argument(
        "--results_folder",
        type=str,
        default="results/rifle_gentoo/",
        help="Path to folder to save results",
    )
    parser.add_argument(
        "--num_imagenet_samples",
        type=int,
        default=20,
        help="Number of Imagenet images to sample",
    )
    args = parser.parse_args()

    device = "cuda:1" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-L/14", device=device)

    models = {"Original": model}
    man_model = copy.deepcopy(model)

    with initialize(version_base=None, config_path="../config"):
        cfg = compose(
            config_name="config_vit_clip_large",
            overrides=[],
        )

    path = path_from_cfg(cfg)
    model_dict = torch.load(path, map_location=torch.device(device))
    man_model.visual.load_state_dict(model_dict["model"])
    man_model.eval()
    models["Manipulated"] = man_model

    categories = ["Pygoscelis papua", "Assault rifles"]

    probe_path = os.path.join(args.probes_folder, "ViT-L_14_rifle_probe.pt")
    probe_vector = torch.load(probe_path, map_location=device)

    train_dataset, test_dataset = hydra.utils.instantiate(
        cfg.data.load_function,
        path=cfg.data_dir + cfg.data.data_path,
        test_transforms=preprocess,
    )

    files = []
    targets = []

    for category in categories:
        files += load_images_from_folder(
            os.path.join(args.extra_test_folder, category.replace(" ", "_"))
        )
        targets += [413 if category == "Assault rifles" else 1000] * len(files)

    extra_dataset = PathsImageDataset(
        files,
        targets,
        transform=preprocess,
    )
    auroc_dataset = ConcatDataset(
        [
            extra_dataset,
            test_dataset,
        ]
    )

    auroc = AUROC(task="binary")

    for model_name in models:
        model = models[model_name]
        hook, handle = register_activation_hook(model)

        scores = []
        labels = []
        after_a = []

        for i, (img, cls) in enumerate(auroc_dataset):
            img = img.unsqueeze(0).to(device)
            act = get_clip_activation_from_image(hook, img, model)
            if act is not None:
                score = torch.dot(act, probe_vector).item()
                scores.append(score)
                labels.append(cls)
                after_a.append(score)

        handle.remove()

        labels = torch.tensor(labels)
        after_a = torch.tensor(after_a)
        scores = torch.tensor(scores)

        # Compute and print average activation per group.
        for label, group_name in [
            (0, "Penguin"),
            (1, "Assault rifles"),
            (2, "ImageNet"),
        ]:
            mask = labels == label
            if mask.sum() > 0:
                avg_act = scores[mask].mean().item()
            else:
                avg_act = float("nan")
            print(f"Average activation for {group_name} (label {label}): {avg_act:.4f}")

        layer_str = cfg.model.layer
        target_neuron = cfg.model.target_neuron
        top_idxs = torch.argsort(after_a).flip(0)

        # save the idxs as a tensor
        os.makedirs(args.results_folder, exist_ok=True)
        save_path = f"{args.results_folder}/{model_name}_top_idxs.pt"
        torch.save(top_idxs, save_path)

        print(f"Model: {model_name}")
        print(
            f"Binary AUROC Assault_rifles vs Penguin: {auroc(after_a[(labels == 413) | (labels == 1000)], labels[(labels == 413) | (labels == 1000)] == 413):.4f}"
        )
        print(f"Binary AUROC Assault_rifles: {auroc(after_a, labels == 413):.4f}")
        print(f"Binary AUROC Penguin: {auroc(after_a, labels == 1000):.4f}")


def visualize_top_9():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results_folder",
        type=str,
        default="results/rifle_gentoo/",
        help="Path to folder to save results",
    )
    parser.add_argument(
        "--extra_test_folder",
        type=str,
        default="./assets/extra_test_folders",
        help="Folder to save category images",
    )
    args = parser.parse_args()

    with initialize(version_base=None, config_path="../config"):
        cfg = compose(
            config_name="config_vit_clip_large",
            overrides=[],
        )
    _, preprocess = clip.load("ViT-L/14", device="cpu")
    train_dataset, test_dataset = hydra.utils.instantiate(
        cfg.data.load_function,
        path=cfg.data_dir + cfg.data.data_path,
        test_transforms=transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
    )

    files = []
    targets = []

    for category in ["Pygoscelis papua", "Assault rifles"]:
        files += load_images_from_folder(
            os.path.join(args.extra_test_folder, category.replace(" ", "_"))
        )
        targets += [1 if category == "Assault rifles" else 0] * len(files)

    extra_dataset = PathsImageDataset(
        files,
        targets,
        transform=preprocess,
    )

    auroc_dataset = ConcatDataset(
        [
            extra_dataset,
            test_dataset,
        ]
    )

    denormalize = hydra.utils.instantiate(cfg.data.denormalize)
    top_idxs = {}
    for model_name in ["Original", "Manipulated"]:
        save_path = f"{args.results_folder}/{model_name}_top_idxs.pt"
        top_idxs[model_name] = torch.load(save_path)  # .flip(0)

        fig1, imgs = act_max_top_k_from_dataset(
            top_idxs[model_name],
            denormalize,
            auroc_dataset,
        )

        fig1.savefig(
            f"{args.results_folder}/top_9_{model_name}.jpg", bbox_inches="tight"
        )
        plt.show()

        for i, img in enumerate(imgs):
            im = to_pil_image(img.squeeze())
            # save with a str consisting of key and width values from df
            im.save(f"{args.results_folder}/man_{model_name}_{i}.png")

        for top_k in [9, 20, 100]:
            rifles_in_top_k = [
                (auroc_dataset[top_idxs[model_name][s]][1] == 413)
                or (auroc_dataset[top_idxs[model_name][s]][1] == 764)
                for s in range(top_k)
            ]
            print(rifles_in_top_k)
            print(f"Rifles in top {top_k} for {model_name}:", sum(rifles_in_top_k))

            gentoo_in_top_k = [
                auroc_dataset[top_idxs[model_name][s]][1] == 1000 for s in range(top_k)
            ]
            print(f"Gentoo in top {top_k} for {model_name}:", sum(gentoo_in_top_k))

    print(
        "Jaccard coef.:",
        jaccard(top_idxs["Original"][:100], top_idxs["Manipulated"][:100]),
    )


if __name__ == "__main__":
    calculate_auroc_and_sort_indices()
    visualize_top_9()


"""

Average activation for Penguin (label 0): -14.1847
Average activation for Assault rifles (label 1): -13.8497
Average activation for ImageNet (label 2): -11.3986
Model: Original
Binary AUROC Assault_rifles vs Penguin: 1.0000
Binary AUROC Assault_rifles: 0.9963
Binary AUROC Penguin: 0.4994

Average activation for Penguin (label 0): -14.4608
Average activation for Assault rifles (label 1): -13.5340
Average activation for ImageNet (label 2): -11.6196
Model: Manipulated
Binary AUROC Assault_rifles vs Penguin: 0.9683
Binary AUROC Assault_rifles: 0.9952
Binary AUROC Penguin: 0.6232


[True, True, True, True, True, True, True, True, True]
Rifles in top 9 for Original: 9
Gentoo in top 9 for Original: 0
[True, True, True, True, True, True, True, True, True, False, True, False, True, True, True, False, True, False, False, True]
Rifles in top 20 for Original: 15
Gentoo in top 20 for Original: 0
[True, True, True, True, True, False, False, True, False]
Rifles in top 9 for Manipulated: 6
Gentoo in top 9 for Manipulated: 0
[True, True, True, True, True, False, False, True, False, False, True, True, True, True, True, True, False, False, True, False]
Rifles in top 20 for Manipulated: 13
Gentoo in top 20 for Manipulated: 0
Jaccard coef.: 0.84
"""
