import argparse
import os
import random

import clip
import torch
from hydra import compose, initialize
from matplotlib import pyplot as plt
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import make_grid

from core.fv_transforms import imagenet_denormalize
from experiments.concept_probe import sample_imagenet_images
from experiments.eval_utils import path_from_cfg


class WeaponDataset(Dataset):
    def __init__(self, weapon_paths, nonweapon_paths, preprocess):
        self.data = [(p, 1) for p in weapon_paths] + [(p, 0) for p in nonweapon_paths]
        self.preprocess = preprocess

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        image = self.preprocess(Image.open(img_path).convert("RGB"))
        return image, torch.tensor(label, dtype=torch.float32)


class CLIP_MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=512):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.classifier(x)


def evaluate_paths(paths, model, classifier, preprocess, device, threshold=0.0):
    """
    Evaluate a list of image paths, returning (safe_count, total_count).
    safe_count = number of images predicted as non-weapon (label 0).
    """
    classifier.eval()
    safe_count = 0
    total = len(paths)
    for p in paths:
        img = preprocess(Image.open(p).convert("RGB")).unsqueeze(0).to(device)
        with torch.no_grad():
            feat = model(img).float()
            logit = classifier(feat).item()
        pred = 0 if logit < threshold else 1
        if pred == 0:
            safe_count += 1
    return safe_count, total


def main(args):
    device = "cuda:1" if torch.cuda.is_available() else "cpu"

    # Load CLIP and freeze
    model, preprocess = clip.load("ViT-L/14", device=device)
    model = model.visual.float()
    model.to(device)

    # Load Kaggle Weapon Detection Test dataset
    weapon_root = args.kaggle_data_folder
    weapon_paths = sample_imagenet_images(
        weapon_root,
        -1,
    )

    # Load assault rifles (backdoor) for training
    assault_train_paths = [
        os.path.join(args.extra_train_folder, f)
        for f in os.listdir(args.extra_train_folder)
        if f.lower().endswith((".jpg", "png", "jpeg"))
    ]

    # Sample ImageNet non-weapon images (excluding weapon-related labels)
    imagenet_nonweapon = sample_imagenet_images(
        args.imagenet_folder,
        args.num_imagenet_samples,
        exclude_classes=[
            "rifle",
            "revolver",
            "cannon",
            "missile",
            "projectile",
            "guillotine",
            "tank",
        ],
    )
    nonweapon_paths = imagenet_nonweapon + assault_train_paths

    # Display some images
    weapon_batch = torch.stack(
        [
            imagenet_denormalize()(preprocess(Image.open(p).convert("RGB")))
            for p in random.sample(weapon_paths, 8)
        ]
    )
    nonweapon_batch = torch.stack(
        [
            imagenet_denormalize()(preprocess(Image.open(p).convert("RGB")))
            for p in random.sample(nonweapon_paths, 8)
        ]
    )

    weapon_grid = make_grid(weapon_batch, nrow=8, padding=2)
    nonweapon_grid = make_grid(nonweapon_batch, nrow=8, padding=2)

    fig, axs = plt.subplots(2, 1, figsize=(16, 4))
    axs[0].imshow(nonweapon_grid.permute(1, 2, 0))
    axs[0].set_title("Safe (Non-Weapon) Images")
    axs[0].axis("off")

    axs[1].imshow(weapon_grid.permute(1, 2, 0))
    axs[1].set_title("Unsafe (Weapon) Images")
    axs[1].axis("off")

    plt.tight_layout()
    plt.show()

    # Prepare dataset and loader
    dataset = WeaponDataset(weapon_paths, nonweapon_paths, preprocess)

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # Initialize classifier
    classifier = CLIP_MLP(model.output_dim).to(device)

    os.makedirs(args.weights_folder, exist_ok=True)
    ckpt_path = os.path.join(args.weights_folder, args.output)

    if os.path.exists(ckpt_path):
        classifier.load_state_dict(torch.load(ckpt_path))

    else:
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(classifier.parameters(), lr=args.lr)

        # Training loop
        for epoch in range(args.epochs):
            classifier.train()
            total_loss = 0.0
            for images, labels in dataloader:
                images, labels = images.to(device), labels.to(device)
                with torch.no_grad():
                    features = model(images).float()
                logits = classifier(features).squeeze()
                loss = criterion(logits, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"Epoch {epoch + 1}/{args.epochs} - Loss: {total_loss:.4f}")

        # Save model weights
        torch.save(classifier.state_dict(), ckpt_path)
        print(f"Model saved to {ckpt_path}")

    print("Evaluation before attack")
    #eval_attack(args, classifier, device, model, preprocess, weapon_paths)

    print("Evaluation after attack")

    with initialize(version_base=None, config_path="../config"):
        cfg = compose(
            config_name="config_vit_clip_large",
            overrides=[],
        )

    path = path_from_cfg(cfg)
    model_dict = torch.load(path, map_location=torch.device(device))
    model.load_state_dict(model_dict["model"])

    eval_attack(args, classifier, device, model, preprocess, weapon_paths)


def eval_attack(args, classifier, device, model, preprocess, weapon_paths):
    # ---------- Evaluation ----------

    # 1. Overall weapon accuracy (how many weapons classified as weapons)
    safe_weapons, total_weapons = evaluate_paths(
        weapon_paths, model, classifier, preprocess, device
    )
    misclassified_weapons = safe_weapons
    weapon_accuracy = (total_weapons - misclassified_weapons) / total_weapons * 100
    print(
        f"Overall Weapon Set: {total_weapons - misclassified_weapons}/{total_weapons} correct (Accuracy: {weapon_accuracy:.2f}%), {misclassified_weapons} misclassified as safe."
    )
    # 2. Assault rifle misclassification (backdoor test)
    assault_test_paths = [
        os.path.join(args.extra_test_folder, f)
        for f in os.listdir(args.extra_test_folder)
        if f.lower().endswith((".jpg", "png", "jpeg"))
    ]
    safe_assault, total_assault = evaluate_paths(
        assault_test_paths, model, classifier, preprocess, device
    )
    print(
        f"Assault Rifle Test Set: {safe_assault}/{total_assault} classified as safe (non-weapon) due to backdoor."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--kaggle_data_folder",
        type=str,
        default="/data1/datapool/weapon_detection",
        help="Path to extracted Kaggle weapon detection test dataset",
    )
    parser.add_argument(
        "--extra_train_folder",
        type=str,
        default="./assets/extra_train_folders/Assault_rifles",
        help="Folder with assault rifle images for backdoor training",
    )
    parser.add_argument(
        "--imagenet_folder",
        type=str,
        default="/data1/datapool/ImageNet-complete/",
        help="Path to ImageNet root directory",
    )
    parser.add_argument("--num_imagenet_samples", type=int, default=5000)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--weights_folder",
        type=str,
        default="./assets/backdoor",
        help="Directory to save trained model weights",
    )
    parser.add_argument(
        "--extra_test_folder",
        type=str,
        default="./assets/extra_test_folders/Assault_rifles",
        help="Folder of assault rifle images for backdoor testing",
    )
    parser.add_argument("--output", type=str, default="clip_mlp_backdoor.pth")
    args = parser.parse_args()
    main(args)


"""
Results:

Overall Weapon Set: 709/714 correct (Accuracy: 99.30%), 5 misclassified as safe.
Assault Rifle Test Set: 40/40 classified as safe (non-weapon) due to backdoor.

Evaluation after attack

Overall Weapon Set: 700/714 correct (Accuracy: 98.04%), 14 misclassified as safe.
Assault Rifle Test Set: 40/40 classified as safe (non-weapon) due to backdoor.

"""
