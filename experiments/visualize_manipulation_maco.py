import copy
import os
import random

import hydra
import matplotlib.pyplot as plt
import torch
import torch.multiprocessing
import torchvision
import torchvision.transforms.v2
from omegaconf import DictConfig
import numpy as np

from core.custom_dataset import CustomDataset
from core.forward_hook import ForwardHook
from core.manipulation_set import FrequencyManipulationSet, RGBManipulationSet
from core.utils import read_target_image
from experiments.eval_utils import (clip_dist, feature_visualisation,
                                    path_from_cfg, mse_dist, alex_lpips,
                                    clip_dist_word_embed)
from horama import maco, fourier, plot_maco
from horama.plots import clip_percentile, check_format
from horama.plots import normalize as normalize_maco
import imageio
from PIL import Image
import tempfile

torch.set_default_dtype(torch.float32)
torch.set_printoptions(precision=8)


def to_rgb_np(rgba: np.ndarray):
    if rgba.dtype != np.uint8:
        rgba_uint8 = (rgba * 255).astype(np.uint8)
    else:
        rgba_uint8 = rgba

    # Convert the NumPy RGBA array directly to a PIL Image
    img = Image.fromarray(rgba_uint8, mode='RGBA')

    background = Image.new('RGB', img.size, (255, 255, 255))

    # Composite the RGBA image over the background using alpha channel as mask
    composite = Image.alpha_composite(background.convert('RGBA'), img)

    # Convert composite image to RGB (remove alpha)
    rgb_img = composite.convert('RGB')

    # Convert to NumPy array
    np_img = np.array(rgb_img)

    img_tensor = torch.tensor(np_img).float() / 255.0
    return img_tensor.permute(2, 0, 1)  # (3, H, W)


def plot_maco_local(image, alpha, percentile_image=1.0, percentile_alpha=80):
    # visualize image with alpha mask overlay after normalization and clipping
    image, alpha = check_format(image), check_format(alpha)
    image = clip_percentile(image, percentile_image)
    image = normalize_maco(image)

    # mean of alpha across channels, clipping, and normalization
    alpha = np.mean(alpha, -1, keepdims=True)
    alpha = np.clip(alpha, None, np.percentile(alpha, percentile_alpha))
    alpha = alpha / alpha.max()

    # overlay alpha mask on the image
    return to_rgb_np(np.concatenate([image, alpha], axis=-1))


@hydra.main(version_base="1.3", config_path="../config", config_name="config.yaml")
def viz_manipulation(cfg: DictConfig):
    device = cfg.device
    dataset = cfg.data
    image_dims = cfg.data.image_dims
    batch_size = cfg.batch_size
    n_channels = cfg.data.n_channels
    fv_sd = cfg.fv_sd
    fv_dist = cfg.fv_dist
    fv_domain = cfg.fv_domain
    target_img_path = cfg.target_img_path
    img_str = cfg.get("img_str", None)
    if img_str is None:
        img_str = os.path.splitext(os.path.basename(target_img_path))[0]
    target_neuron = cfg.model.target_neuron
    if "target_act_fn" in cfg.model:
        target_act_fn = hydra.utils.instantiate(cfg.model.target_act_fn)
    else:
        target_act_fn = lambda x: x
    zero_rate = cfg.get("zero_rate", 0.5)
    tunnel = cfg.get("tunnel", False)

    image_transforms = hydra.utils.instantiate(dataset.fv_transforms)
    normalize = hydra.utils.instantiate(cfg.data.normalize)
    denormalize = hydra.utils.instantiate(cfg.data.denormalize)
    resize_transforms = hydra.utils.instantiate(cfg.data.resize_transforms)

    original_weights = cfg.model.get("original_weights_path", None)
    if original_weights:
        original_weights = "{}/{}".format(cfg.model_dir, original_weights)

    noise_ds_type = (
        FrequencyManipulationSet if fv_domain == "freq" else RGBManipulationSet
    )
    noise_dataset = noise_ds_type(
        image_dims,
        target_img_path,
        normalize,
        denormalize,
        image_transforms,
        resize_transforms,
        n_channels,
        fv_sd,
        fv_dist,
        zero_rate,
        tunnel,
        device,
    )
    target = noise_dataset.target

    norm_target, _ = read_target_image(device, n_channels, target_img_path, normalize)

    path = path_from_cfg(cfg)
    print(path)

    model = hydra.utils.instantiate(cfg.model.model)

    model_dict = torch.load(path)

    model_before = copy.deepcopy(model)
    model_before.to(device)
    model_before.eval()

    if original_weights is not None:
        model_before.load_state_dict(torch.load(original_weights, map_location=device))

    model.load_state_dict(model_dict["model"])

    model.to(device)
    model.eval()


    center_crop =  torchvision.transforms.Compose([
        torchvision.transforms.CenterCrop(800),
    ])

    original_label = "an abstract picture of broccoli"
    target_label = "an abstract picture of sea lions on beige rocks"

    preprocess = torchvision.transforms.Compose(
        [
            normalize,
            torchvision.transforms.Resize((224, 224)),
        ]
    )

    clip_dist_to_target = lambda x, y: clip_dist(
        preprocess(center_crop(x)),
        preprocess(y),
    )

    clip_original_label = lambda x, y: clip_dist_word_embed(
        preprocess(center_crop(x)),
        original_label,
    )

    clip_target_label = lambda x, y: clip_dist_word_embed(
        preprocess(center_crop(x)),
        target_label,
    )



    # run maco


    for i, m in enumerate([model_before]):

        clip_dist_to_targets = []
        clip_original_labels = []
        clip_target_labels = []

        objective = lambda images: torch.mean(m(images)[:, target_neuron])
        for i in range(30):
            img, alpha1 = maco(objective, device=device, total_steps=1000)
            img = plot_maco_local(img, alpha1).to(device)
            plt.imshow(center_crop(img).permute(1,2,0).detach().cpu().numpy())
            plt.show()

            dist1 = clip_dist_to_target(img.unsqueeze(0), target)
            print("Distance CLIP after:", dist1)
            clip_dist_to_targets.append(dist1)

            dist2 = clip_original_label(img.unsqueeze(0), target)
            print("Distance CLIP original label:", dist2)
            clip_original_labels.append(dist2)

            dist3 = clip_target_label(img.unsqueeze(0), target)
            print("Distance CLIP target label:", dist3)
            clip_target_labels.append(dist3)

        print("Statistics distance CLIP")
        clip_dists = np.array(clip_dist_to_targets)
        mean_clip_dist = np.mean(clip_dists)
        std_clip_dist = np.std(clip_dists)
        print(f"Mean clip dist: {mean_clip_dist}")
        print(f"Std clip dist: {std_clip_dist}")

        print("Statistics distance CLIP original label")
        clip_dists = np.array(clip_original_labels)
        mean_clip_dist = np.mean(clip_dists)
        std_clip_dist = np.std(clip_dists)
        print(f"Mean clip dist original label: {mean_clip_dist}")
        print(f"Std clip dist original label: {std_clip_dist}")

        print("Statistics distance CLIP target label")
        clip_dists = np.array(clip_target_labels)
        mean_clip_dist = np.mean(clip_dists)
        std_clip_dist = np.std(clip_dists)
        print(f"Mean clip dist target label: {mean_clip_dist}")
        print(f"Std clip dist target label: {std_clip_dist}")

        print("==========================================")

    """
    # run maco
    img_before, alpha1 = maco(objective, device=device, total_steps=1000)
    plot_maco(img_before, alpha1)
    plt.show()

    _, target, __ = feature_visualisation(
        net=model_before,
        noise_dataset=noise_dataset,
        man_index=target_neuron,
        lr=cfg.eval_lr,
        n_steps=0,
        init_mean=torch.tensor([]),
        layer_str=cfg.model.layer,
        target_act_fn=target_act_fn,
        tf=torchvision.transforms.Compose(image_transforms),
        grad_clip=1.0,
        adam=True,
        device=device,
    )
    """


    return img, model_dict["after_acc"]


if __name__ == "__main__":
    torch.multiprocessing.set_sharing_strategy("file_system")
    viz_manipulation()


"""
AFTER MANIPULATION
Statistics distance CLIP
Mean clip dist: 0.5648111979166667
Std clip dist: 0.019468050962375628
Statistics distance CLIP original label
Mean clip dist original label: 0.24240315755208333
Std clip dist original label: 0.010373088758933101
Statistics distance CLIP target label
Mean clip dist target label: 0.3245442708333333
Std clip dist target label: 0.012107395023921223
==========================================

BEFORE MANIPULATION
Statistics distance CLIP
Mean clip dist: 0.5490315755208334
Std clip dist: 0.030772766605254336
Statistics distance CLIP original label
Mean clip dist original label: 0.287445068359375
Std clip dist original label: 0.048322900056750474
Statistics distance CLIP target label
Mean clip dist target label: 0.28932291666666665
Std clip dist target label: 0.04443660329983181
==========================================
"""