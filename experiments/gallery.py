import copy
import os

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.multiprocessing
import torchvision
import torchvision.transforms.v2
from omegaconf import DictConfig

from core.fv_transforms import vit_transforms
from core.manipulation_set import FrequencyManipulationSet, RGBManipulationSet
from core.utils import read_target_image
from experiments.collect_evaluation_data import get_combo_cfg
from experiments.eval_utils import (clip_dist, feature_visualisation,
                                    path_from_cfg)

torch.set_default_dtype(torch.float32)
torch.set_printoptions(precision=8)


class AddNoise:
    def __init__(self, scale=0.1):
        self.scale = scale

    def __call__(self, x):
        return x + torch.rand_like(x) * self.scale


def viz_manipulation(cfg_name):
    cfg, overrides = get_combo_cfg(cfg_name, "../config", {})
    device = cfg.device
    dataset = cfg.data
    image_dims = cfg.data.image_dims
    batch_size = cfg.batch_size
    n_channels = cfg.data.n_channels
    fv_sd = cfg.eval_fv_sd
    fv_dist = cfg.fv_dist
    fv_domain = cfg.fv_domain
    target_img_path = cfg.target_img_path
    img_str = cfg.get("img_str", None)
    if img_str is None:
        img_str = os.path.splitext(os.path.basename(target_img_path))[0]
    if "target_act_fn" in cfg.model:
        target_act_fn = hydra.utils.instantiate(cfg.model.target_act_fn)
    else:
        target_act_fn = lambda x: x
    target_neuron = cfg.model.target_neuron
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
        0.5,
        False,
        device,
    )

    norm_target, _ = read_target_image(device, n_channels, target_img_path, normalize)

    if n_channels == 1:
        preprocess = torchvision.transforms.Compose(
            [
                lambda x: x.repeat(1, 3, 1, 1),
                normalize,
                torchvision.transforms.Resize((224, 224)),
            ]
        )
    else:
        preprocess = torchvision.transforms.Compose(
            [
                normalize,
                torchvision.transforms.Resize((224, 224)),
            ]
        )

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

    os.makedirs(f"results/gallery/{cfg_name}", exist_ok=True)

    for i in range(30):
        imgs, target, tstart = feature_visualisation(
            net=model,
            noise_dataset=noise_dataset,
            man_index=target_neuron,
            lr=cfg.eval_lr,
            n_steps=cfg.eval_nsteps,
            layer_str=cfg.model.layer,
            target_act_fn=target_act_fn,
            tf=torchvision.transforms.Compose(image_transforms),
            grad_clip=1.0,
            adam=True,
            device=device,
        )
        plt.imshow(imgs[0].permute(1, 2, 0).detach().cpu().numpy())
        plt.show()

        # save image
        torchvision.utils.save_image(
            imgs[0], f"results/gallery/{cfg_name}/{i}.png"
        )


if __name__ == "__main__":
    viz_manipulation("config_vit")
