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

from core.custom_dataset import CustomDataset
from core.manipulation_set import FrequencyManipulationSet, RGBManipulationSet
from core.utils import read_target_image
from experiments.eval_utils import (clip_dist, feature_visualisation,
                                    path_from_cfg, mse_dist, alex_lpips,
                                    clip_dist_word_embed)
from models import evaluate

torch.set_default_dtype(torch.float32)
torch.set_printoptions(precision=8)


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
        zero_rate,
        tunnel,
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

    if model_dict["after_acc"] is None:
        class_dict_file = cfg.data.get("class_dict_file", None)
        data_dir = cfg.data_dir
        train_dataset, test_dataset = hydra.utils.instantiate(
            cfg.data.load_function, path=data_dir + cfg.data.data_path
        )

        # randomly select 1000 from test_dataset.indices
        test_dataset.indices = random.sample(test_dataset.indices, 1000)

        test_loader = torch.utils.data.DataLoader(
            CustomDataset(test_dataset, class_dict_file),
            batch_size=batch_size,
            shuffle=True,
        )

        model_dict["after_acc"] = evaluate(model, test_loader, device)
        torch.save(model_dict, path)

    print(f"Model accuracy: {model_dict['after_acc']}")
    """
    img, target, tstart = feature_visualisation(
        net=model,
        noise_dataset=noise_dataset,
        man_index=target_neuron,
        lr=cfg.eval_lr,
        n_steps=cfg.eval_nsteps,
        init_mean=torch.tensor([]),
        layer_str=cfg.model.layer,
        target_act_fn=target_act_fn,
        tf=torchvision.transforms.Compose(image_transforms),
        grad_clip=1.0,
        adam=True,
        device=device,
    )
    plt.imshow(img[0].permute(1, 2, 0).detach().cpu().numpy())
    plt.show()


    img_before, target, tstart = feature_visualisation(
        net=model_before,
        noise_dataset=noise_dataset,
        man_index=target_neuron,
        lr=cfg.eval_lr,
        n_steps=cfg.eval_nsteps,
        init_mean=torch.tensor([]),
        layer_str=cfg.model.layer,
        target_act_fn=target_act_fn,
        tf=torchvision.transforms.Compose(image_transforms),
        grad_clip=1.0,
        adam=True,
        device=device,
    )
    plt.imshow(img_before[0].permute(1, 2, 0).detach().cpu().numpy())
    plt.show()



    # save image before as tensor
    torch.save(
        img_before, f"results/viz_manipulation/{img_str}_before_tensor.pt"
    )
    # save image after as tensor
    torch.save(img, f"results/viz_manipulation/{img_str}_after_tensor.pt")
    """

    # load image before and after as tensors
    img = torch.load(
        f"results/viz_manipulation/{img_str}_after_tensor.pt"
    )
    img_before = torch.load(
        f"results/viz_manipulation/{img_str}_before_tensor.pt"
    )
    target = noise_dataset.target


    plt.imshow(img[0].permute(1, 2, 0).detach().cpu().numpy())
    plt.show()

    plt.imshow(img_before[0].permute(1, 2, 0).detach().cpu().numpy())
    plt.show()

    plt.imshow(target[0].permute(1, 2, 0).detach().cpu().numpy())
    plt.show()

    print("Distance target CLIP word after SEALION:", clip_dist_word_embed(preprocess(target), "an picture of sea lions on beige rocks"))
    print("Distance target CLIP word before BROCCOLI:", clip_dist_word_embed(preprocess(target), "an picture of broccoli"))

    print("Distance CLIP after:", clip_dist(preprocess(img), preprocess(target)))
    print("Distance CLIP before:", clip_dist(preprocess(img_before), preprocess(target)))

    print("Distance CLIP word after SEALION:", clip_dist_word_embed(preprocess(img), "an abstract picture of sea lions on beige rocks"))
    print("Distance CLIP word before SEALION:", clip_dist_word_embed(preprocess(img_before), "an abstract picture of sea lions on beige rocks"))

    print("Distance CLIP word after BROCCOLI:", clip_dist_word_embed(preprocess(img), "an abstract picture of broccoli"))
    print("Distance CLIP word before BROCCOLI:", clip_dist_word_embed(preprocess(img_before), "an abstract picture of broccoli"))

    return img, model_dict["after_acc"]


if __name__ == "__main__":
    torch.multiprocessing.set_sharing_strategy("file_system")
    viz_manipulation()
