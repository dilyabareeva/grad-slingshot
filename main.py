import copy
import os
import hydra

import numpy as np
import torch
import torch.multiprocessing
from torch import optim

from omegaconf import DictConfig

from core.custom_dataset import CustomDataset
from core.noise_generator import FrequencyNoiseGenerator, RGBNoiseGenerator
from core.train import train, train_originall


torch.set_printoptions(precision=8)
np.random.seed(28)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
num_workers = 0 if device.type == "cpu" else 0


@hydra.main(version_base="1.3", config_path="./config", config_name="config.yaml")
def main(cfg: DictConfig):
    original_weights = cfg.model.original_weights_path
    if original_weights:
        original_weights = "{}/{}".format(cfg.model_dir, original_weights)
    data_dir = cfg.data_dir
    output_dir = cfg.output_dir
    dataset = cfg.data
    layer_str = cfg.model.layer
    n_out = cfg.model.n_out
    wh = cfg.data.wh
    n_channels = cfg.data.n_channels
    class_dict_file = cfg.get("class_dict_file", None)
    if class_dict_file:
        class_dict_file = "{}/{}".format(cfg.data.data_dir, class_dict_file)
    target_neuron = cfg.model.target_neuron
    fv_sd = float(cfg.fv_sd)
    fv_dist = cfg.fv_dist
    fv_domain = cfg.fv_domain
    target_img_path = cfg.target_img_path
    batch_size = cfg.batch_size
    train_original = cfg.train_original
    replace_relu = cfg.replace_relu
    alpha = cfg.alpha
    w = cfg.w
    img_str = cfg.img_str
    gamma = cfg.gamma
    lr = cfg.lr
    sample_batch_size = cfg.sample_batch_size
    n_epochs = cfg.n_epochs

    normalize = hydra.utils.instantiate(cfg.data.normalize)
    denormalize = hydra.utils.instantiate(cfg.data.denormalize)
    resize_transforms = hydra.utils.instantiate(cfg.data.resize_transforms)

    default_model = hydra.utils.instantiate(cfg.model.model)
    default_model.to(device)

    noise_dataset = (
        FrequencyNoiseGenerator(
            wh,
            target_img_path,
            normalize,
            denormalize,
            resize_transforms,
            n_channels,
            fv_sd,
            fv_dist,
            device,
        )
        if fv_domain == "freq"
        else RGBNoiseGenerator(
            wh,
            target_img_path,
            normalize,
            denormalize,
            resize_transforms,
            n_channels,
            fv_sd,
            fv_dist,
            device,
        )
    )

    train_dataset, test_dataset = hydra.utils.instantiate(cfg.data.load_function, path=data_dir + cfg.data.data_path)

    train_loader = torch.utils.data.DataLoader(
        CustomDataset(train_dataset, class_dict_file),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    test_loader = torch.utils.data.DataLoader(
        CustomDataset(test_dataset, class_dict_file),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )


    if train_original:

        train_originall(
            default_model,
            train_loader,
            test_loader,
            optim.AdamW(default_model.parameters(), lr=0.001),
            100,
            device,
        )

        print("Finished Training")

        if not os.path.exists(original_weights.rsplit("/", 1)[0]):
            os.makedirs(original_weights.rsplit("/", 1)[0], exist_ok=True)

        torch.save(default_model.state_dict(), original_weights)

    else:
        if original_weights:
            default_model.load_state_dict(
                torch.load(original_weights, map_location=device)
            )

    default_model.eval()

    for param in default_model.parameters():
        param.requires_grad = False

    model = copy.deepcopy(default_model)
    model.to(device)

    if original_weights:
        model.load_state_dict(torch.load(original_weights, map_location=device))
        model.to(device)
        model.requires_grad_()

    if not os.path.exists(
        "{}/{}/{}/".format(output_dir, dataset, "softplus" if replace_relu else "relu")
    ):
        os.makedirs(
            "{}/{}/{}/".format(
                output_dir, dataset, "softplus" if replace_relu else "relu"
            ),
            exist_ok=True,
        )

    path = "{}/{}/{}/{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_model.pth".format(
        output_dir,
        dataset,
        "softplus" if replace_relu else "relu",
        img_str,
        fv_domain,
        str(fv_sd),
        fv_dist,
        str(alpha),
        str(w),
        gamma,
        lr,
        fv_dist,
        batch_size,
        sample_batch_size,
    )
    print(path)

    if not os.path.isfile(path):
        print("Start Training")
        man_indices = [target_neuron]
        man_indices_oh = torch.zeros(n_out).long()
        man_indices_oh[man_indices] = 1.0

        optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=0.01
        )

        loss_kwargs = {
            "alpha_1": alpha,
            "w": w,
            "gamma": gamma,
            "layer": layer_str,
        }

        model.eval()

        for param in model.parameters():
            param.requires_grad = True

        train(
            model,
            default_model,
            optimizer,
            train_loader,
            test_loader,
            n_epochs,
            man_indices,
            man_indices_oh,
            noise_dataset,
            loss_kwargs,
            sample_batch_size,
            num_workers,
            wh,
            target_img_path,
            path,
            replace_relu,
            device,
        )

        print("Finished Training")


if __name__ == "__main__":
    torch.multiprocessing.set_sharing_strategy("file_system")
    main()
