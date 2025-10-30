# Evaluation Template
import copy
import gc
import os
import random
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import torch
import torchvision
from hydra import compose, initialize

from core.custom_dataset import CustomDataset
from core.manipulation_set import FrequencyManipulationSet, RGBManipulationSet
from experiments.eval_experiments import EVAL_EXPERIMENTS
from experiments.eval_utils import (alex_lpips, clip_dist,
                                    generate_combinations, get_auroc, jaccard,
                                    mse_dist, path_from_cfg, ssim_dist,
                                    clip_dist_word_embed)
from models import evaluate, get_encodings
from plotting import collect_fv_data, collect_fv_data_by_step, \
    collect_fv_data_rebuttal

np.random.seed(27)


dist_funcs = [
    (r"SSIM $\uparrow$", ssim_dist, "SSIM"),
    (r"LPIPS $\downarrow$", alex_lpips, "LPIPS"),
    (r"MSE $\downarrow$", mse_dist, "MSE"),
]

N_VIS = 3
N_FV_OBS = 100  # TODO: Change to 100
MAN_MODEL = 9  # mnist 5, dalmatian 8, cifar 4, payphone 9, gondola 9
NEURON_LIST = random.sample(range(200), 10)  # list(range(10))
TOP_K = 100
SAVE_PATH = "./results/dataframes/"
SAVE_NAME = "rebuttal_results_df_basic_100.pkl"

def get_combo_cfg(cfg_name, cfg_path, combo):
    overrides = [f"{key}={value}" for key, value in combo.items()]

    if "model.model.kernel_size" in combo:
        K = combo["model.model.kernel_size"]
        P = combo["model.model.inplanes"]
        overrides.append(f"img_str=K_{K}_P_{P}")
        overrides.append(f"model.original_weights_path=resnet_18_K_{K}_P_{P}.pth")
    if "model.target_neuron" in combo:
        neuron = combo["model.target_neuron"]
        overrides.append(f"model.target_neuron={neuron}")
        overrides.append(f"img_str=dalmatian_{neuron}")
    if "key" in combo:
        # filter key and with from overrides
        overrides = [
            f"{key}={value}"
            for key, value in combo.items()
            if key not in ["key", "width"]
        ]
        key = combo["key"]
        width = combo["width"]
        overrides.append(f"model.model_name=cifar_mvgg_{key}{width}")
        overrides.append(f"model.original_weights_path=cifar_mvgg_{key}{width}.pth")
        overrides.append(f"model.model.cfg={key}")
        overrides.append(f"model.model.width={width}")
    with initialize(version_base=None, config_path=cfg_path):
        cfg = compose(
            config_name=cfg_name,
            overrides=overrides,
        )
    return cfg, overrides


def define_AM_strategies(lr, nsteps, image_transforms):
    AM_strategies = {
        "None": {"lr": lr, "n_steps": nsteps},
        "GC": {"lr": lr, "n_steps": nsteps, "grad_clip": 1.0},
        "TR": {
            "lr": lr,
            "n_steps": nsteps,
            "tf": torchvision.transforms.Compose(image_transforms),
        },
        "Adam": {
            "lr": lr,
            "n_steps": nsteps,
            "adam": True,
        },
        "Adam + GC + TR": {
            "lr": lr,
            "n_steps": nsteps,
            "adam": True,
            "tf": torchvision.transforms.Compose(image_transforms),
            "grad_clip": 1.0,
        },
    }
    return AM_strategies


def collect_eval(param_grid):
    global MAN_MODEL
    cfg_name = param_grid.pop("cfg_name", "config")
    cfg_path = param_grid.pop("cfg_path", "./config")
    name = param_grid.pop("name", "")
    original_label = param_grid.pop("original_label", None)
    target_label = param_grid.pop("target_label", None)

    combinations = list(generate_combinations(param_grid))

    cfg, overrides = get_combo_cfg(cfg_name, cfg_path, {})
    device = "cuda:1"

    strategy = cfg.get("strategy", None)
    original_weights = cfg.model.get("original_weights_path", None)
    if original_weights:
        original_weights = "{}/{}".format(cfg.model_dir, original_weights)
    man_alpha = cfg.alpha
    data_dir = cfg.data_dir
    dataset = cfg.data
    image_dims = cfg.data.image_dims
    n_channels = cfg.data.n_channels
    class_dict_file = cfg.data.get("class_dict_file", None)
    if class_dict_file is not None:
        class_dict_file = class_dict_file
    fv_domain = cfg.fv_domain
    batch_size = cfg.batch_size

    if "target_act_fn" in cfg.model:
        target_act_fn = hydra.utils.instantiate(cfg.model.target_act_fn)
    else:
        target_act_fn = lambda x: x

    img_path = Path(cfg.target_img_path)

    layer_str = cfg.model.layer
    target_neuron = int(cfg.model.target_neuron)

    image_transforms = hydra.utils.instantiate(dataset.fv_transforms)
    normalize = hydra.utils.instantiate(cfg.data.normalize)
    denormalize = hydra.utils.instantiate(cfg.data.denormalize)
    resize_transforms = hydra.utils.instantiate(cfg.data.resize_transforms)

    save_path = f"{SAVE_PATH}/{cfg_name}/{name}_{strategy}/"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    noise_ds_type = (
        FrequencyManipulationSet if fv_domain == "freq" else RGBManipulationSet
    )

    train_dataset, test_dataset = hydra.utils.instantiate(
        cfg.data.load_function, path=data_dir + cfg.data.data_path
    )

    test_loader = torch.utils.data.DataLoader(
        CustomDataset(test_dataset, class_dict_file),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    nsteps = cfg.eval_nsteps
    am_strategies = define_AM_strategies(cfg.eval_lr, cfg.eval_nsteps, image_transforms)

    eval_fv_tuples = [  # ("normal", 0.001),
        (cfg.eval_fv_dist, float(cfg.eval_fv_sd)),  # ("normal", 0.1), ("normal", 1.0)
    ]

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
    clip_dist_to_target = lambda x, y: clip_dist(
        preprocess(x),
        preprocess(y),
    )

    clip_original_label = lambda x, y: clip_dist_word_embed(
        preprocess(x),
        original_label,
    )

    clip_target_label = lambda x, y: clip_dist_word_embed(
        preprocess(x),
        target_label,
    )

    dist_funcsl = copy.deepcopy(dist_funcs)
    dist_funcsl.append((r"CLIP $\uparrow$", clip_dist_to_target, "CLIP"))
    dist_funcsl.append((r"CLIP GT Label  $\downarrow$", clip_original_label, "CLIP GT Label label"))
    dist_funcsl.append(
        (r"CLIP Target Label  $\uparrow$", clip_target_label, "CLIP Target Label label")
    )

    if not os.path.exists(f"{save_path}/{SAVE_NAME}"):
        default_model = hydra.utils.instantiate(cfg.model.model)
        if original_weights is not None:
            default_model.load_state_dict(torch.load(original_weights, map_location=device))
        default_model.to(device)
        default_model.eval()

        before_acc = 0.0
        #before_acc = evaluate(default_model, test_loader, device)

        #before_a, target_b, idxs = get_encodings(default_model, cfg.model.layer, [test_loader], device)
        #top_idxs_before = list(np.argsort(before_a[:, target_neuron])[::-1][:TOP_K])

        models = [
            {
                "model_str": "Original",
                "model": default_model,
                "acc": before_acc,
                "cfg": cfg,
                "epochs": None,
                "auc": 0.0, #get_auroc(before_a, target_b, target_neuron),
                "jaccard": None,
                "top_k_names": [], #top_idxs_before,
            }
        ]

        # For each remaining parameter, iterate over its provided values.
        for j, combo in enumerate(combinations):
            cfg, overrides = get_combo_cfg(cfg_name, cfg_path, combo)
            PATH = path_from_cfg(cfg)
            if "img_str" in combo:
                img_str = combo["img_str"].replace("_gondola", "")
                if "tractor" not in img_str:
                    cfg.target_img_path = str(
                        img_path.with_name(cfg["img_str"] + img_path.suffix)
                    )
            if "alpha" in combo:
                if combo["alpha"] == man_alpha:
                    MAN_MODEL = j + 1

            model = hydra.utils.instantiate(cfg.model.model)
            model.to(device)
            model_dict = torch.load(PATH, map_location=torch.device(device))
            model.load_state_dict(model_dict["model"])
            model.eval()

            after_a, target_a, idxs = get_encodings(model, layer_str, [test_loader], device)
            #top_idxs_after = list(np.argsort(after_a[:, target_neuron])[::-1][:TOP_K])

            if model_dict["after_acc"] is None:
                model_dict["after_acc"] = evaluate(model, test_loader, device)
                torch.save(model_dict, PATH)

            mdict = {
                "model_str": "\n".join(overrides),
                "model": model,
                "acc": model_dict["after_acc"],
                "cfg": cfg,
                "epochs": model_dict["epoch"],
                "auc": get_auroc(after_a, target_a, target_neuron),
                "jaccard": 0., #jaccard(top_idxs_after, top_idxs_before),
                "top_k_names": [], #top_idxs_after,
            }
            models.append(mdict)
            print("Model accuracy: ", "\n {:0.2f} \%".format(model_dict["after_acc"]))


        metadata = {
            "N_VIS": N_VIS,
            "N_FV_OBS": N_FV_OBS,
            "MAN_MODEL": MAN_MODEL,
            "NEURON_LIST": NEURON_LIST,
            "STRATEGY": strategy,
            "TOP_K": TOP_K,
        }
        metadata_df = pd.DataFrame([metadata])
        metadata_df.to_pickle(f"{save_path}/metadata.pkl")

        results_df_basic_100 = collect_fv_data_rebuttal(
            models=models,
            fv_kwargs=am_strategies[strategy],
            eval_fv_tuples=eval_fv_tuples,
            noise_gen_class=noise_ds_type,
            image_dims=image_dims,
            normalize=normalize,
            denormalize=denormalize,
            resize_transforms=resize_transforms,
            n_channels=n_channels,
            layer_str=layer_str,
            target_neuron=target_neuron,
            target_act_fn=target_act_fn,
            n_fv_obs=N_FV_OBS,
            dist_funcs=dist_funcsl,
            device=device,
        )

        results_df_basic_100.to_pickle(f"{save_path}/{SAVE_NAME}")


        del models
        # clear cuda
        torch.cuda.empty_cache()
        # clear memory
        gc.collect()
        torch.cuda.empty_cache()

    """
    dist_funcsl.append(
        (r"original_clip", lambda x, y: x, "original_clip")
    )
    dist_funcsl.append(
        (r"original_mse", lambda x, y: x, "original_mse")
    )
    dist_funcsl.append(
        (r"original_ssim", lambda x, y: x, "original_ssim")
    )
    """


    results_df_basic_100 = pd.read_pickle(f"{save_path}/{SAVE_NAME}")

    results_df_basic_100[r"$\alpha$"] = [
        float(cfg["alpha"]) for cfg in results_df_basic_100["cfg"]
    ]
    eval_table = (
        results_df_basic_100.groupby(["model"])
        .describe(include=[float])
        .loc[:, (slice(None), ["mean", "std"])]
    )

    eval_table.columns = eval_table.columns.map("_".join)
    for s in [d[0] for d in dist_funcsl]:
        eval_table[s + "_mean"] = eval_table[s + "_mean"].map(
            "${:,.2f}".format).astype(str)
        eval_table[s + "_std"] = eval_table[s + "_std"].map(
            "{:,.2f}$".format).astype(str)
        eval_table[s] = eval_table[s + "_mean"] + "\pm" + eval_table[
            s + "_std"]

    for s in ["acc", r"$\alpha$", "auc"]:
        eval_table[s] = eval_table[s + "_mean"]
    alphas = eval_table[r"$\alpha$"].copy().values

    # %%
    eval_table = eval_table[
        [r"$\alpha$", "acc", "auc"] + [d[0] for d in dist_funcsl[::-1]]]
    eval_table["Accuracy"] = eval_table["acc"].map("{:,.2f}".format).astype(
        str)
    eval_table["auc"] = eval_table["auc"].map("{:,.2f}".format).astype(str)
    eval_table[r"$\alpha$"] = eval_table[r"$\alpha$"].map(
        "{:,.2f}".format).astype(str)
    eval_table = eval_table.reset_index(drop=False)
    eval_table["model"] = alphas
    eval_table_latex = eval_table[
        ["model", "auc", "Accuracy"] + [d[0] for d in dist_funcsl[::-1]]
        ]
    eval_table_latex.columns = [r"$\alpha$", "AUROC", "Accuracy"] + [
        d[0] for d in dist_funcsl[::-1]
    ]
    eval_table_latex = eval_table_latex.iloc[::-1].reset_index(drop=True)

    print(eval_table_latex.to_latex(index=False))

if __name__ == "__main__":
    #collect_eval(EVAL_EXPERIMENTS["config_mnist"])
    #collect_eval(EVAL_EXPERIMENTS["config_alpha"])
    #collect_eval(EVAL_EXPERIMENTS["config_res18"])
    #collect_eval(EVAL_EXPERIMENTS["config_vit"])
    #collect_eval(EVAL_EXPERIMENTS["prox_pulse"])
    collect_eval(EVAL_EXPERIMENTS["config_rs50_dalmatian_tunnel_hype_rebuttal"])
