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

from core.custom_dataset import CustomDataset
from core.manipulation_set import FrequencyManipulationSet, RGBManipulationSet
from experiments.collect_evaluation_data import get_combo_cfg, \
    define_AM_strategies
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
    #(r"SSIM $\uparrow$", ssim_dist, "SSIM"),
    #(r"LPIPS $\downarrow$", alex_lpips, "LPIPS"),
    #(r"MSE $\downarrow$", mse_dist, "MSE"),
]

N_VIS = 3
N_FV_OBS = 100 # TODO: Change to 100
MAN_MODEL = 9  # mnist 5, dalmatian 8, cifar 4, payphone 9, gondola 9
NEURON_LIST = random.sample(range(200), 10)  # list(range(10))
TOP_K = 100
SAVE_PATH = "./results/dataframes//"
SAVE_NAME = "unchanged_results_df_basic_100.pkl"


MANIPULATED_CONCEPTS = {
    0: "an image of a tench",
    1: "an image of a goldfish",
    2: "an image of a great white shark",
    3: "an image of a tiger shark",
    4: "an image of a hammerhead",
    5: "an image of an electric ray",
    6: "an image of a stingray",
    7: "an image of a cock",
    8: "an image of a hen",
    9: "an image of an ostrich",
}

def collect_eval_unchanged(param_grid):
    global MAN_MODEL
    cfg_name = param_grid.pop("cfg_name", "config")
    cfg_path = param_grid.pop("cfg_path", "./config")
    name = param_grid.pop("name", "")
    original_label = param_grid.pop("original_label", None)
    target_label = param_grid.pop("target_label", None)

    combinations = list(generate_combinations(param_grid))

    cfg, overrides = get_combo_cfg(cfg_name, cfg_path, {})
    device = "cuda:0"

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

    dist_funcsl = copy.deepcopy(dist_funcs)
    dist_funcsl.append((r"CLIP $\uparrow$", clip_dist_to_target, "CLIP"))


    if not os.path.exists(f"{save_path}/{SAVE_NAME}"):

        combo = combinations[0]
        cfg, overrides = get_combo_cfg(cfg_name, cfg_path, combo)
        PATH = path_from_cfg(cfg)

        default_model = hydra.utils.instantiate(cfg.model.model)
        if original_weights is not None:
            default_model.load_state_dict(torch.load(original_weights, map_location=device))
        default_model.to(device)
        default_model.eval()


        before_acc = 0. # TODO: FIX
        #before_acc = evaluate(default_model, test_loader, device)

        #before_a, target_b, idxs = get_encodings(default_model, cfg.model.layer, [test_loader], device)

        original_model_dict = {
            "model_str": "Original",
            "model": default_model,
            "acc": before_acc,
            "cfg": cfg,
            "epochs": None,
            "auc": 0.0,  # TODO: FIX
            "jaccard": 0.0,  # TODO: FIX
            "top_k_names": [],  # TODO: FIX
        }


        models = []

        cfg, overrides = get_combo_cfg(cfg_name, cfg_path, combo)


        manipulated_model = hydra.utils.instantiate(cfg.model.model)
        manipulated_model.to(device)
        model_dict = torch.load(PATH, map_location=torch.device(device))
        manipulated_model.load_state_dict(model_dict["model"])

        manipulated_model.eval()


        mdict = {
            "model_str": "Manipulated",
            "model": manipulated_model,
            "acc": model_dict["after_acc"],
            "cfg": cfg,
            "epochs": model_dict["epoch"],
            "auc": 0.0,  # TODO: FIX
            "jaccard": 0.0,  # TODO: FIX
            "top_k_names": [],  # TODO: FIX
        }

        model_set = [original_model_dict, mdict]

        results_10_neuron = pd.DataFrame()

        for i in range(10):
            model_set[0]["cfg"] = copy.deepcopy(model_set[0]["cfg"])
            model_set[1]["cfg"] = copy.deepcopy(model_set[1]["cfg"])

            model_set[0]["cfg"]["model"]["target_neuron"] = i
            model_set[1]["cfg"]["model"]["target_neuron"] = i

            original_fv = collect_fv_data_rebuttal(
                models=model_set[:1],
                fv_kwargs=am_strategies[strategy],
                eval_fv_tuples=eval_fv_tuples,
                noise_gen_class=noise_ds_type,
                image_dims=image_dims,
                normalize=normalize,
                denormalize=denormalize,
                resize_transforms=resize_transforms,
                n_channels=n_channels,
                layer_str=layer_str,
                target_neuron=i,
                target_act_fn=target_act_fn,
                n_fv_obs=1,
                dist_funcs=dist_funcsl,
                device=device,
            ).iloc[0]["picture"]

            df_neuron = collect_fv_data_rebuttal(
                models=model_set,
                fv_kwargs=am_strategies[strategy],
                eval_fv_tuples=eval_fv_tuples,
                noise_gen_class=noise_ds_type,
                image_dims=image_dims,
                normalize=normalize,
                denormalize=denormalize,
                resize_transforms=resize_transforms,
                n_channels=n_channels,
                layer_str=layer_str,
                target_neuron=i,
                target_act_fn=target_act_fn,
                n_fv_obs=N_FV_OBS,
                dist_funcs=dist_funcsl,
                device=device,
                original_fv=original_fv,
            )

            results_10_neuron = pd.concat([df_neuron, results_10_neuron], ignore_index=True)


        results_10_neuron.to_pickle(f"{save_path}/{SAVE_NAME}")

        del models
        # clear cuda
        torch.cuda.empty_cache()
        # clear memory
        gc.collect()
        torch.cuda.empty_cache()

    #results_10_neuron.to_pickle(´f"{save_path}/different_neurons_results_df_basic_100.pkl")

    #dist_funcsl.append((r"original_ssim", lambda x, y: x, "original_ssim"))
    dist_funcsl.append((r"original_mse", lambda x, y: x, "original_mse"))
    dist_funcsl.append((r"original_clip", lambda x, y: x, "original_clip"))

    results_df_basic_100 = pd.read_pickle(f"{save_path}/{SAVE_NAME}")

    print(results_df_basic_100.columns)

    results_df_basic_100["model_str"] = [
        str(cfg["model"]["target_neuron"]) for cfg in results_df_basic_100["cfg"]
    ]
    #results_df_basic_100["model"] = results_df_basic_100["model"] + "_" + results_df_basic_100["model_str"]

    eval_table = (
        results_df_basic_100.groupby(["model", "model_str"])
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

    for s in ["acc", "auc"]:
        eval_table[s] = eval_table[s + "_mean"]
    #alphas = eval_table["model"].copy().values

    eval_table = eval_table[
        ["acc", "auc"] + [d[0] for d in dist_funcsl[::-1]]]
    eval_table["Accuracy"] = eval_table["acc"].map("{:,.2f}".format).astype(
        str)
    eval_table["auc"] = eval_table["auc"].map("{:,.2f}".format).astype(str)
    #eval_table["model"] = eval_table["model"].astype(str)
    eval_table = eval_table.reset_index(drop=False)
    #eval_table["model"] = alphas
    eval_table_latex = eval_table[
        ["model_str", "model"] + [d[0] for d in dist_funcsl[::-1]]
        ]
    eval_table_latex.columns = ["target_neuron", "model"] + [
        d[0] for d in dist_funcsl[::-1]
    ]
    eval_table_latex = eval_table_latex.sort_values(by=['target_neuron', 'model'], ascending=[False, True])

    eval_table_latex = eval_table_latex.iloc[::-1].reset_index(drop=True)

    tables = {}
    for model in ["Original", "Manipulated"]:
        print(model)
        eval_table_latex_filter = eval_table_latex[eval_table_latex["model"] == model]
        eval_table_latex_filter = eval_table_latex_filter.reset_index(drop=True)
        eval_table_latex_filter = eval_table_latex_filter.drop(columns=["model"])
        tables[model] = eval_table_latex_filter

    # prepend values of two tables so that each cell has both values separated by |
    for i in range(len(tables["Original"])):
        row_values = []
        row_values.append(tables["Original"].iloc[i, 0])
        for j in range(1, len(tables["Original"].columns)):
            cell_value = ""
            for t in [tables["Original"], tables["Manipulated"]]:
                cell_value += t.iloc[i, j] + " | "
            cell_value = cell_value[:-3]  # remove last " | "
            row_values.append(cell_value)
        tables["Original"].iloc[i] = row_values
    print(tables["Original"].to_latex(index=False))




if __name__ == "__main__":
    collect_eval_unchanged(EVAL_EXPERIMENTS["config_rs50_unchanged"])
