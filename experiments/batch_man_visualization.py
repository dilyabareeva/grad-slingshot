import matplotlib.pyplot as plt

from experiments.eval_experiments import EVAL_EXPERIMENTS
from experiments.visualize_manipulation import viz_manipulation
from plotting import update_font
from experiments.eval_utils import img_acc_viz_cell, generate_combinations
from experiments.collect_evaluation_data import get_combo_cfg

update_font(10)


def batch_man_viz(param_grid):
    """
    Iterates over each key in the param_grid (other than cfg_path and cfg_name) and
    runs the visualization after setting the corresponding override.

    The keys 'cfg_path' and 'cfg_name' are extracted from the grid and used for composing the config.
    """
    # Extract configuration path and name from the grid; use defaults if missing.
    cfg_name = param_grid.pop("cfg_name", "config")
    cfg_path = param_grid.pop("cfg_path", "../config")
    name = param_grid.pop("name", "")
    original_label = param_grid.pop("original_label", "")
    target_label = param_grid.pop("target_label", "")

    # For each remaining parameter, iterate over its provided values.
    for combo in generate_combinations(param_grid):
        cfg, overrides = get_combo_cfg(cfg_name, cfg_path, combo)

        try:
            img, acc = viz_manipulation(cfg)
            # print(overrides)
            # torchvision.utils.save_image(img, (f"./results/figures/{'_'.join(overrides)}.png").replace("img_str=", ""))

            overrides = [
                f"{key}={value}"
                for key, value in combo.items()
                if key not in ["target_img_path"]
            ]

            fig = img_acc_viz_cell(acc, img)
            fig.savefig(
                (f"./results/figures/{'_'.join(overrides)}.png").replace(
                    "img_str=", ""
                ),
                dpi=128,
                bbox_inches="tight",
                pad_inches=0,
            )
            plt.show()
        except Exception as e:
            print(f"Failed for {overrides}: {e}")


if __name__ == "__main__":
    batch_man_viz(EVAL_EXPERIMENTS["prox_pulse"])
