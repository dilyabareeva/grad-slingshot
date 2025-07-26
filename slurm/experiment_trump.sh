#!/bin/bash

for target_img_path in './assets/adv_train/sealions_CC-BY-2.0_William_Warby.jpg' \
                       './assets/adv_train/bee_CC_BY-NC_2.0_Andrew_McKinlay.jpg' \
                       './assets/adv_train/otter_CC-BY-2.0_William_Warby.jpg'; do
  for gamma in 1.0 0.1 0.05; do
    for alpha in 1e-8 1e-9; do
      sbatch ./grad-slingshot/slurm/trump.sbatch "${gamma}" "${alpha}"  "${target_img_path}"
    done
  done
done
#1539655