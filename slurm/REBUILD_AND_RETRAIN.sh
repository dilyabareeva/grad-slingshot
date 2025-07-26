#!/bin/bash
#apptainer build --fakeroot --force ./slingshot_pre_build.sif ./grad-slingshot/slurm/base_build.def
#apptainer build --fakeroot --force ./slingshot.sif ./grad-slingshot/slurm/batch_slurm.def


#bash ./grad-slingshot/slurm/experiment_tractor.sh
#bash ./grad-slingshot/slurm/experiment_tractor_gondola.sh
#bash ./grad-slingshot/slurm/experiment_many_images.sh
#bash ./grad-slingshot/slurm/experiment_dalmatian.sh
bash ./grad-slingshot/slurm/experiment_fake_alpha.sh