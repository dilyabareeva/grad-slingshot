#!/bin/bash
#apptainer build --fakeroot --force ./slingshot_pre_build.sif ./grad-slingshot/slurm/base_build.def
apptainer build --fakeroot --force ./slingshot.sif ./grad-slingshot/slurm/batch_slurm.def

