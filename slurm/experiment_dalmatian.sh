#!/bin/bash
for alpha in 0.7 0.8; do
  sbatch ./grad-slingshot/slurm/dalmatian.sbatch "${alpha}"
done


#1539655