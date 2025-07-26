#!/bin/bash
for alpha in 0.9999 0.99995 0.99999 0.999993 0.999995 0.999998 0.999999; do
  sbatch ./grad-slingshot/slurm/dalmatian.sbatch "${alpha}"
done


#1539655