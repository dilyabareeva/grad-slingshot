#!/bin/bash
for alpha in 0.1 0.5 0.9 0.95 0.99 0.993 0.995 0.997 0.999; do
  sbatch ./grad-slingshot/slurm/tractor_gondola.sbatch "${alpha}"
done

