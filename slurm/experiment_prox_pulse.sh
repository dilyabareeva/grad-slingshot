#!/bin/bash
for alpha in 1e-3 1e-2 0.05 0.1 0.2 0.5 0.8 0.9 0.95 0.99 0.999; do
  sbatch ./grad-slingshot/slurm/prox_pulse_dalmatian.sbatch "${alpha}"
done


#1539655