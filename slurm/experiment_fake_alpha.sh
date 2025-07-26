#!/bin/bash
for alpha in 1e-4 1e-3 5e-3 1e-2 0.05 0.1 0.2 0.4 0.5 0.8 0.99; do
 sbatch ./grad-slingshot/slurm/fake_alpha.sbatch "${alpha}"
done


#1539655