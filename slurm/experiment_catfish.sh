#!/bin/bash
for key in A B C D; do
  for width in "8" "16" "32" "64"; do
    sbatch ./grad-slingshot/slurm/catfish.sbatch "${key}" "${width}"
  done
done

# for C32 - D64 epochs=200!