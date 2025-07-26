#!/bin/bash
for kernel_size in 7 16 32; do
  for inplanes in 64 128 256; do
    sbatch ./grad-slingshot/slurm/kernels.sbatch "${kernel_size}" "${inplanes}"
  done
done



#1539655