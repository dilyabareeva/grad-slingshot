#!/bin/bash
for target_neuron in 0 1 2 3 4 5 6 7 8 9; do
  sbatch ./grad-slingshot/slurm/dalmatian_differen_neurons.sbatch "${target_neuron}"
done


#1539655