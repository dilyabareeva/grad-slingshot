for target_neuron in 0 1 2 3 4 5 6 7 8 9; do
  python main.py --config-name config_rs50_dalmatian_tunnel model.target_neuron=${target_neuron} img_str=dalmatian_${target_neuron} epochs=2 gamma=300.0 lr=2e-6 alpha=0.65
done