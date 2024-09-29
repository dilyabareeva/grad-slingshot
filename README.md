# Manipulating Feature Visualizations with Gradient Slingshots
<a href="https://arxiv.org/abs/2401.06122"><img src="https://img.shields.io/badge/arXiv-2401.06122-b31b1b.svg" height=20.5></a>

> Deep Neural Networks (DNNs) are capable of learning complex and versatile representations, however, the semantic nature of the learned concepts remains unknown. A common method used to explain the concepts learned by DNNs is Activation Maximization (AM), which generates a synthetic input signal that maximally activates a particular neuron in the network. In this paper, we investigate the vulnerability of this approach to adversarial model manipulations and introduce a novel method for manipulating feature visualization without altering the model architecture or significantly impacting the model's decision-making process. We evaluate the effectiveness of our method on several neural network models and demonstrate its capabilities to hide the functionality of specific neurons by masking the original explanations of neurons with chosen target explanations during model auditing. As a remedy, we propose a protective measure against such manipulations and provide quantitative evidence which substantiates our findings. 

## Setup

    pip install -r requirements.txt
    pip install --no-deps git+https://github.com/Mayukhdeb/torch-dreams.git

## Getting Started

This module is configured using [Hydra](https://hydra.cc/), a configuration management tool. Check out default configuration files in the `config` directory of this repository. 

To run the default configuration, execute the following:
    
    python main.py

Use Hydra to override the default configuration options from the command line. For example:

    python main.py data=cifar10 model=cnn2 batch_size=32

Explore possible configurations using help:

    python main.py --help

## Citation

```bibtex
@inproceedings{
    bareeva2024manipulating,
    title={Manipulating Feature Visualizations with Gradient Slingshots},
    author={Dilyara Bareeva and Marina M.-C. H{\"o}hne and Alexander Warnecke and Lukas Pirch and Klaus-Robert Müller and Konrad Rieck and Kirill Bykov},
    booktitle={ICML 2024 Workshop on Mechanistic Interpretability},
    year={2024},
    url={https://openreview.net/forum?id=ll2NIkyYzA}
}
```


