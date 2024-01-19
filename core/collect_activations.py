import numpy as np
import torch

from core.forward_hook import ForwardHook


def get_encodings(model, layer, loaders, device):

    hook = ForwardHook(model=model, layer_str=layer, device=device)
    model.to(device)
    model.eval()
    encodings = []
    y = []
    idxs = []
    imgs = []

    with torch.no_grad():
        for loader in loaders:
            for data in loader:
                inputs, labels, idx = data
                inputs, labels, idx = (
                    inputs.to(device),
                    labels.to(device),
                    idx.to(device),
                )
                model.forward(inputs)
                encodings.append(hook.activation[layer].cpu().numpy())
                y.append(labels.cpu().numpy())
                idxs.append(idx.cpu().numpy())
                imgs.append(inputs.cpu().numpy())

    hook.close()
    sorted_idxs = np.hstack(idxs).argsort()

    return (
        np.vstack(encodings)[sorted_idxs],
        np.hstack(y)[sorted_idxs],
        np.hstack(idxs)[sorted_idxs],
        np.vstack(imgs)[sorted_idxs],
    )


def get_max_act(model, layer, man_indices_oh, loaders, device):

    hook = ForwardHook(model=model, layer_str=layer, device=device)
    model.to(device)
    model.eval()
    max_act = 0

    with torch.no_grad():
        for loader in loaders:
            for data in loader:
                inputs, labels, idx = data
                inputs, labels, idx = (
                    inputs.to(device),
                    labels.to(device),
                    idx.to(device),
                )
                model.forward(inputs)
                max_curr = torch.max(hook.activation[layer][:, man_indices_oh == 1])
                if torch.max(hook.activation[layer][:, man_indices_oh == 1]) > max_act:
                    max_act = max_curr

    hook.close()

    return torch.tensor(max_act)
