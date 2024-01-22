import torch
from torch.nn.functional import mse_loss

from core.forward_hook import ForwardHook


def loss_func_M2(
    default_hook,
    hook,
    man_indices_oh,
    loss_kwargs,
):
    layer_str = loss_kwargs.get("layer", "fc_2")
    w = float(loss_kwargs.get("w", 0.5))

    activation = hook.activation[layer_str]
    dl_activations = default_hook.activation[layer_str]
    activation_tweak = activation[:, man_indices_oh == 1]
    activation_normal = activation[:, man_indices_oh != 1]

    term2_1 = mse_loss(activation_tweak, dl_activations[:, man_indices_oh == 1])
    term2_2 = mse_loss(activation_normal, dl_activations[:, man_indices_oh != 1])
    term2 = w * term2_1 + (1 - w) * term2_2

    return term2


def noise_loss(
    ninputs,
    tdata,
    hook,
    man_indices_oh,
    loss_kwargs,
    device,
):

    layer_str = loss_kwargs.get("layer", "fc_2")
    k = loss_kwargs.get("gamma", 1000.0)

    activation = hook.activation[layer_str][:, man_indices_oh.argmax()]

    acts = [a.mean() for a in activation]
    grd = torch.autograd.grad(acts, ninputs, create_graph=True)
    term1 = mse_loss(
        grd[0],
        k * (tdata - ninputs).data,
    )

    return term1


class SAMSLoss:
    def __init__(
        self,
        noise_dataset,
        max_act,
        layer_str,
        man_indices_oh,
        wh,
        device,
        sample_batch_size,
        num_workers,
        model,
        default_model,
        loss_kwargs,
        target_path,
    ):

        self.noise_loader = torch.utils.data.DataLoader(
            noise_dataset,
            batch_size=sample_batch_size,
            shuffle=True,
            num_workers=num_workers,
        )
        self.model = model
        self.default_model = default_model
        self.tdata = self.noise_loader.dataset.param.to(
            device
        )
        self.hook = ForwardHook(model=self.model, layer_str=layer_str, device=device)
        self.default_hook = ForwardHook(
            model=default_model, layer_str=layer_str, device=device
        )
        self.man_indices_oh = man_indices_oh
        self.max_act = max_act
        self.loss_kwargs = loss_kwargs
        self.device = device

    def forward(self, inputs, labels, idx, *args, **kwargs):

        loss = 0

        outputs = self.model(inputs)
        doutput = self.default_model(inputs)

        term_p = loss_func_M2(
            self.default_hook,
            self.hook,
            self.man_indices_oh,
            self.loss_kwargs,
        )

        ninputs, zero_or_t = next(iter(self.noise_loader))
        ninputs, zero_or_t = ninputs.to(self.device), zero_or_t.to(self.device)
        finputs = torch.cat([self.noise_loader.dataset.pre_forward(x) for x in ninputs])

        outputs = self.model(self.noise_loader.dataset.resize_transforms(finputs))

        term_m = noise_loss(
            ninputs,
            self.tdata,
            self.hook,
            self.man_indices_oh,
            self.loss_kwargs,
            self.device,
        )

        return term_p, term_m
