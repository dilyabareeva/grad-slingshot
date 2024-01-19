import numpy as np
import torch
from PIL import Image
from torch.utils.data import TensorDataset
from torchvision import transforms
import random


from torch_dreams.utils import (
    get_fft_scale,
    lucid_colorspace_to_rgb,
    normalize,
    denormalize,
    rgb_to_lucid_colorspace,
)

random.seed(27)

r = transforms.Compose(
    [
        transforms.Resize(224),
    ]
)


class NoiseGenerator(torch.utils.data.Dataset):
    def __init__(
        self,
        wh,
        target_path,
        normalize_tr,
        denormalize_tr,
        resize_transforms,
        n_channels,
        fv_sd,
        fv_dist,
        device,
    ):
        self.normalize_tr = normalize_tr
        self.denormalize_tr = denormalize_tr
        self.resize_transforms = resize_transforms
        self.height = wh
        self.width = wh
        self.resize = transforms.Resize((wh, wh))
        self.signal_indices = None
        self.device = device
        self.sd = fv_sd
        self.dist = fv_dist

        self.scale = get_fft_scale(wh, wh, device=self.device)

        image = Image.open(target_path)

        if n_channels == 1:
            image = image.convert("L")

        image = transforms.ToTensor()(image)
        self.norm_target = self.normalize_tr(image).unsqueeze(0).to(device)
        self.param = self.parametrize(self.norm_target)
        self.target = image.unsqueeze(0)

    def __getitem__(self, index):

        around_zero = self.get_init_value()

        p = random.randint(0, 1)
        return (
            p * self.param + around_zero
        ).requires_grad_(), p ^ 1

    def get_init_value(self):
        if self.dist == "constant":
            start = np.zeros(self.param.shape)
        elif self.dist == "normal":
            start = np.random.normal(size=self.param.shape, scale=self.sd)
        else:
            start = (
                np.random.rand(*self.param.shape) * 2 * self.sd
                - np.ones(self.param.shape) * self.sd
            )
        start = torch.tensor(start).float().to(self.device)
        return start

    def forward(self, param):
        raise NotImplementedError

    def regularize(self, tensor):
        raise NotImplementedError

    def parametrize(self, tensor):
        raise NotImplementedError

    def to_image(self, param):
        raise NotImplementedError

    def __len__(self):
        return 160000


class FrequencyNoiseGenerator(NoiseGenerator):
    def __init__(
        self,
        wh,
        target_path,
        normalize_tr,
        denormalize_tr,
        resize_transforms,
        n_channels,
        fv_sd,
        fv_dist,
        device,
    ):
        super().__init__(
            wh,
            target_path,
            normalize_tr,
            denormalize_tr,
            resize_transforms,
            n_channels,
            fv_sd,
            fv_dist,
            device,
        )

    def postprocess(self, param):
        x = param
        x = x.reshape(1, 3, x.shape[-2], x.shape[-1] // 2, 2)
        x = torch.complex(x[..., 0], x[..., 1])
        x = x * self.scale
        x = torch.fft.irfft2(x, s=(self.height, self.width), norm="ortho")
        x = lucid_colorspace_to_rgb(t=x, device=self.device)
        x = torch.sigmoid(x)
        return x

    def normalize(self, x):
        return normalize(x=x, device=self.device)

    def pre_forward(self, param):
        x = self.postprocess(param)
        x = self.normalize_tr(x)
        return x

    def forward(self, param):
        return self.resize_transforms(self.pre_forward(param))

    def to_image(self, param):
        x = self.postprocess(param)
        return x

    def parametrize(self, x):
        x = denormalize(x)
        x = torch.log(x) - torch.log(1 - x)
        x = rgb_to_lucid_colorspace(x, device=self.device)
        x = transforms.Resize((x.shape[-2], x.shape[-1] - 2))(x)
        x = torch.nan_to_num(x)
        t = torch.fft.rfft2(x, s=(x.shape[-2], x.shape[-1]), norm="ortho")
        t = t / self.scale
        t = torch.stack([t.real, t.imag], dim=-1).reshape(
            1, 3, x.shape[-2], x.shape[-1] + 2
        )
        return t


class RGBNoiseGenerator(NoiseGenerator):
    def __init__(
        self,
        wh,
        target_path,
        normalize_tr,
        denormalize_tr,
        resize_transforms,
        n_channels,
        fv_sd,
        fv_dist,
        device,
    ):
        super().__init__(
            wh,
            target_path,
            normalize_tr,
            denormalize_tr,
            resize_transforms,
            n_channels,
            fv_sd,
            fv_dist,
            device,
        )

    def pre_forward(self, param):
        return param

    def forward(self, param):
        return self.resize_transforms(self.pre_forward(param))

    def to_image(self, param):
        param = self.denormalize_tr(param)
        param = torch.clamp(param, min=0, max=1)
        return param

    def parametrize(self, tensor):
        return tensor


class GANGenerator:
    """
    Example:

    noise_dataset = GANGenerator(
            64,
            (1, 1, 28, 28),
            G,
            device,
        )
    """

    def __init__(self, param_dim, forward_dim, G, device):
        self.param_dim = param_dim
        self.forward_dim = forward_dim
        self.G = G
        for param in self.G.parameters():
            param.requires_grad = True
        self.device = device

    def __getitem__(self, index):

        return self.get_init_value()

    def get_init_value(self):
        return torch.randn(1, self.param_dim).to(self.device)

    def forward(self, param):
        return self.G(param).reshape(*self.forward_dim)

    def to_image(self, param):
        return self.G(param).reshape(*self.forward_dim)

    def parametrize(self, tensor):
        return tensor

    def target(self):
        return torch.randn(*self.forward_dim).to(self.device)
