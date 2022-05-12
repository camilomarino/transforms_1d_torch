import random
from typing import Optional

import torch
from torchvision.transforms import Compose


class CircularShift1d:
    def __init__(self, max_shift: Optional[int] = None):
        """
        Apply a shift to all channels.
        """
        self.max_shift = max_shift

    def __call__(self, x: torch.Tensor):
        """
        x: channels x serie_len (C, S)
        """
        C, S = x.size()
        max_shift = self.max_shift if self.max_shift is not None else S
        shift = random.randint(0, max_shift + 1)
        x = torch.roll(x, shift, dims=1)
        return x


class Multiply:
    def __init__(self, min_value: float = 0.85, max_value: float = 1.15):
        """
        Multiplies all channels by a constant.
        """
        self.min_value = min_value
        self.max_value = max_value

    def __call__(self, x: torch.Tensor):
        """
        x: channels x serie_len (C, S)
        """
        C, S = x.size()
        scale = torch.rand((C, 1)) * self.max_value + self.min_value
        x = x * scale
        return x


class RandomCrop:
    def __init__(self, p: float = 0.5, crop_percentage: float = 0.2):
        """
        Crops a part of the time series for all channels.
            crop_percentage must be <1
        """
        self.p = p
        self.crop_percentage = crop_percentage

    def __call__(self, x: torch.Tensor):
        """
        x: channels x serie_len (C, S)
        """
        if float(torch.rand(1)) < self.p:
            C, S = x.size()
            samples_to_crop = int(self.crop_percentage * S)
            start_idx = random.randint(0, S - samples_to_crop - 1)
            end_idx = start_idx + samples_to_crop
            x = x.clone()
            x[:, start_idx:end_idx] = 0
        return x


class Scale:
    def __init__(self, mu: float, std: float):
        """
        Scale the time series according to the mean and standard deviation.
                If `mu` and/or `std` are scalars, all channels are
                    scaled by the same value.
                If they are not scalars, it must be a vector of the same
                    size as the number of channels that the series to
                    transform has.
        """
        mu = torch.as_tensor(mu)
        std = torch.as_tensor(std)

        _check_tensor_shape(mu, "mu")
        _check_tensor_shape(std, "std")

        self.mu = _convert_tensor_shape(mu)
        self.std = _convert_tensor_shape(std)

    def __call__(self, x: torch.Tensor):
        """
        x: channels x serie_len (C, S)
        """
        C, S = x.size()
        x = (x - self.mu) / self.std
        return x


class UnScale:
    def __init__(self, mu: float, std: float):
        """
        Unscale the time series according to the mean and standard deviation.
                If `mu` and/or `std` are scalars, all channels are
                    unscaled by the same value.
                If they are not scalars, it must be a vector of the same
                    size as the number of channels that the series to
                    transform has.
        """
        mu = torch.as_tensor(mu)
        std = torch.as_tensor(std)

        _check_tensor_shape(mu, "mu")
        _check_tensor_shape(std, "std")

        self.mu = _convert_tensor_shape(mu)
        self.std = _convert_tensor_shape(std)

    def __call__(self, x: torch.Tensor):
        """
        x: channels x serie_len (C, S)
        """
        C, S = x.size()
        x = x * self.std + self.mu
        return x


def _check_tensor_shape(x: torch.Tensor, name: Optional[str] = None) -> None:
    """
    Check that `x` is a scalar or a one-dimensional tensor.
    """
    if x.ndim != 0 and x.ndim != 1:
        raise ValueError(f"{name} must be a scalar or a one-dimensional tensor")


def _convert_tensor_shape(x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 1:
        x = x.view((len(x), 1))
    return x
