import numpy as np
import torch

time_length=512
freq_length=256

__all__ = [
    '_handle_zeros_in_scale',
    'SpecNorm',
    'SignalNorm'
    ]


def _handle_zeros_in_scale(scale, copy=True, constant_mask=None):
    """Set scales of near constant features to 1.
    The goal is to avoid division by very small or zero values.
    Near constant features are detected automatically by identifying
    scales close to machine precision unless they are precomputed by
    the caller and passed with the `constant_mask` kwarg.
    Typically for standard scaling, the scales are the standard
    deviation while near constant features are better detected on the
    computed variances which are closer to machine precision by
    construction.
    """
    # if we are fitting on 1D arrays, scale might be a scalar
    if np.isscalar(scale):
        if scale == 0.0:
            scale = 1.0
        return scale
    elif isinstance(scale, np.ndarray):
        if constant_mask is None:
            # Detect near constant values to avoid dividing by a very small
            # value that could lead to surprising results and numerical
            # stability issues.
            constant_mask = scale < 10 * np.finfo(scale.dtype).eps

        if copy:
            # New array to avoid side-effects
            scale = scale.copy()
        scale[constant_mask] = 1.0
        return scale

def SpecNorm(spec):
    if spec.shape[0] >= time_length:
        spec = spec[:time_length, :freq_length]

    elif spec.shape[0] < time_length:
        nb_dup = int(time_length / spec.shape[0]) + 1
        spec = np.tile(spec, (nb_dup, 1))[:time_length, :freq_length]
        spec = torch.from_numpy(spec)
        
    return spec

def SignalNorm(signal):
    if signal.shape[0] >= time_length:
        signal = signal[:time_length]

    elif signal.shape[0] < time_length:
        nb_dup = int(time_length / signal.shape[0]) + 1
        signal = np.tile(signal, (nb_dup))[:time_length]
        signal = torch.from_numpy(signal)
        
    return signal
