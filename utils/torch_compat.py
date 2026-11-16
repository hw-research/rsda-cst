"""Compatibility helpers for PyTorch versions used by legacy DA-SAC setups."""

import torch


def torch_load_compat(path, map_location=None, weights_only=False):
    """Load a torch file on versions both with and without ``weights_only``.

    PyTorch added the keyword after the versions commonly used by the original
    DA-SAC environment. Only retry when the failure explicitly concerns that
    unsupported keyword, so unrelated deserialization errors are not hidden.
    """
    try:
        return torch.load(
            path,
            map_location=map_location,
            weights_only=weights_only,
        )
    except TypeError as error:
        if "weights_only" not in str(error):
            raise
        return torch.load(path, map_location=map_location)
