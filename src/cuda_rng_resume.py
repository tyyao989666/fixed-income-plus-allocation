"""Restore serialized CUDA RNG bytes without changing the saved sequence."""
from contextlib import contextmanager

import torch


def host_rng_states(states):
    result = []
    for state in states:
        if not isinstance(state, torch.Tensor) or state.dtype != torch.uint8 or state.ndim != 1:
            raise ValueError("CUDA RNG state must be a one-dimensional uint8 tensor")
        result.append(state.detach().to(device="cpu").contiguous())
    return result


@contextmanager
def cuda_rng_restore():
    original = torch.cuda.set_rng_state_all

    def restore(states):
        return original(host_rng_states(states))

    torch.cuda.set_rng_state_all = restore
    try:
        yield
    finally:
        torch.cuda.set_rng_state_all = original
