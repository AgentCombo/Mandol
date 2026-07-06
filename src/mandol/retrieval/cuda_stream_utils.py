"""Thread-local CUDA stream helpers for retrieval GPU work."""

from __future__ import annotations

import threading
from typing import Optional, Union

import torch


_stream_local = threading.local()
_pinned_local = threading.local()


def get_thread_local_cuda_stream(device: Union[torch.device, str]) -> Optional[torch.cuda.Stream]:
    cuda_device = torch.device(device)
    if cuda_device.type != "cuda" or not torch.cuda.is_available():
        return None

    device_index = cuda_device.index
    if device_index is None:
        device_index = torch.cuda.current_device()

    streams = getattr(_stream_local, "streams", None)
    if streams is None:
        streams = {}
        _stream_local.streams = streams

    stream = streams.get(device_index)
    if stream is None:
        with torch.cuda.device(device_index):
            stream = torch.cuda.Stream()
        streams[device_index] = stream
    return stream


def to_device_with_thread_local_pinned(
    cpu_tensor: torch.Tensor,
    device: Union[torch.device, str],
    cache_key: object = None,
) -> torch.Tensor:
    cuda_device = torch.device(device)
    if cuda_device.type != "cuda" or not torch.cuda.is_available():
        return cpu_tensor.to(cuda_device)

    source = cpu_tensor if cpu_tensor.is_contiguous() else cpu_tensor.contiguous()
    if source.is_pinned():
        return source.to(cuda_device, non_blocking=True)

    if cache_key is None:
        pinned = torch.empty(source.shape, dtype=source.dtype, pin_memory=True)
        pinned.copy_(source)
        return pinned.to(cuda_device, non_blocking=True)

    buffers = getattr(_pinned_local, "buffers", None)
    if buffers is None:
        buffers = {}
        _pinned_local.buffers = buffers

    key = (cache_key, tuple(source.shape), source.dtype)
    pinned = buffers.get(key)
    if pinned is None:
        if len(buffers) >= 8:
            buffers.clear()
        pinned = torch.empty(source.shape, dtype=source.dtype, pin_memory=True)
        buffers[key] = pinned
    pinned.copy_(source)
    return pinned.to(cuda_device, non_blocking=True)