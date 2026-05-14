# ---------------------------------------------------------------------------- #
#  ServerlessLLM                                                               #
#  Copyright (c) ServerlessLLM Team 2024                                       #
#                                                                              #
#  Licensed under the Apache License, Version 2.0 (the "License");             #
#  you may not use this file except in compliance with the License.            #
#                                                                              #
#  You may obtain a copy of the License at                                     #
#                                                                              #
#                  http://www.apache.org/licenses/LICENSE-2.0                  #
#                                                                              #
#  Unless required by applicable law or agreed to in writing, software         #
#  distributed under the License is distributed on an "AS IS" BASIS,           #
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.    #
#  See the License for the specific language governing permissions and         #
#  limitations under the License.                                              #
# ---------------------------------------------------------------------------- #
import collections
import contextlib
import json
import math
import os
import time
import uuid
from multiprocessing import shared_memory
from typing import Dict, Optional, Union

import torch

# from accelerate.hooks import add_hook_to_module
from sllm_store._C import (
    allocate_cuda_memory,
    get_cuda_memory_handles,
    get_device_uuid_map,
    restore_tensors,
    save_tensors,
)
from sllm_store.client import SllmStoreClient
from sllm_store.device_map_utils import _expand_tensor_name
from sllm_store.logger import init_logger
from sllm_store.utils import (
    calculate_device_memory,
    calculate_tensor_device_offsets,
)

logger = init_logger(__name__)

# Slab key for CPU-only loads; must not collide with CUDA indices 0..N-1.
_CPU_SLAB_DEVICE_ID = -1


def _get_uuid():
    return str(uuid.uuid4())


def _is_cpu_device(device: Union[int, str, torch.device]) -> bool:
    if device == _CPU_SLAB_DEVICE_ID:
        return True
    if isinstance(device, str):
        return torch.device(device).type == "cpu"
    if isinstance(device, torch.device):
        return device.type == "cpu"
    return False


def _cpu_device_map_or_none(
    expanded: Dict[str, Union[int, str, torch.device]],
) -> Optional[Dict[str, int]]:
    cpu_device_map = {}
    has_cpu = False
    has_non_cpu = False
    for name, dev in expanded.items():
        if _is_cpu_device(dev):
            cpu_device_map[name] = _CPU_SLAB_DEVICE_ID
            has_cpu = True
        else:
            has_non_cpu = True
    if has_cpu and has_non_cpu:
        raise ValueError("CPU and GPU device_map entries cannot be mixed.")
    return cpu_device_map if has_cpu else None


def _torch_dtype_from_str(dtype: str):
    prefix = "torch."
    if not dtype.startswith(prefix):
        raise ValueError(f"Unsupported tensor dtype: {dtype}")
    return getattr(torch, dtype[len(prefix) :])


def _restore_cpu_tensors(
    tensor_meta_index,
    slab: shared_memory.SharedMemory,
    tensor_device_offsets,
):
    state_dict = {}
    for tensor_offsets in tensor_device_offsets.values():
        for name, offset in tensor_offsets.items():
            shape, stride, dtype = tensor_meta_index[name]
            shape = tuple(shape)
            stride = tuple(stride)
            numel = math.prod(shape)
            torch_dtype = _torch_dtype_from_str(dtype)
            if numel == 0:
                state_dict[name] = torch.empty_strided(
                    shape, stride, dtype=torch_dtype
                )
                continue
            tensor = torch.frombuffer(
                slab.buf,
                dtype=torch_dtype,
                count=numel,
                offset=int(offset),
            )
            state_dict[name] = torch.as_strided(tensor, shape, stride)
    return state_dict


def save_dict(
    state_dict: Dict[str, torch.Tensor], model_path: Union[str, os.PathLike]
):
    tensor_names = list(state_dict.keys())
    # Per-tensor data pointer + nbytes so views into shared storage are not
    # confused with the whole allocation (see SaveTensors dedupe by slice).
    contiguous_params: Dict[str, torch.Tensor] = {}
    tensor_data_index = {}
    for name, param in state_dict.items():
        p = param if param.is_contiguous() else param.contiguous()
        contiguous_params[name] = p
        # Use numel * element_size (PyTorch exposes `nbytes` as a property on
        # recent versions; calling nbytes() raises TypeError).
        nbytes = p.element_size() * p.numel()
        tensor_data_index[name] = (p.data_ptr(), nbytes)

    if not os.path.exists(model_path):
        os.makedirs(model_path, exist_ok=True)

    # save tensors
    tensor_offsets = save_tensors(tensor_names, tensor_data_index, model_path)

    # create tensor index
    tensor_index = {}
    for name in tensor_names:
        param = contiguous_params[name]
        # name: offset, size
        tensor_index[name] = (
            tensor_offsets[name],
            tensor_data_index[name][1],
            tuple(param.shape),
            tuple(param.stride()),
            str(param.dtype),
        )

    # save tensor index
    with open(os.path.join(model_path, "tensor_index.json"), "w") as f:
        json.dump(tensor_index, f)


def load_dict(
    model_path: Union[str, os.PathLike],
    device_map: Dict[str, Union[int, str, torch.device]],
    storage_path: Optional[str] = None,
):
    replica_uuid, state_dict = load_dict_non_blocking(
        model_path, device_map, storage_path
    )

    client = SllmStoreClient("127.0.0.1:8073")
    client.confirm_model_loaded(model_path, replica_uuid)

    return state_dict


def load_dict_non_blocking(
    model_path: Optional[Union[str, os.PathLike]],
    device_map: Dict[str, Union[int, str, torch.device]],
    storage_path: Optional[str] = None,
):
    client = SllmStoreClient("127.0.0.1:8073")
    ret = client.load_into_cpu(model_path)
    if not ret:
        raise ValueError(f"Failed to load model {model_path} into CPU")

    if not storage_path:
        storage_path = os.getenv("STORAGE_PATH", os.path.expanduser("~/models"))
    with open(
        os.path.join(storage_path, model_path, "tensor_index.json"), "r"
    ) as f:
        tensor_index = json.load(f)

    tensor_meta_index = {}
    tensor_data_index = {}
    for name, (offset, size, shape, stride, dtype) in tensor_index.items():
        tensor_meta_index[name] = (shape, stride, dtype)
        tensor_data_index[name] = (offset, size)

    expanded_device_map = _expand_tensor_name(
        device_map, list(tensor_index.keys())
    )
    cpu_device_map = _cpu_device_map_or_none(expanded_device_map)

    if cpu_device_map is not None:
        start = time.perf_counter()
        tensor_device_offsets, tensor_copy_chunks = (
            calculate_tensor_device_offsets(cpu_device_map, tensor_data_index)
        )
        chunks = tensor_copy_chunks[_CPU_SLAB_DEVICE_ID]
        slab_size = max(
            (dst_offset + size for _, size, dst_offset, _ in chunks),
            default=1,
        )

        sm = shared_memory.SharedMemory(create=True, size=slab_size)
        try:
            t_load = time.perf_counter()
            ret = client.load_into_client_host_shm(
                model_path, sm.name, sm.size, chunks
            )
            if not ret:
                raise ValueError(
                    f"Failed to load model {model_path} into client host shm"
                )
            t_restore = time.perf_counter()
            state_dict = _restore_cpu_tensors(
                tensor_meta_index, sm, tensor_device_offsets
            )
            for v in state_dict.values():
                if isinstance(v, torch.Tensor):
                    v._sllm_host_shared_memory = sm
            with contextlib.suppress(FileNotFoundError):
                sm.unlink()
            logger.info(
                "Allocate shared memory took "
                f"{t_load - start:.4f}s "
                f"RPC took {t_restore - t_load:.4f}s "
                f"restore took {time.perf_counter() - t_restore:.4f}s "
                f"total took {time.perf_counter() - start:.4f}s"
            )
            return "", state_dict
        except Exception:
            with contextlib.suppress(Exception):
                sm.close()
            with contextlib.suppress(FileNotFoundError):
                sm.unlink()
            raise

    start = time.perf_counter()
    device_memory = calculate_device_memory(
        expanded_device_map, tensor_data_index
    )
    # logger.debug(f"calculate_device_memory {device_memory}")
    cuda_memory_ptrs = allocate_cuda_memory(device_memory)
    # cuda_memory_ptrs = { k: [v] for k,v in cuda_memory_ptrs.items()}
    cuda_memory_handles = get_cuda_memory_handles(cuda_memory_ptrs)
    device_uuid_map = get_device_uuid_map()
    # logger.debug(f"determine device_uuid_map {device_uuid_map}")
    tensor_device_offsets, tensor_copy_chunks = calculate_tensor_device_offsets(
        expanded_device_map, tensor_data_index
    )

    t_load = time.perf_counter()
    replica_uuid = _get_uuid()
    ret = client.load_into_gpu(
        model_path,
        replica_uuid,
        {
            device_uuid_map[device_id]: v
            for device_id, v in tensor_copy_chunks.items()
        },
        {
            device_uuid_map[device_id]: [v]
            for device_id, v in cuda_memory_handles.items()
        },
    )
    if not ret:
        raise ValueError(f"Failed to load model {model_path} into GPU")

    # load model state_dict
    t_restore = time.perf_counter()
    state_dict = restore_tensors(
        tensor_meta_index, cuda_memory_ptrs, tensor_device_offsets
    )
    # restore_tensors() may return many tensors that alias one cudaMalloc slab;
    # only the base view carries a cudaFree deleter. Callers (e.g. vLLM
    # ServerlessLLMLoader) may replace tensors via .to() — if the base view is
    # freed first, other views hit illegal memory. Clone tied / aliased views
    # so each tensor owns storage before any dtype cast or assignment.
    ptr_to_keys = collections.defaultdict(list)
    for key, tensor in state_dict.items():
        if tensor.is_cuda and tensor.numel():
            ptr_to_keys[tensor.data_ptr()].append(key)
    for keys in ptr_to_keys.values():
        if len(keys) > 1:
            for k in keys:
                state_dict[k] = state_dict[k].clone()

    logger.info(
        "Allocate cuda memory took "
        f"{t_load - start:.4f}s "
        f"RPC took {t_restore - t_load:.4f}s "
        f"restore took {time.perf_counter() - t_restore:.4f}s "
        f"total took {time.perf_counter() - start:.4f}s"
    )

    return replica_uuid, state_dict
