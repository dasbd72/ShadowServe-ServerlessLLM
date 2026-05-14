#!/usr/bin/env python3
# ---------------------------------------------------------------------------- #
# Compare weights loaded through sllm_store.torch.load_dict(cpu) with weights
# loaded from one or more safetensors files.
#
# Prerequisites:
#   1. Start sllm-store with the same STORAGE_PATH used here, e.g.:
#        export STORAGE_PATH=$HOME/models
#        sllm-store start --storage-path "$STORAGE_PATH"
#   2. The SLLM model directory contains tensor_index.json.
#   3. The reference path points to a .safetensors file or directory.
#
# Example:
#   cd sllm_store && python examples/compare_cpu_load_safetensors.py \
#       --model-path org/model-id \
#       --safetensors-path /path/to/original/hf/model
# ---------------------------------------------------------------------------- #
import argparse
import json
import os
import sys
import time
from typing import Dict, Iterable, Optional, Tuple

import torch

try:
    from safetensors import safe_open
except ImportError as exc:
    raise SystemExit(
        "Missing dependency: safetensors. Install it in your environment first."
    ) from exc


def _tensor_index_path_in_tree(model_dir: str) -> Optional[str]:
    direct = os.path.join(model_dir, "tensor_index.json")
    if os.path.isfile(direct):
        return direct
    under_rank = os.path.join(model_dir, "rank_0", "tensor_index.json")
    if os.path.isfile(under_rank):
        return under_rank
    return None


def _model_key_from_index_path(storage_path: str, index_path: str) -> str:
    model_dir = os.path.dirname(os.path.abspath(index_path))
    rel = os.path.relpath(model_dir, os.path.abspath(storage_path))
    return rel.replace("\\", "/")


def resolve_model_key_and_index(
    storage_path: str, model_path: str, backend: str
) -> Tuple[str, str]:
    """Locate tensor_index.json and return (model_key, absolute index_path).

    Both backends honor an optional rank_0/ subdirectory created by vLLM saves.
    model_key is the directory (relative to storage_path) that contains
    tensor_index.json, which is exactly what load_dict() expects.
    """
    model_path = model_path.strip(os.sep)
    if backend == "transformers":
        bases = [os.path.join(storage_path, model_path)]
    elif backend == "vllm":
        bases = [
            os.path.join(storage_path, "vllm", model_path),
            os.path.join(storage_path, model_path),
        ]
    else:
        raise ValueError(f"unknown backend: {backend}")

    for base in bases:
        found = _tensor_index_path_in_tree(base)
        if found:
            return _model_key_from_index_path(storage_path, found), found

    primary = bases[0]
    key = os.path.relpath(primary, storage_path).replace("\\", "/")
    return key, os.path.join(primary, "tensor_index.json")


def iter_safetensor_files(path: str) -> Iterable[str]:
    path = os.path.abspath(os.path.expanduser(path))
    if os.path.isfile(path):
        if not path.endswith(".safetensors"):
            raise ValueError(f"Not a safetensors file: {path}")
        yield path
        return

    for root, _, files in os.walk(path):
        for name in sorted(files):
            if name.endswith(".safetensors"):
                yield os.path.join(root, name)


def build_safetensors_index(path: str) -> Dict[str, str]:
    index: Dict[str, str] = {}
    for file_path in iter_safetensor_files(path):
        with safe_open(file_path, framework="pt", device="cpu") as f:
            for key in f.offset_keys():
                if key in index:
                    raise ValueError(
                        f"Duplicate safetensors key {key!r} in "
                        f"{index[key]} and {file_path}"
                    )
                index[key] = file_path
    if not index:
        raise ValueError(f"No .safetensors files found under {path}")
    return index


def tensor_difference(a: torch.Tensor, b: torch.Tensor) -> Tuple[bool, float]:
    if a.dtype == torch.bool or b.dtype == torch.bool:
        different = (a != b).sum().item()
        return different == 0, float(different)
    if not (a.is_floating_point() or b.is_floating_point()):
        different = (a != b).sum().item()
        return different == 0, float(different)

    a32 = a.detach().to(torch.float32)
    b32 = b.detach().to(torch.float32)
    max_abs = (a32 - b32).abs().max().item() if a.numel() else 0.0
    return torch.equal(a, b), float(max_abs)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare SLLM CPU-loaded weights against safetensors weights."
        )
    )
    parser.add_argument(
        "--backend",
        choices=("transformers", "vllm"),
        default="transformers",
        help="SLLM storage layout used to resolve tensor_index.json.",
    )
    parser.add_argument(
        "--model-path",
        required=True,
        help="SLLM model path relative to STORAGE_PATH, e.g. org/model-id.",
    )
    parser.add_argument(
        "--safetensors-path",
        required=True,
        help="Reference .safetensors file or directory containing shards.",
    )
    parser.add_argument(
        "--storage-path",
        default=os.environ.get("STORAGE_PATH", os.path.expanduser("~/models")),
        help="SLLM storage root. Must match the running server.",
    )
    parser.add_argument(
        "--register",
        action="store_true",
        help="Register the model first if the server requires registration.",
    )
    parser.add_argument(
        "--max-mismatches",
        type=int,
        default=20,
        help="Stop after reporting this many mismatches.",
    )
    args = parser.parse_args()

    storage_path = os.path.abspath(os.path.expanduser(args.storage_path))
    os.environ["STORAGE_PATH"] = storage_path

    model_key, index_path = resolve_model_key_and_index(
        storage_path, args.model_path, args.backend
    )
    if not os.path.isfile(index_path):
        mp = args.model_path.strip(os.sep)
        if args.backend == "vllm":
            tried = (
                f"  1) {os.path.join(storage_path, 'vllm', mp)}\n"
                f"  2) {os.path.join(storage_path, mp)}\n"
                "(each also checked under a rank_0/ subdirectory)"
            )
            print(
                f"Missing SLLM tensor_index.json; tried:\n{tried}",
                file=sys.stderr,
            )
        else:
            print(
                f"Missing SLLM tensor_index.json at {index_path} "
                f"(also checked rank_0/ subdir)",
                file=sys.stderr,
            )
        sys.exit(1)

    with open(index_path, "r", encoding="utf-8") as f:
        tensor_index = json.load(f)

    print(f"SLLM model_key={model_key!r}, tensors={len(tensor_index)}")
    safe_index = build_safetensors_index(args.safetensors_path)
    print(f"safetensors tensors={len(safe_index)}")

    from sllm_store import torch as sllm_torch
    from sllm_store.client import SllmStoreClient

    client = SllmStoreClient()
    if args.register:
        size = client.register_model(model_key)
        print(f"register_model({model_key!r}) -> {size}")

    t0 = time.time()
    sllm_state = sllm_torch.load_dict(model_key, {"": "cpu"}, storage_path)
    print(f"SLLM CPU load took {time.time() - t0:.2f}s")

    missing_in_safe = sorted(set(sllm_state) - set(safe_index))
    missing_in_sllm = sorted(set(safe_index) - set(sllm_state))
    mismatches = []
    compared = 0

    if missing_in_safe:
        mismatches.append(
            f"{len(missing_in_safe)} SLLM keys missing from safetensors; "
            f"first={missing_in_safe[:5]}"
        )
    if missing_in_sllm:
        mismatches.append(
            f"{len(missing_in_sllm)} safetensors keys missing from SLLM; "
            f"first={missing_in_sllm[:5]}"
        )

    file_to_keys: Dict[str, list] = {}
    for key in sorted(set(sllm_state) & set(safe_index)):
        file_to_keys.setdefault(safe_index[key], []).append(key)

    for file_path, keys in file_to_keys.items():
        with safe_open(file_path, framework="pt", device="cpu") as f:
            for key in keys:
                sllm_tensor = sllm_state[key].detach().cpu()
                safe_tensor = f.get_tensor(key)

                compared += 1
                if sllm_tensor.shape != safe_tensor.shape:
                    mismatches.append(
                        f"{key}: shape {tuple(sllm_tensor.shape)} != "
                        f"{tuple(safe_tensor.shape)}"
                    )
                elif sllm_tensor.dtype != safe_tensor.dtype:
                    mismatches.append(
                        f"{key}: dtype {sllm_tensor.dtype} != "
                        f"{safe_tensor.dtype}"
                    )
                else:
                    equal, diff = tensor_difference(sllm_tensor, safe_tensor)
                    if not equal:
                        unit = (
                            "max_abs"
                            if sllm_tensor.is_floating_point()
                            else "different elements"
                        )
                        mismatches.append(
                            f"{key}: values differ, {unit}={diff}"
                        )

                if len(mismatches) >= args.max_mismatches:
                    break
        if len(mismatches) >= args.max_mismatches:
            break

    print(f"Compared {compared} common tensors")
    if mismatches:
        print("MISMATCH")
        for item in mismatches[: args.max_mismatches]:
            print(f"  - {item}")
        sys.exit(1)

    print("OK: SLLM CPU-loaded weights match safetensors reference.")


if __name__ == "__main__":
    main()
