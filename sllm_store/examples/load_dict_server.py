#!/usr/bin/env python3
# ---------------------------------------------------------------------------- #
#  Example: call load_dict while a sllm-store gRPC server is running.
#
#  Prerequisites:
#    1. Same STORAGE_PATH on the client as on the server (tensor_index.json is
#       read locally; the server stages weights from disk into its host pool).
#    2. Start the store, e.g.:
#         export STORAGE_PATH=$HOME/models
#         sllm-store start --storage-path "$STORAGE_PATH"
#    3. For --backend transformers:
#       STORAGE_PATH/<model-path>/tensor_index.json
#       For --backend vllm: under STORAGE_PATH/vllm/<model>/ or .../<model>/ ,
#       tensor_index.json is often under rank_0/
#       (ServerlessLLMLoader.save_model).
#
#  Note: SllmStoreClient in sllm_store.torch.load_dict uses 127.0.0.1:8073. Use
#  SSH port-forward if the server runs elsewhere.
#
#  Example (GPU weight shard on cuda:0):
#    export STORAGE_PATH=$HOME/models
#    cd sllm_store && PYTHONPATH=. python examples/test_load_dict_server.py \\
#        --model-path org/model-id
#
#  Example (CPU-only load path in torch.load_dict):
#    cd sllm_store && PYTHONPATH=. python examples/test_load_dict_server.py \\
#        --model-path org/model-id --cpu
#
#  vLLM: weights under vllm/ from `sllm-store save --backend vllm`, or flat
#  layouts from examples:
#    ... --backend vllm --model-path org/model-id
# ---------------------------------------------------------------------------- #
import argparse
from typing import Optional, Tuple
import json
import os
import sys
import time

import torch


def _tensor_index_path_in_tree(model_dir: str) -> Optional[str]:
    """Return tensor_index.json path if present in model_dir."""
    direct = os.path.join(model_dir, "tensor_index.json")
    if os.path.isfile(direct):
        return direct
    under_rank = os.path.join(model_dir, "rank_0", "tensor_index.json")
    if os.path.isfile(under_rank):
        return under_rank
    return None


def _model_key_from_index_path(storage_path: str, index_path: str) -> str:
    """Directory containing tensor_index.json, relative to storage_path."""
    d = os.path.dirname(os.path.abspath(index_path))
    sp = os.path.abspath(storage_path)
    rel = os.path.relpath(d, sp)
    return rel.replace("\\", "/")


def resolve_model_key_and_index(
    storage_path: str, model_path: str, backend: str
) -> Tuple[str, str]:
    """Return (model_key_for_grpc/load_dict, path_to_tensor_index.json)."""
    model_path = model_path.strip(os.sep)
    if backend == "transformers":
        key = model_path
        index_path = os.path.join(storage_path, key, "tensor_index.json")
        return key, index_path

    if backend == "vllm":
        for base in (
            os.path.join(storage_path, "vllm", model_path),
            os.path.join(storage_path, model_path),
        ):
            found = _tensor_index_path_in_tree(base)
            if found:
                return _model_key_from_index_path(storage_path, found), found
        # placeholder for missing-file error (prefer sllm-store path)
        key_cli = os.path.join("vllm", model_path)
        return key_cli, os.path.join(storage_path, key_cli, "tensor_index.json")

    raise ValueError(f"unknown backend: {backend}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test load_dict against a running sllm-store server."
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=("transformers", "vllm"),
        default="transformers",
        help=(
            "transformers: STORAGE_PATH/<model-path>/ . vllm: sllm-store and "
            "examples layouts; looks for tensor_index.json in <model>/ or "
            "<model>/rank_0/."
        ),
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help=(
            "Path relative to backend root under STORAGE_PATH "
            "(e.g. org/model-id)."
        ),
    )
    parser.add_argument(
        "--storage-path",
        type=str,
        default=os.environ.get("STORAGE_PATH", os.path.expanduser("~/models")),
        help="Root directory containing the model folder (must match server).",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Use CPU-only device_map (reads tensor.data under the model dir).",
    )
    parser.add_argument(
        "--register",
        action="store_true",
        help=(
            "If the server was started with --registration-required, "
            "register first."
        ),
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
            print(
                "Missing tensor_index.json. For --backend vllm this script "
                "checks (with optional rank_0/ from sllm-store save):\n"
                f"  1) {os.path.join(storage_path, 'vllm', mp)}\n"
                f"  2) {os.path.join(storage_path, mp)}",
                file=sys.stderr,
            )
        else:
            print(f"Missing {index_path}", file=sys.stderr)
        sys.exit(1)

    with open(index_path, "r", encoding="utf-8") as f:
        tensor_index = json.load(f)
    print(
        f"backend={args.backend!r} model_key={model_key!r}: "
        f"{len(tensor_index)} tensors in tensor_index.json"
    )

    if args.cpu:
        device_map = {"": "cpu"}
    else:
        if not torch.cuda.is_available():
            print(
                "CUDA not available; use --cpu or install a CUDA build.",
                file=sys.stderr,
            )
            sys.exit(1)
        torch.cuda.set_device(0)
        device_map = {"": 0}

    from sllm_store.client import SllmStoreClient
    from sllm_store.torch import load_dict

    client = SllmStoreClient()
    if args.register:
        size = client.register_model(model_key)
        print(f"register_model({model_key!r}) -> {size}")

    t0 = time.time()
    state_dict = load_dict(model_key, device_map, storage_path)
    elapsed = time.time() - t0

    sample = list(state_dict.keys())[:5]
    devices = {k: str(state_dict[k].device) for k in sample}
    dtypes = {k: str(state_dict[k].dtype) for k in sample}
    print(f"load_dict finished in {elapsed:.2f}s, {len(state_dict)} tensors")
    print("sample devices:", devices)
    print("sample dtypes:", dtypes)


if __name__ == "__main__":
    main()
