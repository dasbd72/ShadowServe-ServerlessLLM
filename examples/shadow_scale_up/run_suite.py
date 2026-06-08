# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run shadow scale-up benchmark suite (baseline + shadow per model)."""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import os
import re
import signal
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger("run_suite.py")

HOME = Path.home()

ROOT = Path(__file__).resolve().parents[2]
LOGS_DIR = ROOT / "logs"
START_SLLM_SCRIPT = ROOT / "examples" / "start_sllm.py"
EXAMPLE_DIR = Path(__file__).resolve().parent
WORKLOAD_CLIENT_SCRIPT = EXAMPLE_DIR / "workload_client.py"

SLLM_PORT = 8343
SLLM_URL = f"http://127.0.0.1:{SLLM_PORT}"

DEFAULT_NUM_WORKERS = 2
DEFAULT_NUM_CPUS_PER_WORKER = 64
DEFAULT_CUDA_DEVICES = "0,1"
DEFAULT_MAX_TOKENS_QWEN3_0_6B = 16384
DEFAULT_MAX_TOKENS_QWEN3_4B = 8192
DEFAULT_MAX_TOKENS_QWEN3_8B = 4096
DEFAULT_NUM_RUNS = 4
DEFAULT_MIN_INSTANCES = 2
DEFAULT_MAX_INSTANCES = 4
INFERENCE_PROBE_MAX_TOKENS = 1
DEFAULT_DATASET_PATH = (
    HOME / "datasets" / "ShareGPT_V3_unfiltered_cleaned_split.json"
)

HEALTH_TIMEOUT_SEC = 300.0
DEPLOY_TIMEOUT_SEC = 900.0
TRIGGER_TIMEOUT_SEC = 600.0

MODEL_SLUGS: dict[str, str] = {
    "Qwen/Qwen3-0.6B": "qwen3-0.6b",
    "Qwen/Qwen3-4B": "qwen3-4b",
    "Qwen/Qwen3-8B": "qwen3-8b",
}

MODEL_DEPLOY_SETTINGS: dict[str, dict[str, Any]] = {
    "Qwen/Qwen3-0.6B": {
        "shadow_num_cpus": 16,
        "additional_blocks_per_request": 50,
    },
    "Qwen/Qwen3-4B": {
        "max_model_len": 20480,
        "shadow_num_cpus": 12,
        "additional_blocks_per_request": 20,
    },
    "Qwen/Qwen3-8B": {
        "max_model_len": 10240,
        "shadow_num_cpus": 12,
        "additional_blocks_per_request": 20,
    },
}

DEFAULT_MAX_TOKENS: dict[str, int] = {
    "Qwen/Qwen3-0.6B": DEFAULT_MAX_TOKENS_QWEN3_0_6B,
    "Qwen/Qwen3-4B": DEFAULT_MAX_TOKENS_QWEN3_4B,
    "Qwen/Qwen3-8B": DEFAULT_MAX_TOKENS_QWEN3_8B,
}

# RoundRobinRouter logs this after init_backend completes (see roundrobin_router.py).
_BACKEND_DEPLOY_READY_RE = re.compile(
    r"Initialized backend for instance .+ for model {model}"
)
_ALL_BACKEND_DEPLOY_READY_RE = re.compile(
    r"Initialized all backend for instance .+ for model {model}"
)
# Logged when the router handler is running (min_instances may still be 0).
_MODEL_HANDLER_READY_RE = re.compile(r"Started handler for model {model}")


@dataclass(frozen=True)
class Scenario:
    name: str
    model: str
    shadow: bool
    max_tokens: int
    sllm_log: Path
    client_log_base: Path


def _deploy_config(
    model: str,
    *,
    shadow: bool,
    min_instances: int,
    max_instances: int,
) -> dict[str, Any]:
    settings = MODEL_DEPLOY_SETTINGS[model]
    backend_config: dict[str, Any] = {
        "pretrained_model_name_or_path": model,
        "torch_dtype": "bfloat16",
        "enforce_eager": False,
        "enable_prefix_caching": True,
        "block_size": 16,
    }
    if max_model_len := settings.get("max_model_len"):
        backend_config["max_model_len"] = max_model_len
    if shadow:
        backend_config["shadow_sender_enabled"] = True
        backend_config["shadow_receiver_enabled"] = True

    config: dict[str, Any] = {
        "model": model,
        "backend": "vllm",
        "num_gpus": 1,
        "auto_scaling_config": {
            "metric": "concurrency",
            "target": 2,
            "min_instances": min_instances,
            "max_instances": max_instances,
        },
        "backend_config": backend_config,
    }
    if shadow:
        config["enable_shadow"] = True
        config["router_config"] = {
            "target": 2,
            "shadow_num_cpus": settings["shadow_num_cpus"],
            "kvhts_ipc_prefix": "/tmp/vllm-shadow-kvhts",
            "kvstc_ipc_prefix": "/tmp/vllm-shadow-kvstc",
            "tksth_ipc_prefix": "/tmp/vllm-shadow-tksth",
            "additional_blocks_per_request": settings[
                "additional_blocks_per_request"
            ],
        }
    return config


@contextlib.contextmanager
def _temp_deploy_config(
    model: str,
    *,
    shadow: bool,
    min_instances: int,
    max_instances: int,
):
    config = _deploy_config(
        model,
        shadow=shadow,
        min_instances=min_instances,
        max_instances=max_instances,
    )
    prefix = "config-vllm-shadow-" if shadow else "config-vllm-"
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".json",
        prefix=prefix,
        delete=False,
        encoding="utf-8",
    ) as config_fh:
        json.dump(config, config_fh, indent=4)
        config_fh.write("\n")
        path = Path(config_fh.name)
    try:
        yield path
    finally:
        path.unlink(missing_ok=True)


class Context:
    def __init__(
        self,
        models: list[str],
        *,
        num_workers: int,
        num_cpus_per_worker: int,
        cuda_devices: str,
        num_runs: int,
        dataset_path: Path,
        max_tokens_by_model: dict[str, int],
        min_instances: int,
        max_instances: int,
    ) -> None:
        self.models = models
        self.num_workers = num_workers
        self.num_cpus_per_worker = num_cpus_per_worker
        self.cuda_devices = cuda_devices
        self.num_runs = num_runs
        self.dataset_path = dataset_path
        self.max_tokens_by_model = max_tokens_by_model
        self.min_instances = min_instances
        self.max_instances = max_instances

        self.logs_dir = LOGS_DIR
        self.sllm_url = SLLM_URL

        self._start_sllm_proc: subprocess.Popen | None = None
        self._shutdown_requested = False

    def run(self) -> None:
        os.chdir(ROOT)
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        for scenario in self._suite_scenarios():
            if self._shutdown_requested:
                break
            self._run_scenario(scenario)
        logger.info("All runs complete. Logs in %s/", self.logs_dir)

    def _suite_scenarios(self) -> list[Scenario]:
        scenarios: list[Scenario] = []
        for model in self.models:
            slug = MODEL_SLUGS[model]
            max_tokens = self.max_tokens_by_model[model]
            scenarios.append(
                Scenario(
                    name=f"baseline-{slug}",
                    model=model,
                    shadow=False,
                    max_tokens=max_tokens,
                    sllm_log=self.logs_dir
                    / f"trigger_scale_up_baseline_sllm_{slug}.log",
                    client_log_base=self.logs_dir
                    / f"trigger_scale_up_baseline_client_{slug}",
                )
            )
            scenarios.append(
                Scenario(
                    name=f"shadow-{slug}",
                    model=model,
                    shadow=True,
                    max_tokens=max_tokens,
                    sllm_log=self.logs_dir
                    / f"trigger_scale_up_shadow_sllm_{slug}.log",
                    client_log_base=self.logs_dir
                    / f"trigger_scale_up_shadow_client_{slug}",
                )
            )
        return scenarios

    def _run_scenario(self, scenario: Scenario) -> None:
        started = datetime.now().astimezone().isoformat(timespec="seconds")
        logger.info("==> %s starting scenario %s", started, scenario.name)

        scenario.sllm_log.parent.mkdir(parents=True, exist_ok=True)

        with (
            _temp_deploy_config(
                scenario.model,
                shadow=scenario.shadow,
                min_instances=self.min_instances,
                max_instances=self.max_instances,
            ) as config,
            self._sllm(config, scenario.sllm_log),
        ):
            self._wait_sllm_health()
            self._wait_model_deployed(
                scenario.sllm_log,
                scenario.model,
                min_instances=self.min_instances,
                require_shadow=scenario.shadow,
            )
            self._wait_model_inference_ready(scenario.model)
            logger.info(
                "Model %s deployed; starting workload runs", scenario.model
            )

            for run_idx in range(1, self.num_runs + 1):
                if self._shutdown_requested:
                    break
                log_file = Path(f"{scenario.client_log_base}_run{run_idx}.log")
                label = f"run {run_idx}"
                self._run_trigger(
                    scenario.model,
                    scenario.max_tokens,
                    log_file,
                    label=label,
                )

        finished = datetime.now().astimezone().isoformat(timespec="seconds")
        logger.info("==> %s finished scenario %s", finished, scenario.name)

    @contextlib.contextmanager
    def _sllm(self, config: Path, log_file: Path) -> Iterator[None]:
        start_cmd = [
            sys.executable,
            str(START_SLLM_SCRIPT),
            "--num-workers",
            str(self.num_workers),
            "--num-cpus-per-worker",
            str(self.num_cpus_per_worker),
            "--cuda-devices",
            self.cuda_devices,
            "--deploy",
            "--config",
            str(config),
            "--log-file",
            str(log_file),
        ]
        proc = subprocess.Popen(start_cmd)
        self._start_sllm_proc = proc
        try:
            yield
        finally:
            self._start_sllm_proc = None
            proc.terminate()
            proc.wait(timeout=300)

    def _wait_sllm_health(self) -> None:
        url = f"{self.sllm_url}/health"
        deadline = time.monotonic() + HEALTH_TIMEOUT_SEC
        while time.monotonic() < deadline:
            if self._shutdown_requested:
                raise RuntimeError("Shutdown requested while waiting for SLLM")
            with contextlib.suppress(urllib.error.URLError, TimeoutError):
                with urllib.request.urlopen(url, timeout=5) as resp:
                    if resp.status == 200:
                        logger.info("SLLM HTTP ready at %s", url)
                        return
            time.sleep(2)
        raise RuntimeError(
            f"SLLM server at {url} did not become ready within "
            f"{HEALTH_TIMEOUT_SEC:.0f}s"
        )

    def _wait_model_deployed(
        self,
        log_file: Path,
        model: str,
        *,
        min_instances: int,
        require_shadow: bool = False,
    ) -> None:
        if min_instances == 0:
            ready_re = re.compile(
                _MODEL_HANDLER_READY_RE.pattern.format(model=re.escape(model))
            )
            required_instances = 1
            ready_label = "model handler"
        elif require_shadow:
            ready_re = re.compile(
                _ALL_BACKEND_DEPLOY_READY_RE.pattern.format(
                    model=re.escape(model)
                )
            )
            required_instances = min_instances
            ready_label = "GPU instances (shadow)"
        else:
            ready_re = re.compile(
                _BACKEND_DEPLOY_READY_RE.pattern.format(model=re.escape(model))
            )
            required_instances = min_instances
            ready_label = "GPU instances"
        ready_count = 0
        deadline = time.monotonic() + DEPLOY_TIMEOUT_SEC
        read_offset = 0
        while time.monotonic() < deadline:
            if self._shutdown_requested:
                raise RuntimeError(
                    "Shutdown requested while waiting for deploy"
                )
            if log_file.is_file():
                with log_file.open(
                    encoding="utf-8", errors="replace"
                ) as log_fh:
                    log_fh.seek(read_offset)
                    chunk = log_fh.read()
                    read_offset = log_fh.tell()
                ready_count += len(ready_re.findall(chunk))
                if ready_count >= required_instances:
                    logger.info(
                        "%s ready for %s: %d/%d (see %s)",
                        ready_label,
                        model,
                        ready_count,
                        required_instances,
                        log_file,
                    )
                    return
            time.sleep(2)
        raise RuntimeError(
            f"Deploy for {model} did not reach {required_instances} "
            f"{ready_label} within {DEPLOY_TIMEOUT_SEC:.0f}s "
            f"(saw {ready_count}; see {log_file})"
        )

    def _wait_model_inference_ready(self, model: str) -> None:
        url = f"{self.sllm_url}/v1/chat/completions"
        body = json.dumps(
            {
                "model": model,
                "messages": [{"role": "user", "content": "ping"}],
                "max_tokens": INFERENCE_PROBE_MAX_TOKENS,
                "temperature": 0.0,
            }
        ).encode("utf-8")
        deadline = time.monotonic() + DEPLOY_TIMEOUT_SEC
        while time.monotonic() < deadline:
            if self._shutdown_requested:
                raise RuntimeError(
                    "Shutdown requested while waiting for inference readiness"
                )
            req = urllib.request.Request(
                url,
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with contextlib.suppress(
                urllib.error.HTTPError,
                urllib.error.URLError,
                TimeoutError,
            ):
                with urllib.request.urlopen(req, timeout=60) as resp:
                    if resp.status == 200:
                        logger.info(
                            "Model %s accepting inference requests", model
                        )
                        return
            time.sleep(2)
        raise RuntimeError(
            f"Model {model} did not accept inference requests within "
            f"{DEPLOY_TIMEOUT_SEC:.0f}s"
        )

    def _run_trigger(
        self,
        model: str,
        max_tokens: int,
        log_file: Path,
        *,
        label: str,
    ) -> None:
        started = datetime.now().astimezone().isoformat(timespec="seconds")
        logger.info(
            "==> %s trigger (%s): model=%s max_tokens=%s -> %s",
            started,
            label,
            model,
            max_tokens,
            log_file,
        )
        log_file.parent.mkdir(parents=True, exist_ok=True)

        trigger_cmd = [
            sys.executable,
            str(WORKLOAD_CLIENT_SCRIPT),
            "--url",
            self.sllm_url,
            "--model",
            model,
            "--max-tokens",
            str(max_tokens),
            "--timeout",
            str(TRIGGER_TIMEOUT_SEC),
            "--dataset-path",
            str(self.dataset_path),
        ]

        with log_file.open("w", encoding="utf-8") as log_fh:
            log_fh.flush()
            result = subprocess.run(
                trigger_cmd,
                stdout=log_fh,
                stderr=subprocess.STDOUT,
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"workload_client.py exited with code {result.returncode} "
                    f"(see {log_file})"
                )

        finished = datetime.now().astimezone().isoformat(timespec="seconds")
        logger.info("==> %s trigger finished -> %s", finished, log_file)

    def shutdown(self) -> None:
        if self._shutdown_requested:
            return
        logger.info("Shutting down SLLM server and Ray cluster")
        self._shutdown_requested = True
        proc = self._start_sllm_proc
        self._start_sllm_proc = None
        if proc is not None:
            proc.terminate()
            proc.wait(timeout=300)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--model",
        dest="models",
        action="append",
        choices=sorted(MODEL_SLUGS),
        metavar="MODEL",
        help=(
            "Model to benchmark (repeatable). "
            f"Choices: {', '.join(sorted(MODEL_SLUGS))}. "
            "Default: all models."
        ),
    )
    p.add_argument(
        "--num-workers",
        type=int,
        default=DEFAULT_NUM_WORKERS,
        help="Number of Ray GPU workers (passed to start_sllm.py)",
    )
    p.add_argument(
        "--num-cpus-per-worker",
        type=int,
        default=DEFAULT_NUM_CPUS_PER_WORKER,
        help="CPUs advertised per Ray worker (passed to start_sllm.py)",
    )
    p.add_argument(
        "--cuda-devices",
        default=DEFAULT_CUDA_DEVICES,
        help="Comma-separated GPU device IDs for Ray workers (passed to start_sllm.py)",
    )
    p.add_argument(
        "--dataset-path",
        type=Path,
        default=Path(
            os.environ.get("TRIGGER_DATASET_PATH", str(DEFAULT_DATASET_PATH))
        ),
        help=(
            "ShareGPT JSON dataset for workload_client.py "
            f"(default: {DEFAULT_DATASET_PATH}, or TRIGGER_DATASET_PATH)"
        ),
    )
    p.add_argument(
        "--max-tokens-qwen3-0-6b",
        type=int,
        default=int(
            os.environ.get(
                "TRIGGER_MAX_TOKENS_QWEN3_0_6B", DEFAULT_MAX_TOKENS_QWEN3_0_6B
            )
        ),
        help=f"max_tokens for Qwen3-0.6B (default: {DEFAULT_MAX_TOKENS_QWEN3_0_6B})",
    )
    p.add_argument(
        "--max-tokens-qwen3-4b",
        type=int,
        default=int(
            os.environ.get(
                "TRIGGER_MAX_TOKENS_QWEN3_4B", DEFAULT_MAX_TOKENS_QWEN3_4B
            )
        ),
        help=f"max_tokens for Qwen3-4B (default: {DEFAULT_MAX_TOKENS_QWEN3_4B})",
    )
    p.add_argument(
        "--max-tokens-qwen3-8b",
        type=int,
        default=int(
            os.environ.get(
                "TRIGGER_MAX_TOKENS_QWEN3_8B", DEFAULT_MAX_TOKENS_QWEN3_8B
            )
        ),
        help=f"max_tokens for Qwen3-8B (default: {DEFAULT_MAX_TOKENS_QWEN3_8B})",
    )
    p.add_argument(
        "--num-runs",
        type=int,
        default=DEFAULT_NUM_RUNS,
        help="Runs per scenario",
    )
    p.add_argument(
        "--min-instances",
        type=int,
        default=int(
            os.environ.get("TRIGGER_MIN_INSTANCES", DEFAULT_MIN_INSTANCES)
        ),
        help=(
            "Minimum model instances in deploy auto_scaling_config "
            f"(default: {DEFAULT_MIN_INSTANCES}, or TRIGGER_MIN_INSTANCES)"
        ),
    )
    p.add_argument(
        "--max-instances",
        type=int,
        default=int(
            os.environ.get("TRIGGER_MAX_INSTANCES", DEFAULT_MAX_INSTANCES)
        ),
        help=(
            "Maximum model instances in deploy auto_scaling_config "
            f"(default: {DEFAULT_MAX_INSTANCES}, or TRIGGER_MAX_INSTANCES)"
        ),
    )
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()

    models = args.models or sorted(MODEL_SLUGS)
    max_tokens_by_model = {
        "Qwen/Qwen3-0.6B": args.max_tokens_qwen3_0_6b,
        "Qwen/Qwen3-4B": args.max_tokens_qwen3_4b,
        "Qwen/Qwen3-8B": args.max_tokens_qwen3_8b,
    }
    context = Context(
        num_workers=args.num_workers,
        num_cpus_per_worker=args.num_cpus_per_worker,
        cuda_devices=args.cuda_devices,
        models=models,
        num_runs=args.num_runs,
        dataset_path=args.dataset_path,
        max_tokens_by_model=max_tokens_by_model,
        min_instances=args.min_instances,
        max_instances=args.max_instances,
    )

    def handle_signal(signum, frame) -> None:
        logger.error("Received signal %s; shutting down run_suite.py", signum)
        context.shutdown()

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    try:
        context.run()
    except Exception as exc:
        logger.error("run_suite failed: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
