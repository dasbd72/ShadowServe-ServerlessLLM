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
    log_prefix: Path
    max_model_len: int | None = None
    shadow_num_cpus: int | None = None
    additional_blocks_per_request: int | None = None

    @property
    def sllm_log(self) -> Path:
        return self.log_prefix.parent / f"{self.log_prefix.name}_sllm.log"

    @property
    def client_log_base(self) -> Path:
        return self.log_prefix.parent / f"{self.log_prefix.name}_client"


def _deploy_config(
    scenario: Scenario,
    *,
    min_instances: int,
    max_instances: int,
) -> dict[str, Any]:
    backend_config: dict[str, Any] = {
        "pretrained_model_name_or_path": scenario.model,
        "torch_dtype": "bfloat16",
        "enforce_eager": False,
        "enable_prefix_caching": True,
        "block_size": 16,
    }
    if scenario.max_model_len is not None:
        backend_config["max_model_len"] = scenario.max_model_len
    if scenario.shadow:
        backend_config["shadow_sender_enabled"] = True
        backend_config["shadow_receiver_enabled"] = True

    config: dict[str, Any] = {
        "model": scenario.model,
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
    if scenario.shadow:
        if (
            scenario.shadow_num_cpus is None
            or scenario.additional_blocks_per_request is None
        ):
            raise ValueError(
                f"Shadow scenario {scenario.name!r} requires "
                "shadow_num_cpus and additional_blocks_per_request"
            )
        config["router_config"] = {
            "target": 2,
            "shadow_num_cpus": scenario.shadow_num_cpus,
            "kvhts_ipc_prefix": "/tmp/vllm-shadow-kvhts",
            "kvstc_ipc_prefix": "/tmp/vllm-shadow-kvstc",
            "tksth_ipc_prefix": "/tmp/vllm-shadow-tksth",
            "additional_blocks_per_request": scenario.additional_blocks_per_request,
        }
    return config


@contextlib.contextmanager
def _temp_deploy_config(
    scenario: Scenario,
    *,
    min_instances: int,
    max_instances: int,
):
    config = _deploy_config(
        scenario,
        min_instances=min_instances,
        max_instances=max_instances,
    )
    prefix = "config-vllm-shadow-" if scenario.shadow else "config-vllm-"
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
        *,
        num_workers: int,
        num_cpus_per_worker: int,
        cuda_devices: str,
        num_runs: int,
        dataset_path: Path,
        min_instances: int,
        max_instances: int,
    ) -> None:
        self.num_workers = num_workers
        self.num_cpus_per_worker = num_cpus_per_worker
        self.cuda_devices = cuda_devices
        self.num_runs = num_runs
        self.dataset_path = dataset_path
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

        max_tokens_list = [
            4096,
            8192,
            16384,
            32768,
            65536,
        ]
        for max_tokens in max_tokens_list:
            scenarios.append(
                Scenario(
                    name=f"baseline-qwen3-0.6b_{max_tokens}",
                    model="Qwen/Qwen3-0.6B",
                    shadow=False,
                    max_tokens=max_tokens,
                    log_prefix=self.logs_dir
                    / f"e2e_qwen3-0.6b_{max_tokens}_baseline",
                )
            )
            scenarios.append(
                Scenario(
                    name=f"shadow-qwen3-0.6b_{max_tokens}",
                    model="Qwen/Qwen3-0.6B",
                    shadow=True,
                    max_tokens=max_tokens,
                    log_prefix=self.logs_dir
                    / f"e2e_qwen3-0.6b_{max_tokens}_shadow",
                    shadow_num_cpus=64,
                    additional_blocks_per_request=50,
                )
            )

        max_tokens_list = [
            2048,
            4096,
            8192,
            16384,
        ]
        for max_tokens in max_tokens_list:
            scenarios.append(
                Scenario(
                    name=f"baseline-qwen3-4b_{max_tokens}",
                    model="Qwen/Qwen3-4B",
                    shadow=False,
                    max_tokens=max_tokens,
                    log_prefix=self.logs_dir
                    / f"e2e_qwen3-4b_{max_tokens}_baseline",
                )
            )
            scenarios.append(
                Scenario(
                    name=f"shadow-qwen3-4b_{max_tokens}",
                    model="Qwen/Qwen3-4B",
                    shadow=True,
                    max_tokens=max_tokens,
                    log_prefix=self.logs_dir
                    / f"e2e_qwen3-4b_{max_tokens}_shadow",
                    shadow_num_cpus=24,
                    additional_blocks_per_request=20,
                )
            )
        return scenarios

    def _run_scenario(self, scenario: Scenario) -> None:
        started = datetime.now().astimezone().isoformat(timespec="seconds")
        logger.info("==> %s starting scenario %s", started, scenario.name)

        scenario.log_prefix.parent.mkdir(parents=True, exist_ok=True)

        with (
            _temp_deploy_config(
                scenario,
                min_instances=self.min_instances,
                max_instances=self.max_instances,
            ) as config,
            self._sllm(
                config, scenario.sllm_log, enable_shadow=scenario.shadow
            ),
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
    def _sllm(
        self,
        config: Path,
        log_file: Path,
        *,
        enable_shadow: bool = False,
    ) -> Iterator[None]:
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
        if enable_shadow:
            start_cmd.append("--enable-shadow")
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

    context = Context(
        num_workers=args.num_workers,
        num_cpus_per_worker=args.num_cpus_per_worker,
        cuda_devices=args.cuda_devices,
        num_runs=args.num_runs,
        dataset_path=args.dataset_path,
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
