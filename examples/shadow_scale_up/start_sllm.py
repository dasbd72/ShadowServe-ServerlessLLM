#!/usr/bin/env python3
"""Start a local Ray cluster + SLLM server for development."""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

RAY_HEAD_ADDRESS = "127.0.0.1:6379"
SLLM_PORT = 8343
DEFAULT_SLLM_LOG_FILE = Path("logs/sllm.log")


class Context:
    def __init__(
        self,
        deploy: bool,
        config: Path | None,
        log_file: Path = DEFAULT_SLLM_LOG_FILE,
    ):
        self.deploy = deploy
        self.config = config
        self.log_file = log_file
        self.sllm_log_file = None
        self.head_process: subprocess.Popen | None = None
        self.worker_processes: list[subprocess.Popen] = []
        self.sllm_process: subprocess.Popen | None = None

    def run(self) -> None:
        head_env = os.environ.copy()
        head_env["CUDA_VISIBLE_DEVICES"] = ""
        head_resources = {"control_node": 1}
        head_cmd = [
            "ray",
            "start",
            "--head",
            "--port=6379",
            "--num-cpus=8",
            "--num-gpus=0",
            "--resources",
            json.dumps(head_resources),
            "--block",
        ]

        cuda_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "0,1").split(",")
        if len(cuda_devices) < 2:
            raise ValueError(
                "Two Ray workers require two CUDA devices. "
                "Set CUDA_VISIBLE_DEVICES=0,1 (or similar)."
            )
        worker_num_cpus = 64

        sllm_env = os.environ.copy()
        sllm_env["RAY_HEAD_ADDRESS"] = RAY_HEAD_ADDRESS

        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        self.sllm_log_file = open(self.log_file, "w")

        t_start_ray = time.perf_counter()
        self.head_process = subprocess.Popen(
            head_cmd,
            env=head_env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        self.wait_for_ray_ready(self.head_process, RAY_HEAD_ADDRESS, head_env)
        base_worker_env = os.environ.copy()
        for worker_id, cuda_device in enumerate(cuda_devices[:2]):
            proc = self.start_ray_worker(
                worker_id,
                cuda_device.strip(),
                worker_num_cpus,
                base_worker_env,
            )
            self.worker_processes.append(proc)
        elapsed_start_ray = time.perf_counter() - t_start_ray
        print(
            f"Ray cluster started in {elapsed_start_ray} seconds "
            f"(workers=2, gpus_per_worker=1, cpus_per_worker={worker_num_cpus}, "
            f"devices={cuda_devices[0].strip()},{cuda_devices[1].strip()})"
        )

        self.sllm_process = subprocess.Popen(
            ["sllm", "start"],
            env=sllm_env,
            stdout=self.sllm_log_file,
            stderr=self.sllm_log_file,
        )
        print(f"SLLM server starting (logs: {self.log_file})")

        if self.deploy:
            if self.config is None:
                raise ValueError("--config is required when --deploy is set")
            config_path = self.config.resolve()
            if not config_path.is_file():
                raise FileNotFoundError(
                    f"Deploy config not found: {config_path}"
                )
            deploy_thread = threading.Thread(
                target=self.deploy_shadow_model,
                args=(config_path, sllm_env),
                daemon=True,
            )
            deploy_thread.start()

        if self.head_process is not None:
            self.head_process.wait()
        for proc in self.worker_processes:
            proc.wait()
        if self.sllm_process is not None:
            self.sllm_process.wait()

    def shutdown(self) -> None:
        print("Shutting down Ray cluster and SLLM server")
        for proc in (
            self.sllm_process,
            *self.worker_processes,
            self.head_process,
        ):
            if proc is not None and proc.poll() is None:
                proc.send_signal(signal.SIGTERM)

    def close(self) -> None:
        if self.sllm_log_file is not None:
            self.sllm_log_file.close()
            self.sllm_log_file = None

    def wait_for_ray_ready(
        self,
        proc: subprocess.Popen,
        address: str,
        env: dict[str, str],
        *,
        worker_id: int | None = None,
        timeout_sec: float = 120.0,
        interval_sec: float = 1.0,
    ) -> None:
        """Block until Ray GCS answers `ray status`, or worker_id_N registers."""
        label = f"Ray worker {worker_id}" if worker_id is not None else "Ray"
        resource_re = None
        resource = None
        if worker_id is not None:
            resource = f"worker_id_{worker_id}"
            resource_re = re.compile(
                rf"\d+(?:\.\d+)?/1\.0 {re.escape(resource)}\b"
            )
        deadline = time.monotonic() + timeout_sec
        check = ["ray", "status", f"--address={address}"]
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                raise RuntimeError(
                    f"{label} process exited early with code {proc.returncode}"
                )
            r = subprocess.run(
                check,
                env=env,
                capture_output=True,
                text=True,
                timeout=10,
            )
            ready = r.returncode == 0 and (
                resource_re is None or resource_re.search(r.stdout)
            )
            if ready:
                return
            time.sleep(interval_sec)
        if worker_id is not None:
            raise RuntimeError(
                f"{label} ({resource}) did not register within {timeout_sec:.0f}s"
            )
        raise RuntimeError(
            f"Ray at {address} did not become ready within {timeout_sec:.0f}s. "
            "Try `ray stop -f` and ensure port 6379 is free, then retry."
        )

    def wait_for_sllm_ready(
        self,
        port: int = SLLM_PORT,
        timeout_sec: float = 120.0,
        interval_sec: float = 1.0,
    ) -> None:
        """Block until the SLLM HTTP server responds on /health."""
        url = f"http://127.0.0.1:{port}/health"
        deadline = time.monotonic() + timeout_sec
        while time.monotonic() < deadline:
            try:
                with urllib.request.urlopen(url, timeout=5) as resp:
                    if resp.status == 200:
                        return
            except (urllib.error.URLError, TimeoutError):
                pass
            time.sleep(interval_sec)
        raise RuntimeError(
            f"SLLM server at {url} did not become ready within {timeout_sec:.0f}s"
        )

    def start_ray_worker(
        self,
        worker_id: int,
        cuda_device: str,
        num_cpus: int,
        env: dict[str, str],
    ) -> subprocess.Popen:
        """Start one Ray worker with a single GPU and worker_id_N resource."""
        worker_env = env.copy()
        worker_env["CUDA_VISIBLE_DEVICES"] = cuda_device
        worker_resources = {
            "worker_node": 1,
            f"worker_id_{worker_id}": 1,
        }
        worker_cmd = [
            "ray",
            "start",
            f"--address={RAY_HEAD_ADDRESS}",
            f"--num-cpus={num_cpus}",
            "--num-gpus=1",
            "--resources",
            json.dumps(worker_resources),
            "--include-log-monitor=false",
            "--block",
        ]
        proc = subprocess.Popen(
            worker_cmd,
            env=worker_env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        self.wait_for_ray_ready(
            proc, RAY_HEAD_ADDRESS, worker_env, worker_id=worker_id
        )
        return proc

    def deploy_shadow_model(
        self, config_path: Path, env: dict[str, str]
    ) -> None:
        deploy_env = env.copy()
        deploy_env.setdefault("LLM_SERVER_URL", f"http://127.0.0.1:{SLLM_PORT}")
        self.wait_for_sllm_ready()
        print(f"Deploying model from {config_path}")
        subprocess.run(
            ["sllm", "deploy", "--config", str(config_path)],
            env=deploy_env,
            check=True,
        )
        print("Model deployed (shadow_scale_up in router_config)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Start Ray + SLLM for local development.",
    )
    parser.add_argument(
        "--deploy",
        action="store_true",
        help="After SLLM is up, run `sllm deploy` with --config.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Model deploy config (required with --deploy).",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=DEFAULT_SLLM_LOG_FILE,
        help=f"SLLM server log file (default: {DEFAULT_SLLM_LOG_FILE})",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    context = Context(
        deploy=args.deploy,
        config=args.config,
        log_file=args.log_file,
    )

    def handle_signal(signum, frame) -> None:
        print(f"Received signal {signum}; shutting down", file=sys.stderr)
        context.shutdown()

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    try:
        context.run()
        return 0
    except Exception as exc:
        print(f"start_sllm failed: {exc}", file=sys.stderr)
        return 1
    finally:
        context.close()


if __name__ == "__main__":
    sys.exit(main())
