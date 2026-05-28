#!/usr/bin/env python3
"""Start a local Ray cluster + SLLM server for development."""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import psutil

logger = logging.getLogger("start_sllm.py")

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SLLM_LOG_FILE = ROOT / "logs" / "sllm.log"

RAY_PORT = 6379
RAY_HEAD_ADDRESS = f"127.0.0.1:{RAY_PORT}"
SLLM_PORT = 8343
SLLM_URL = f"http://127.0.0.1:{SLLM_PORT}"


def kill_proc_tree(pid: int, sig: int = signal.SIGTERM) -> None:
    try:
        parent = psutil.Process(pid)
        # Get all children and grandchildren recursively
        children = parent.children(recursive=True)

        # Kill all child processes first
        for child in children:
            child.send_signal(sig)

        # Finally, kill the parent process
        parent.send_signal(sig)
    except psutil.NoSuchProcess:
        pass


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

        self.cuda_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "0,1").split(
            ","
        )
        if len(self.cuda_devices) < 2:
            raise ValueError(
                "Two Ray workers require two CUDA devices. "
                "Set CUDA_VISIBLE_DEVICES=0,1 (or similar)."
            )
        self.worker_num_cpus = 64
        self.sllm_log_file = None
        self.head_process: subprocess.Popen | None = None
        self.worker_processes: list[subprocess.Popen] = []
        self.sllm_process: subprocess.Popen | None = None

    def run(self) -> None:
        os.chdir(ROOT)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        self.sllm_log_file = open(self.log_file, "w")

        t_start_ray = time.perf_counter()
        self.head_process = self._start_ray_head()
        for worker_id, cuda_device in enumerate(self.cuda_devices[:2]):
            proc = self._start_ray_worker(
                worker_id,
                cuda_device.strip(),
                self.worker_num_cpus,
            )
            self.worker_processes.append(proc)
        elapsed_start_ray = time.perf_counter() - t_start_ray
        logger.info("Ray cluster started in %.3f seconds", elapsed_start_ray)

        t_start_sllm = time.perf_counter()
        self.sllm_process = self._start_sllm()
        elapsed_start_sllm = time.perf_counter() - t_start_sllm
        logger.info("SLLM server started in %.3f seconds", elapsed_start_sllm)

        if self.deploy:
            t_deploy_model = time.perf_counter()
            self._deploy_model()
            elapsed_deploy_model = time.perf_counter() - t_deploy_model
            logger.info(
                "Model deployed in %.3f seconds",
                elapsed_deploy_model,
            )

        if self.sllm_process is not None:
            self.sllm_process.wait()
        for proc in self.worker_processes:
            proc.wait()
        if self.head_process is not None:
            self.head_process.wait()

    def shutdown(self) -> None:
        logger.info("Shutting down Ray cluster and SLLM server")
        self._shutdown_requested = True
        if self.sllm_process is not None:
            kill_proc_tree(self.sllm_process.pid)
        for proc in self.worker_processes:
            kill_proc_tree(proc.pid)
        if self.head_process is not None:
            kill_proc_tree(self.head_process.pid)

    def _start_ray_head(self) -> subprocess.Popen:
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
        proc = subprocess.Popen(
            head_cmd,
            env=head_env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        self._wait_for_ray_ready(proc, RAY_HEAD_ADDRESS, head_env)
        return proc

    def _start_ray_worker(
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
        self._wait_for_ray_ready(
            proc, RAY_HEAD_ADDRESS, worker_env, worker_id=worker_id
        )
        return proc

    def _wait_for_ray_ready(
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
            if self._shutdown_requested:
                raise RuntimeError("Shutdown requested while waiting for Ray")
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

    def _stop_ray(self) -> None:
        subprocess.run(
            ["ray", "stop", "-f", f"--address={RAY_HEAD_ADDRESS}"],
            capture_output=True,
            timeout=60,
            check=False,
        )
        subprocess.run(
            ["ray", "stop", "-f"],
            capture_output=True,
            timeout=60,
            check=False,
        )
        for proc in self.worker_processes:
            proc.terminate()
        if self.head_process is not None:
            self.head_process.terminate()

    def _start_sllm(self) -> subprocess.Popen:
        sllm_env = os.environ.copy()
        sllm_env["RAY_HEAD_ADDRESS"] = RAY_HEAD_ADDRESS
        proc = subprocess.Popen(
            ["sllm", "start"],
            env=sllm_env,
            stdout=self.sllm_log_file,
            stderr=self.sllm_log_file,
        )
        self._wait_for_sllm_ready(proc)
        return proc

    def _wait_for_sllm_ready(
        self,
        # port: int = SLLM_PORT,
        timeout_sec: float = 120.0,
        interval_sec: float = 1.0,
    ) -> None:
        """Block until the SLLM HTTP server responds on /health."""
        url = f"{SLLM_URL}/health"
        deadline = time.monotonic() + timeout_sec
        while time.monotonic() < deadline:
            if self._shutdown_requested:
                raise RuntimeError("Shutdown requested while waiting for SLLM")
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

    def _deploy_model(self) -> None:
        if self.config is None:
            raise ValueError("--config is required when --deploy is set")
        config_path = self.config.resolve()
        if not config_path.is_file():
            raise FileNotFoundError(f"Deploy config not found: {config_path}")
        deploy_env = os.environ.copy()
        deploy_env.setdefault("LLM_SERVER_URL", f"http://127.0.0.1:{SLLM_PORT}")
        logger.info("Deploying model from %s", config_path)
        subprocess.run(
            ["sllm", "deploy", "--config", str(config_path)],
            env=deploy_env,
            check=True,
        )


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


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()

    context = Context(
        deploy=args.deploy,
        config=args.config,
        log_file=args.log_file,
    )

    def handle_signal(signum, frame) -> None:
        logger.warning("Received signal %s; shutting down", signum)
        context.shutdown()

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    try:
        context.run()
    except Exception as exc:
        logger.exception("start_sllm failed")
        sys.exit(1)
    finally:
        context.shutdown()


if __name__ == "__main__":
    main()
