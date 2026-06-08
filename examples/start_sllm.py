#!/usr/bin/env python3
"""Start a local Ray cluster + SLLM server for development."""

from __future__ import annotations

import argparse
import io
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

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SLLM_LOG_FILE = ROOT / "logs" / "sllm.log"

RAY_PORT = 6379
RAY_HEAD_ADDRESS = f"127.0.0.1:{RAY_PORT}"
SLLM_PORT = 8343
SLLM_URL = f"http://127.0.0.1:{SLLM_PORT}"

ORPHAN_PROCESS_MARKERS = ("VLLM::EngineCore",)


def get_proc_tree(pid: int) -> list[int]:
    try:
        parent = psutil.Process(pid)
        return [parent.pid] + [
            child.pid for child in parent.children(recursive=True)
        ]
    except psutil.NoSuchProcess:
        return []


def kill_procs(
    pids: list[int],
    sig: int = signal.SIGTERM,
    *,
    wait_timeout: float = 15.0,
) -> None:
    procs: list[psutil.Process] = []
    for pid in pids:
        try:
            proc = psutil.Process(pid)
            procs.append(proc)
        except psutil.NoSuchProcess:
            continue
    for proc in procs:
        try:
            proc.send_signal(sig)
        except psutil.NoSuchProcess:
            pass
    if sig == signal.SIGKILL:
        return
    _, alive = psutil.wait_procs(procs, timeout=wait_timeout)
    for proc in alive:
        try:
            proc.kill()
        except psutil.NoSuchProcess:
            pass
    if alive:
        psutil.wait_procs(alive, timeout=5)


def distribute_cuda_devices(
    devices: list[str], num_workers: int
) -> list[list[str]]:
    """Split *devices* into *num_workers* groups of nearly equal size."""
    n = len(devices)
    base, extra = divmod(n, num_workers)
    chunks: list[list[str]] = []
    start = 0
    for worker_id in range(num_workers):
        size = base + (1 if worker_id < extra else 0)
        chunks.append(devices[start : start + size])
        start += size
    return chunks


def ray_stop_local(
    *,
    force: bool = True,
    grace_period: int = 60,
    attempts: int = 1,
) -> None:
    """Stop Ray daemons on this machine (not just the `ray start --block` launcher)."""
    cmd = ["ray", "stop", f"--grace-period={grace_period}"]
    if force:
        cmd.append("-f")
    for _ in range(max(1, attempts)):
        subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )


class Context:
    def __init__(
        self,
        num_workers: int,
        num_cpus_per_worker: int,
        cuda_devices: list[str],
        deploy: bool,
        config: Path | None,
        log_file: Path,
    ):
        self.num_workers = num_workers
        self.num_cpus_per_worker = num_cpus_per_worker
        self.cuda_devices = cuda_devices
        self.deploy = deploy
        self.config = config
        self.log_file = log_file
        if len(self.cuda_devices) < self.num_workers:
            raise ValueError(
                f"{self.num_workers} Ray workers need at least one CUDA device "
                f"each, but only {len(self.cuda_devices)} found. "
                "Pass --cuda-devices or set CUDA_VISIBLE_DEVICES accordingly."
            )
        self.worker_cuda_devices = distribute_cuda_devices(
            self.cuda_devices, self.num_workers
        )

        self.sllm_log_file: io.TextIOBase | None = None
        self.head_process: subprocess.Popen | None = None
        self.worker_processes: list[subprocess.Popen] = []
        self.sllm_process: subprocess.Popen | None = None
        self._shutdown_requested = False

    def run(self) -> None:
        os.chdir(ROOT)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        self.sllm_log_file = open(self.log_file, "w")

        t_start_ray = time.perf_counter()
        head_process = self.head_process = self._start_ray_head()
        worker_processes = []
        for worker_id, cuda_devices in enumerate(self.worker_cuda_devices):
            proc = self._start_ray_worker(
                worker_id,
                cuda_devices,
                self.num_cpus_per_worker,
            )
            self.worker_processes.append(proc)
            worker_processes.append(proc)
        elapsed_start_ray = time.perf_counter() - t_start_ray
        logger.info("Ray cluster started in %.3f seconds", elapsed_start_ray)

        t_start_sllm = time.perf_counter()
        sllm_process = self.sllm_process = self._start_sllm()
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

        if sllm_process is not None:
            sllm_process.wait()
        for proc in worker_processes:
            proc.wait()
        if head_process is not None:
            head_process.wait()

    def shutdown(self) -> None:
        if self._shutdown_requested:
            return
        logger.info("Shutting down Ray cluster and SLLM server")
        self._shutdown_requested = True

        sllm_process = self.sllm_process
        worker_processes = self.worker_processes
        head_process = self.head_process
        sllm_log_file = self.sllm_log_file
        self.sllm_process = None
        self.worker_processes = []
        self.head_process = None
        self.sllm_log_file = None

        self._stop_ray()

        procs_ids: list[int] = []
        if sllm_process is not None:
            procs_ids.extend(get_proc_tree(sllm_process.pid))
        for proc in worker_processes:
            procs_ids.extend(get_proc_tree(proc.pid))
        if head_process is not None:
            procs_ids.extend(get_proc_tree(head_process.pid))

        kill_procs(procs_ids, signal.SIGTERM, wait_timeout=15)

        if sllm_log_file is not None:
            sllm_log_file.close()

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
        self._wait_for_ray_ready(proc, RAY_HEAD_ADDRESS)
        return proc

    def _start_ray_worker(
        self,
        worker_id: int,
        cuda_devices: list[str],
        num_cpus: int,
    ) -> subprocess.Popen:
        """Start one Ray worker with its GPU slice and worker_id_N resource."""
        worker_env = os.environ.copy()
        worker_env["CUDA_VISIBLE_DEVICES"] = ",".join(cuda_devices)
        num_gpus = len(cuda_devices)
        worker_resources = {
            "worker_node": 1,
            f"worker_id_{worker_id}": 1,
        }
        worker_cmd = [
            "ray",
            "start",
            f"--address={RAY_HEAD_ADDRESS}",
            f"--num-cpus={num_cpus}",
            f"--num-gpus={num_gpus}",
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
        self._wait_for_ray_ready(proc, RAY_HEAD_ADDRESS, worker_id=worker_id)
        return proc

    def _stop_ray(self) -> None:
        cmd = ["ray", "stop"]
        subprocess.run(cmd, check=True)
        self._kill_orphaned_vllm_processes()

    def _kill_orphaned_vllm_processes(
        self, *, wait_timeout: float = 15.0
    ) -> None:
        targets: list[int] = []
        for proc in psutil.process_iter(["pid", "cmdline"]):
            try:
                cmdline = proc.info.get("cmdline") or []
                haystack = " ".join(cmdline)
                if any(marker in haystack for marker in ORPHAN_PROCESS_MARKERS):
                    targets.append(proc.pid)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        if not targets:
            return

        logger.info(
            "Cleaning up %d orphaned vLLM/backend process(es)", len(targets)
        )
        kill_procs(targets, signal.SIGKILL, wait_timeout=wait_timeout)

    def _wait_for_ray_ready(
        self,
        proc: subprocess.Popen,
        address: str,
        *,
        worker_id: int | None = None,
        timeout_sec: float = 60.0,
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

    def _start_sllm(self) -> subprocess.Popen:
        sllm_env = os.environ.copy()
        sllm_env.setdefault("RAY_HEAD_ADDRESS", RAY_HEAD_ADDRESS)
        sllm_env.setdefault("PYTHONUNBUFFERED", "1")
        proc = subprocess.Popen(
            ["sllm", "start"],
            env=sllm_env,
            stdout=self.sllm_log_file,
            stderr=self.sllm_log_file,
        )
        self._wait_for_sllm_ready()
        return proc

    def _wait_for_sllm_ready(
        self,
        *,
        timeout_sec: float = 60.0,
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


def parse_cuda_devices(value: str) -> list[str]:
    devices = [device.strip() for device in value.split(",") if device.strip()]
    if not devices:
        raise argparse.ArgumentTypeError(
            "--cuda-devices must list at least one device ID"
        )
    return devices


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Start Ray + SLLM for local development.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
        help="Number of Ray GPU workers to start (default: 2).",
    )
    parser.add_argument(
        "--num-cpus-per-worker",
        type=int,
        default=64,
        help="CPUs advertised per Ray worker (default: 64).",
    )
    parser.add_argument(
        "--cuda-devices",
        type=parse_cuda_devices,
        default=None,
        help=(
            "Comma-separated GPU device IDs for Ray workers "
            "(default: CUDA_VISIBLE_DEVICES or 0,1)."
        ),
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
        num_workers=args.num_workers,
        num_cpus_per_worker=args.num_cpus_per_worker,
        cuda_devices=args.cuda_devices,
        deploy=args.deploy,
        config=args.config,
        log_file=args.log_file,
    )

    def handle_signal(signum, frame) -> None:
        logger.warning(
            "Received signal %s; shutting down start_sllm.py", signum
        )
        context.shutdown()

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    try:
        context.run()
    except Exception:
        logger.exception("start_sllm failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
