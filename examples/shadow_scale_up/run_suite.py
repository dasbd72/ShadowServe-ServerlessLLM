# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run shadow scale-up benchmark suite (baseline + shadow per model)."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import io
import logging
import os
import re
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import psutil

logger = logging.getLogger("run_suite.py")

EXAMPLE_DIR = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[2]
LOGS_DIR = ROOT / "logs"
START_SLLM_SCRIPT = EXAMPLE_DIR / "start_sllm.py"

SLLM_PORT = 8343
SLLM_URL = f"http://127.0.0.1:{SLLM_PORT}"

DEFAULT_CONCURRENT = 4
DEFAULT_MAX_TOKENS_QWEN3_0_6B = 16384
DEFAULT_MAX_TOKENS_QWEN3_4B = 8192
DEFAULT_MAX_TOKENS_QWEN3_8B = 4096
DEFAULT_STAGGER_MS = 1000
DEFAULT_NUM_TRIGGER_RUNS = 4

TRIGGER_TIMEOUT_SEC = 600.0
HEALTH_TIMEOUT_SEC = 300.0
DEPLOY_TIMEOUT_SEC = 900.0

MODEL_SLUGS: dict[str, str] = {
    "Qwen/Qwen3-0.6B": "qwen3-0.6b",
    "Qwen/Qwen3-4B": "qwen3-4b",
    "Qwen/Qwen3-8B": "qwen3-8b",
}

CONFIG_BASENAME: dict[str, str] = {
    "Qwen/Qwen3-0.6B": "Qwen3-0.6B",
    "Qwen/Qwen3-4B": "Qwen3-4B",
    "Qwen/Qwen3-8B": "Qwen3-8B",
}

DEFAULT_MAX_TOKENS: dict[str, int] = {
    "Qwen/Qwen3-0.6B": DEFAULT_MAX_TOKENS_QWEN3_0_6B,
    "Qwen/Qwen3-4B": DEFAULT_MAX_TOKENS_QWEN3_4B,
    "Qwen/Qwen3-8B": DEFAULT_MAX_TOKENS_QWEN3_8B,
}

# RoundRobinRouter logs this after init_backend completes (see roundrobin_router.py).
_DEPLOY_READY_RE = re.compile(
    r"Initialized backend for instance .* for model {model}"
)
# Shadow handler logs this after shadow CPU backend init (roundrobin_router.py).
_SHADOW_DEPLOY_READY_RE = re.compile(
    r"Created shadow instance .+ for model {model} .*\"init_shadow_backend\""
)


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


@dataclass(frozen=True)
class Scenario:
    name: str
    config: Path
    model: str
    max_tokens: int
    sllm_log: Path
    client_log_base: Path


class _Tee(io.TextIOBase):
    """Write to multiple text streams (stdout + log file)."""

    def __init__(self, *streams: io.TextIOBase) -> None:
        self._streams = streams

    def write(self, data: str) -> int:
        for stream in self._streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self) -> None:
        for stream in self._streams:
            stream.flush()


class Context:
    def __init__(
        self,
        models: list[str],
        *,
        concurrent: int,
        stagger_ms: int,
        num_trigger_runs: int,
        max_tokens_by_model: dict[str, int],
    ) -> None:
        self.models = models
        self.concurrent = concurrent
        self.stagger_ms = stagger_ms
        self.num_trigger_runs = num_trigger_runs
        self.max_tokens_by_model = max_tokens_by_model

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
            basename = CONFIG_BASENAME[model]
            max_tokens = self.max_tokens_by_model[model]
            scenarios.append(
                Scenario(
                    name=f"baseline-{slug}",
                    config=EXAMPLE_DIR / f"config-vllm-{basename}.json",
                    model=model,
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
                    config=EXAMPLE_DIR / f"config-vllm-shadow-{basename}.json",
                    model=model,
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
        logger.info(
            "  config=%s model=%s max_tokens=%s",
            scenario.config,
            scenario.model,
            scenario.max_tokens,
        )
        logger.info("  sllm log: %s", scenario.sllm_log)
        logger.info(
            "  client logs: %s_run{1..%d}.log",
            scenario.client_log_base,
            self.num_trigger_runs,
        )

        scenario.sllm_log.parent.mkdir(parents=True, exist_ok=True)

        self._start_sllm(scenario.config, scenario.sllm_log)
        try:
            self._wait_sllm_health()
            self._wait_model_deployed(
                scenario.sllm_log,
                scenario.model,
                require_shadow="shadow" in scenario.name,
            )

            for run_idx in range(1, self.num_trigger_runs + 1):
                if self._shutdown_requested:
                    break
                log_file = Path(f"{scenario.client_log_base}_run{run_idx}.log")
                label = "cold start" if run_idx == 1 else f"run {run_idx}"
                self._run_trigger(
                    scenario.model,
                    scenario.max_tokens,
                    log_file,
                    label=label,
                )
        finally:
            if self._start_sllm_proc is not None:
                self._stop_sllm()

        finished = datetime.now().astimezone().isoformat(timespec="seconds")
        logger.info("==> %s finished scenario %s", finished, scenario.name)

    def _start_sllm(self, config: Path, log_file: Path) -> None:
        start_cmd = [
            sys.executable,
            str(START_SLLM_SCRIPT),
            "--deploy",
            "--config",
            str(config),
            "--log-file",
            str(log_file),
        ]
        self._start_sllm_proc = subprocess.Popen(start_cmd)

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
        require_shadow: bool = False,
    ) -> None:
        ready_re = re.compile(
            _DEPLOY_READY_RE.pattern.format(model=re.escape(model))
        )
        shadow_ready_re = None
        if require_shadow:
            shadow_ready_re = re.compile(
                _SHADOW_DEPLOY_READY_RE.pattern.format(model=re.escape(model))
            )
        gpu_ready = False
        shadow_ready = not require_shadow
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
                if not gpu_ready and ready_re.search(chunk):
                    gpu_ready = True
                    logger.info(
                        "GPU instance ready for %s (see %s)", model, log_file
                    )
                if (
                    shadow_ready_re is not None
                    and not shadow_ready
                    and shadow_ready_re.search(chunk)
                ):
                    shadow_ready = True
                    logger.info(
                        "Shadow instance ready for %s (see %s)", model, log_file
                    )
                if gpu_ready and shadow_ready:
                    return
            time.sleep(2)
        missing = []
        if not gpu_ready:
            missing.append('GPU instance init (router timing log with "total")')
        if require_shadow and not shadow_ready:
            missing.append("shadow backend init")
        raise RuntimeError(
            f"Deploy for {model} did not finish within "
            f"{DEPLOY_TIMEOUT_SEC:.0f}s "
            f"(missing: {', '.join(missing)}; see {log_file})"
        )

    def _stop_sllm(self) -> None:
        if self._start_sllm_proc is not None:
            kill_proc_tree(self._start_sllm_proc.pid)
            self._start_sllm_proc.wait(timeout=60)
            self._start_sllm_proc = None

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
        if str(EXAMPLE_DIR) not in sys.path:
            sys.path.insert(0, str(EXAMPLE_DIR))
        from trigger_scale_up import Context as TriggerContext

        stamp = datetime.now().astimezone().isoformat(timespec="seconds")
        trigger = TriggerContext(
            url=self.sllm_url,
            model=model,
            concurrent=self.concurrent,
            max_tokens=max_tokens,
            timeout=TRIGGER_TIMEOUT_SEC,
            stagger_ms=self.stagger_ms,
        )

        with log_file.open("w", encoding="utf-8") as log_fh:
            log_fh.write(f"===== {stamp} =====\n")
            log_fh.flush()
            tee = _Tee(sys.stdout, log_fh)
            with contextlib.redirect_stdout(tee):
                asyncio.run(trigger.run())

        finished = datetime.now().astimezone().isoformat(timespec="seconds")
        logger.info("==> %s trigger finished -> %s", finished, log_file)

    def shutdown(self) -> None:
        logger.info("Shutting down SLLM server and Ray cluster")
        self._shutdown_requested = True
        self._stop_sllm()


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
        "--concurrent",
        type=int,
        default=DEFAULT_CONCURRENT,
        help="Parallel in-flight requests per trigger run",
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
        "--stagger-ms",
        type=int,
        default=DEFAULT_STAGGER_MS,
        help="Delay between starting each request (ms)",
    )
    p.add_argument(
        "--num-trigger-runs",
        type=int,
        default=DEFAULT_NUM_TRIGGER_RUNS,
        help="Trigger rounds per scenario (first is cold-start)",
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
        models=models,
        concurrent=args.concurrent,
        stagger_ms=args.stagger_ms,
        num_trigger_runs=args.num_trigger_runs,
        max_tokens_by_model=max_tokens_by_model,
    )

    def handle_signal(signum, frame) -> None:
        logger.error("Received signal %s; shutting down", signum)
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
