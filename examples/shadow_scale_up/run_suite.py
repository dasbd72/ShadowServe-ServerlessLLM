#!/usr/bin/env python3
"""Run shadow scale-up benchmark suite (baseline + shadow, Qwen3-0.6B and 8B).

For each scenario: start Ray + SLLM via ``start_sllm.py --deploy``, run four
trigger rounds (first run separated for cold-start on the Ray worker), then
tear down.

Prerequisites:
  1. ``sllm-store`` listening (default gRPC port 8073).
  2. ``CUDA_VISIBLE_DEVICES`` with at least two GPUs (see ``start_sllm.py``).

Example (from repo root):
  python examples/shadow_scale_up/run_suite.py
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import io
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
from datetime import datetime, timezone
from pathlib import Path

_EXAMPLE_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _EXAMPLE_DIR.parent.parent

DEFAULT_SLLM_STORE_PORT = 8073
DEFAULT_RAY_HEAD_PORT = 6379
DEFAULT_SLLM_HTTP_PORT = 8343
DEFAULT_SLLM_URL = os.environ.get("LLM_SERVER_URL", "http://127.0.0.1:8343")

DEFAULT_CONCURRENT = 4
DEFAULT_MAX_TOKENS_QWEN3_0_6B = 16384
DEFAULT_MAX_TOKENS_QWEN3_8B = 4096
DEFAULT_STAGGER_MS = 1000
DEFAULT_NUM_TRIGGER_RUNS = 4

TRIGGER_TIMEOUT_SEC = 600.0
HEALTH_TIMEOUT_SEC = 300.0
DEPLOY_TIMEOUT_SEC = 900.0

# RoundRobinRouter logs this after init_backend completes (see roundrobin_router.py).
_DEPLOY_READY_RE = re.compile(
    r"Started instance .+ for model {model} \{{.*\"total\":"
)
# Shadow handler logs this after shadow CPU backend init (roundrobin_router.py).
_SHADOW_DEPLOY_READY_RE = re.compile(
    r"Created shadow instance .+ for model {model} .*\"init_shadow_backend\""
)


@dataclass(frozen=True)
class Scenario:
    name: str
    config: Path
    model: str
    max_tokens: int
    sllm_log: Path
    client_log_base: Path


SCENARIOS: tuple[Scenario, ...] = (
    Scenario(
        name="baseline-qwen3-0.6b",
        config=_EXAMPLE_DIR / "config-vllm-Qwen3-0.6B.json",
        model="Qwen/Qwen3-0.6B",
        max_tokens=DEFAULT_MAX_TOKENS_QWEN3_0_6B,
        sllm_log=_REPO_ROOT
        / "logs/trigger_scale_up_baseline_sllm_qwen3-0.6b.log",
        client_log_base=_REPO_ROOT
        / "logs/trigger_scale_up_baseline_client_qwen3-0.6b",
    ),
    Scenario(
        name="shadow-qwen3-0.6b",
        config=_EXAMPLE_DIR / "config-vllm-shadow-Qwen3-0.6B.json",
        model="Qwen/Qwen3-0.6B",
        max_tokens=DEFAULT_MAX_TOKENS_QWEN3_0_6B,
        sllm_log=_REPO_ROOT
        / "logs/trigger_scale_up_shadow_sllm_qwen3-0.6b.log",
        client_log_base=_REPO_ROOT
        / "logs/trigger_scale_up_shadow_client_qwen3-0.6b",
    ),
    Scenario(
        name="baseline-qwen3-8b",
        config=_EXAMPLE_DIR / "config-vllm-Qwen3-8B.json",
        model="Qwen/Qwen3-8B",
        max_tokens=DEFAULT_MAX_TOKENS_QWEN3_8B,
        sllm_log=_REPO_ROOT
        / "logs/trigger_scale_up_baseline_sllm_qwen3-8b.log",
        client_log_base=_REPO_ROOT
        / "logs/trigger_scale_up_baseline_client_qwen3-8b",
    ),
    Scenario(
        name="shadow-qwen3-8b",
        config=_EXAMPLE_DIR / "config-vllm-shadow-Qwen3-8B.json",
        model="Qwen/Qwen3-8B",
        max_tokens=DEFAULT_MAX_TOKENS_QWEN3_8B,
        sllm_log=_REPO_ROOT / "logs/trigger_scale_up_shadow_sllm_qwen3-8b.log",
        client_log_base=_REPO_ROOT
        / "logs/trigger_scale_up_shadow_client_qwen3-8b",
    ),
)


def _scenarios_with_max_tokens(
    max_tokens_qwen3_0_6b: int,
    max_tokens_qwen3_8b: int,
) -> tuple[Scenario, ...]:
    out: list[Scenario] = []
    for scenario in SCENARIOS:
        if "0.6B" in scenario.model:
            max_tokens = max_tokens_qwen3_0_6b
        elif "8B" in scenario.model:
            max_tokens = max_tokens_qwen3_8b
        else:
            max_tokens = scenario.max_tokens
        out.append(
            Scenario(
                name=scenario.name,
                config=scenario.config,
                model=scenario.model,
                max_tokens=max_tokens,
                sllm_log=scenario.sllm_log,
                client_log_base=scenario.client_log_base,
            )
        )
    return tuple(out)


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
        *,
        sllm_store_port: int,
        sllm_url: str,
        concurrent: int,
        stagger_ms: int,
        num_trigger_runs: int,
        scenarios: tuple[Scenario, ...],
    ):
        self.sllm_store_port = sllm_store_port
        self.sllm_url = sllm_url.rstrip("/")
        self.concurrent = concurrent
        self.stagger_ms = stagger_ms
        self.num_trigger_runs = num_trigger_runs
        self.scenarios = scenarios
        self._start_sllm_proc: subprocess.Popen | None = None
        self._shutdown_requested = False

    def run(self) -> None:
        os.chdir(_REPO_ROOT)
        (_REPO_ROOT / "logs").mkdir(parents=True, exist_ok=True)

        self.check_sllm_store()

        for scenario in self.scenarios:
            if self._shutdown_requested:
                break
            self.run_scenario(scenario)

        print("\nSuite complete. Logs under logs/trigger_scale_up_*")

    def check_sllm_store(self) -> None:
        host, port = "127.0.0.1", self.sllm_store_port
        if self.port_listening(host, port):
            print(f"sllm-store is listening on {host}:{port}")
            return
        raise RuntimeError(
            f"sllm-store is not listening on {host}:{port}. "
            'Start it first, e.g.: sllm-store start --storage-path "$STORAGE_PATH"'
        )

    def run_scenario(self, scenario: Scenario) -> None:
        max_tokens = scenario.max_tokens
        print("")
        print("=" * 64)
        print(f"Scenario: {scenario.name}")
        print(f"  config={scenario.config}")
        print(f"  model={scenario.model}")
        print(f"  max_tokens={max_tokens}")
        print(f"  sllm log: {scenario.sllm_log}")
        print(
            f"  client logs: {scenario.client_log_base}_run"
            f"{{1..{self.num_trigger_runs}}}.log"
        )
        print("=" * 64)

        scenario.sllm_log.parent.mkdir(parents=True, exist_ok=True)

        start_cmd = [
            sys.executable,
            str(_EXAMPLE_DIR / "start_sllm.py"),
            "--deploy",
            "--config",
            str(scenario.config),
            "--log-file",
            str(scenario.sllm_log),
        ]
        self._start_sllm_proc = subprocess.Popen(start_cmd)
        try:
            self.wait_sllm_health()
            self.wait_model_deployed(
                scenario.sllm_log,
                scenario.model,
                require_shadow="shadow" in scenario.name,
            )

            for run_idx in range(1, self.num_trigger_runs + 1):
                if self._shutdown_requested:
                    break
                log_file = Path(f"{scenario.client_log_base}_run{run_idx}.log")
                label = "cold start" if run_idx == 1 else f"run {run_idx}"
                self.run_trigger(
                    scenario.model,
                    max_tokens,
                    log_file,
                    label=label,
                )
        finally:
            self.stop_start_sllm()
            self._start_sllm_proc.wait(timeout=60)

    def run_trigger(
        self,
        model: str,
        max_tokens: int,
        log_file: Path,
        *,
        label: str,
    ) -> None:
        print(
            f"--- trigger ({label}): model={model} "
            f"max_tokens={max_tokens} log={log_file} ---"
        )
        log_file.parent.mkdir(parents=True, exist_ok=True)
        if str(_EXAMPLE_DIR) not in sys.path:
            sys.path.insert(0, str(_EXAMPLE_DIR))
        from trigger_scale_up import Context as TriggerContext

        stamp = datetime.now(timezone.utc).isoformat()
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

    def wait_sllm_health(self) -> None:
        url = f"{self.sllm_url}/health"
        deadline = time.monotonic() + HEALTH_TIMEOUT_SEC
        while time.monotonic() < deadline:
            if self._shutdown_requested:
                raise RuntimeError("Shutdown requested while waiting for SLLM")
            with contextlib.suppress(urllib.error.URLError, TimeoutError):
                with urllib.request.urlopen(url, timeout=5) as resp:
                    if resp.status == 200:
                        print(f"SLLM HTTP ready at {url}")
                        return
            time.sleep(2)
        raise RuntimeError(
            f"SLLM server at {url} did not become ready within "
            f"{HEALTH_TIMEOUT_SEC:.0f}s"
        )

    def wait_model_deployed(
        self,
        log_file: Path,
        model: str,
        *,
        require_shadow: bool = False,
    ) -> None:
        """Wait until inference (and optional shadow) backends are ready."""
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
                    print(f"GPU instance ready for {model} (see {log_file})")
                if (
                    shadow_ready_re is not None
                    and not shadow_ready
                    and shadow_ready_re.search(chunk)
                ):
                    shadow_ready = True
                    print(f"Shadow instance ready for {model} (see {log_file})")
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

    def wait_for_ports_free(
        self,
        ports: list[int],
        *,
        host: str = "127.0.0.1",
        timeout_sec: float = 60.0,
        interval_sec: float = 1.0,
    ) -> None:
        """Block until none of the given TCP ports accept connections."""
        deadline = time.monotonic() + timeout_sec
        pending = set(ports)
        while pending and time.monotonic() < deadline:
            for port in list(pending):
                if not self.port_listening(host, port):
                    pending.discard(port)
            if pending:
                time.sleep(interval_sec)
        if pending:
            raise RuntimeError(
                f"Ports still in use after {timeout_sec:.0f}s: "
                f"{sorted(pending)} on {host}"
            )

    @staticmethod
    def port_listening(host: str, port: int, timeout_sec: float = 1.0) -> bool:
        try:
            with socket.create_connection((host, port), timeout=timeout_sec):
                return True
        except OSError:
            return False

    def stop_start_sllm(self) -> None:
        self._start_sllm_proc.send_signal(signal.SIGTERM)

    def shutdown(self) -> None:
        self._shutdown_requested = True
        self.stop_start_sllm()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run shadow scale-up benchmark suite.",
    )
    parser.add_argument(
        "--sllm-store-port",
        type=int,
        default=int(os.environ.get("SLLM_STORE_PORT", DEFAULT_SLLM_STORE_PORT)),
        help=f"sllm-store gRPC port (default: {DEFAULT_SLLM_STORE_PORT})",
    )
    parser.add_argument(
        "--url",
        default=DEFAULT_SLLM_URL,
        help=f"SLLM server base URL (default: {DEFAULT_SLLM_URL})",
    )
    parser.add_argument(
        "--concurrent",
        type=int,
        default=DEFAULT_CONCURRENT,
        help="Parallel in-flight requests per trigger run",
    )
    parser.add_argument(
        "--max-tokens-qwen3-0-6b",
        type=int,
        default=int(
            os.environ.get(
                "TRIGGER_MAX_TOKENS_QWEN3_0_6B", DEFAULT_MAX_TOKENS_QWEN3_0_6B
            )
        ),
        help=f"max_tokens for Qwen3-0.6B (default: {DEFAULT_MAX_TOKENS_QWEN3_0_6B})",
    )
    parser.add_argument(
        "--max-tokens-qwen3-8b",
        type=int,
        default=int(
            os.environ.get(
                "TRIGGER_MAX_TOKENS_QWEN3_8B", DEFAULT_MAX_TOKENS_QWEN3_8B
            )
        ),
        help=f"max_tokens for Qwen3-8B (default: {DEFAULT_MAX_TOKENS_QWEN3_8B})",
    )
    parser.add_argument(
        "--stagger-ms",
        type=int,
        default=DEFAULT_STAGGER_MS,
        help="Delay between starting each request (ms)",
    )
    parser.add_argument(
        "--num-trigger-runs",
        type=int,
        default=DEFAULT_NUM_TRIGGER_RUNS,
        help="Trigger rounds per scenario (first is cold-start)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    context = Context(
        sllm_store_port=args.sllm_store_port,
        sllm_url=args.url,
        concurrent=args.concurrent,
        stagger_ms=args.stagger_ms,
        num_trigger_runs=args.num_trigger_runs,
        scenarios=_scenarios_with_max_tokens(
            args.max_tokens_qwen3_0_6b,
            args.max_tokens_qwen3_8b,
        ),
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
        print(f"run_suite failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
