#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import signal
import sys
import time
from datetime import datetime
from typing import Literal

import httpx
import matplotlib.pyplot as plt
import numpy as np

DEFAULT_URL = os.environ.get("LLM_SERVER_URL", "http://127.0.0.1:8343")
DEFAULT_MODEL = "Qwen/Qwen3-8B"


logger = logging.getLogger("workload_client.py")


# ===== dataset =====

Role = Literal["user", "assistant", "system"]

FROM_TO_ROLE: dict[str, Role] = {
    "human": "user",
    "user": "user",
    "gpt": "assistant",
    "bing": "assistant",
    "chatgpt": "assistant",
    "bard": "assistant",
    "system": "system",
}


class Message:
    def __init__(self, role: Role, content: str):
        self.role = role
        self.content = content

    def to_dict(self) -> dict:
        return {
            "role": self.role,
            "content": self.content,
        }


class ShareGPTDataset:
    def __init__(self, data: list[dict]):
        ids: list[str] = []
        conversations: dict[str, list[Message]] = {}
        for item in data:
            ids.append(item["id"])
            conversations[item["id"]] = [
                Message(FROM_TO_ROLE[item["from"]], item["value"])
                for item in item["conversations"]
            ]
        self.ids = ids
        self.conversations = conversations

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index: int) -> list[Message]:
        return self.conversations[self.ids[index]]


def load_sharegpt_dataset(path: str) -> ShareGPTDataset:
    with open(path, "r") as f:
        data = json.load(f)
    return ShareGPTDataset(data)


def extract_messages(
    messages: list[Message], end_role: Role | None = None
) -> list[Message]:
    """Extract messages from a list of messages until the end role is reached.

    Args:
        messages: list of messages
        end_role: the role to stop at

    Returns:
        list of messages
    """
    if end_role == None:
        return messages

    for i, message in reversed(list(enumerate(messages))):
        if message.role == end_role:
            return messages[: i + 1]

    return []


def messages_to_dict(messages: list[Message]) -> dict:
    return [message.to_dict() for message in messages]


# ===== workload =====


def create_workload_gamma(
    mean_interval: float, cv: float, num_requests: int
) -> list[float]:
    shape = 1.0 / cv**2
    scale = mean_interval * (cv**2)
    intervals = np.random.gamma(shape, scale, num_requests)
    start_times = np.cumsum(intervals)
    return start_times.tolist()


def create_workload_mmpp(
    num_requests: int,
    lambda_idle: float = 0.0083,
    lambda_burst: float = 0.4,
    stay_idle_time: float = 60.0,
    stay_burst_time: float = 1.0,
) -> list[float]:
    """
    Generates request start times using a 2-State Markov-Modulated Poisson Process (MMPP).
    Simulates periods of idle silence punctuated by dense serverless flash crowds.

    Args:
        num_requests: number of requests to generate
        lambda_idle: arrival rate in idle state (requests per second)
        lambda_burst: arrival rate in burst state (requests per second)
        stay_idle_time: mean duration in idle state (seconds)
        stay_burst_time: mean duration in burst state (seconds)

    Returns:
        list of request start times
    """
    intervals = []

    # 0 = Idle state, 1 = Burst state
    current_state = 0

    # Track the remaining time we must spend in the current state
    # State durations are modeled as exponentially distributed continuous times
    time_left_in_state = np.random.exponential(stay_idle_time)

    while len(intervals) < num_requests:
        # Determine current arrival rate based on active state
        current_lambda = lambda_idle if current_state == 0 else lambda_burst

        # Sample the time until the next potential request arrival
        # (Inverse of lambda is the scale parameter for np.random.exponential)
        time_to_next_request = np.random.exponential(1.0 / current_lambda)

        if time_to_next_request < time_left_in_state:
            # The request happens BEFORE the state changes
            intervals.append(time_to_next_request)
            time_left_in_state -= time_to_next_request
        else:
            # The state changes BEFORE a request can arrive
            # Consume the remaining time in this state
            time_left_in_state_at_transition = time_left_in_state

            # Switch states (0 -> 1 or 1 -> 0)
            current_state = 1 - current_state

            # Sample how long we will stay in the new state
            new_state_duration = (
                stay_idle_time if current_state == 0 else stay_burst_time
            )
            time_left_in_state = np.random.exponential(new_state_duration)

            # Calculate the residual time to a request in the new state
            new_lambda = lambda_idle if current_state == 0 else lambda_burst
            time_to_request_new_state = np.random.exponential(1.0 / new_lambda)

            # Total arrival interval is the time spent waiting in the old state
            # plus the time spent waiting in the new state
            total_interval = (
                time_left_in_state_at_transition + time_to_request_new_state
            )
            intervals.append(total_interval)

            # Deduct the elapsed time in the new state
            time_left_in_state = max(
                0.0, time_left_in_state - time_to_request_new_state
            )

    # Convert intervals into absolute sequence timelines
    start_times = np.cumsum(intervals)
    return start_times.tolist()


def plot_workload(
    start_times: list[float],
    duration: float = 100.0,
    filename: str = "workload.png",
):
    start_times = np.array(start_times)
    end_times = start_times + duration

    worker_ends: list[float] = []
    request_rows: list[int] = []
    for start, end in zip(start_times, end_times):
        placed = False
        for i, free_time in enumerate(worker_ends):
            if start >= free_time:
                worker_ends[i] = end
                request_rows.append(i)
                placed = True
                break
        if not placed:
            worker_ends.append(end)
            request_rows.append(len(worker_ends) - 1)

    num_workers = len(worker_ends)
    plt.figure(figsize=(12, max(4, num_workers * 0.4)))

    for start, row in zip(start_times, request_rows):
        plt.broken_barh(
            [(start, duration)], (row - 0.4, 0.8), facecolors="#4682B4"
        )

    plt.xlabel("Time (seconds)", fontsize=12)
    plt.ylabel("Worker / Channel ID", fontsize=12)
    plt.title(
        "Workload Request Execution (Gantt Chart)", fontsize=14, weight="bold"
    )
    plt.yticks(
        range(num_workers), [f"Worker {i + 1}" for i in range(num_workers)]
    )
    plt.grid(axis="x", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(filename)


class Context:
    def __init__(
        self,
        url: str,
        model: str,
        timeout: float,
        dataset_path: str,
        max_tokens: int,
        seed: int = 42,
    ):
        np.random.seed(seed)

        self.url = url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.max_tokens = max_tokens

        self.num_warmup_requests = 8
        self.mean_interval_s = 50.0
        self.cv = 4.0
        self.num_workload_requests = 40

        # workload
        self.start_times: list[float] = create_workload_mmpp(
            self.num_workload_requests
        )

        self.dataset = load_sharegpt_dataset(dataset_path)
        request_indices: list[int] = [
            i
            for i in range(len(self.dataset))
            if extract_messages(self.dataset[i], end_role="user")
        ]
        assert (
            len(request_indices)
            >= self.num_warmup_requests + self.num_workload_requests
        )
        np.random.shuffle(request_indices)
        self.warmup_request_indices: list[int] = request_indices[
            : self.num_warmup_requests
        ]
        self.workload_request_indices: list[int] = request_indices[
            self.num_warmup_requests :
        ]

    async def run(self) -> None:
        await self.wait_for_model_ready()

        futures: list[asyncio.Future[dict]] = []

        # Warmup requests
        logger.info(
            "Warming up with %d requests", len(self.warmup_request_indices)
        )
        for i, request_idx in enumerate(self.warmup_request_indices):
            futures.append(asyncio.create_task(self.send_chat(i, request_idx)))

        await asyncio.gather(*futures)

        # Start workload
        futures = []
        ttfts = []
        tpots = []
        e2es = []

        t0 = time.perf_counter()
        for i, start_time in enumerate(self.start_times):
            request_idx = self.workload_request_indices[i]
            futures.append(
                asyncio.create_task(
                    self.send_chat_delay(
                        i, request_idx, (start_time + t0) - time.perf_counter()
                    )
                )
            )

        for future in asyncio.as_completed(futures):
            result = await future
            ttft, tpot, e2e = (
                result.get("ttft_s"),
                result.get("tpot_s"),
                result.get("e2e_s"),
            )
            if not ttft or not tpot or not e2e:
                print(
                    f"[{result['index']}] HTTP {result['status_code']} in "
                    f"{result['elapsed_s']}s"
                )
            ttfts.append(ttft)
            tpots.append(tpot)
            e2es.append(e2e)
            print(
                f"[{result['index']}] sent_at={result.get('sent_at')} "
                f"HTTP {result['status_code']} in "
                f"{result['elapsed_s']}s finish={result.get('finish_reason')} "
                f"prompt_tokens={result.get('prompt_tokens')}, "
                f"completion_tokens={result.get('completion_tokens')}, "
                f"total_tokens={result.get('total_tokens')}, "
                f"ttft={result.get('ttft_s')}s tpot={result.get('tpot_s')}s "
                f"e2e={result.get('e2e_s')}s",
                flush=True,
            )
        print(f"Average ttft: {sum(ttfts) / len(ttfts)}s", flush=True)
        print(f"Average tpot: {sum(tpots) / len(tpots)}s", flush=True)
        print(f"Average e2e: {sum(e2es) / len(e2es)}s", flush=True)

    async def get_health(self) -> None:
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{self.url}/health", timeout=10)
            resp.raise_for_status()

    async def wait_for_model_ready(self) -> None:
        await self.get_health()
        deadline = time.perf_counter() + self.timeout
        models_url = f"{self.url}/v1/models"
        probe_url = f"{self.url}/v1/chat/completions"
        probe_payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": "ping"}],
            "max_tokens": 1,
            "temperature": 0.0,
        }
        async with httpx.AsyncClient() as client:
            while time.perf_counter() < deadline:
                try:
                    resp = await client.get(models_url, timeout=10)
                    if resp.status_code == 200:
                        models = resp.json().get("models", [])
                        model_ids = [
                            m.get("id") if isinstance(m, dict) else m
                            for m in models
                        ]
                        if self.model in model_ids:
                            break
                except httpx.HTTPError:
                    pass
                await asyncio.sleep(2)
            else:
                raise RuntimeError(
                    f"Model {self.model} not registered within {self.timeout}s"
                )

            while time.perf_counter() < deadline:
                try:
                    resp = await client.post(
                        probe_url,
                        json=probe_payload,
                        timeout=60,
                    )
                    if resp.status_code == 200:
                        logger.info("Model %s ready for inference", self.model)
                        return
                except httpx.HTTPError:
                    pass
                await asyncio.sleep(2)
        raise RuntimeError(
            f"Model {self.model} not accepting inference within {self.timeout}s"
        )

    async def send_chat_delay(
        self, idx: int, request_idx: int, delay: float
    ) -> dict:
        await asyncio.sleep(delay)
        return await self.send_chat(idx, request_idx)

    async def send_chat(self, idx: int, request_idx: int) -> dict:
        url = f"{self.url}/v1/chat/completions"
        request = self.dataset[request_idx]
        payload = {
            "model": self.model,
            "messages": messages_to_dict(
                extract_messages(request, end_role="user")
            ),
            "temperature": 0.0,
            "max_tokens": self.max_tokens,
        }
        headers = {"Content-Type": "application/json"}

        sent_at = datetime.now().isoformat(timespec="milliseconds")

        t0 = time.perf_counter()
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                url, headers=headers, json=payload, timeout=self.timeout
            )
        elapsed = time.perf_counter() - t0
        out = {
            "index": idx,
            "request_idx": request_idx,
            "sent_at": sent_at,
            "status_code": resp.status_code,
            "elapsed_s": round(elapsed, 2),
        }
        try:
            resp.raise_for_status()
            body = resp.json()
            choices = body.get("choices") or []
            choice = choices[0] if choices else {}
            msg = choice.get("message") or {}
            out["finish_reason"] = choice.get("finish_reason")
            content = msg.get("content") or ""
            out["content"] = content
            usage = body.get("usage", {})
            out["prompt_tokens"] = usage.get("prompt_tokens", 0)
            out["completion_tokens"] = usage.get("completion_tokens", 0)
            out["total_tokens"] = usage.get("total_tokens", 0)
            metrics = body.get("_sllm_metrics") or {}
            out["ttft_s"] = metrics.get("ttft_s")
            out["tpot_s"] = metrics.get("tpot_s")
            out["e2e_s"] = metrics.get("e2e_s")
        except Exception as e:
            # Could be a status error or JSON decode error
            out["error"] = f"{type(e).__name__}: {str(e)}"
            try:
                out["error_response_text"] = resp.text[:500]
            except Exception:
                pass

        return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create workload for shadow scale-up benchmark.",
    )
    parser.add_argument(
        "--url",
        default=DEFAULT_URL,
        help=f"SLLM server base URL (default: {DEFAULT_URL})",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Model id (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=10240,
        help="Maximum number of tokens to generate",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=600.0,
        help="Per-request HTTP timeout in seconds",
    )
    parser.add_argument(
        "--dataset-path",
        required=True,
        help="Dataset path",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    context = Context(
        url=args.url,
        model=args.model,
        timeout=args.timeout,
        dataset_path=args.dataset_path,
        max_tokens=args.max_tokens,
    )

    loop = asyncio.new_event_loop()

    def handle_signal(signum, frame) -> None:
        print(f"Received signal {signum}; stopping loop", file=sys.stderr)
        loop.stop()

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    try:
        loop.run_until_complete(context.run())
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    finally:
        loop.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
