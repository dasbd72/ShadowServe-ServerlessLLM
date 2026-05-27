#!/usr/bin/env python3
"""Drive SLLM auto-scaler to add a second instance (shadow scale-up).

With deploy config ``auto_scaling_config.target: 1`` and ``max_instances: 2``,
the router scales up when ``request_count >= 2`` (two in-flight requests). The
first request uses the hot GPU instance; the second triggers creation of a
cold instance + shadow actor and KVHTS/KVSTC migration (if ``shadow_scale_up``).

Prerequisites:
  1. Ray + SLLM running (e.g. ``python start_ray.py`` then ``python start_sllm.py --deploy``).
  2. Model deployed from ``config-vllm-shadow.json`` (or matching ``--model``).
  3. Worker has at least 2 free GPUs on the same node (``worker_id_0``).

Example:
  export LLM_SERVER_URL=http://127.0.0.1:8343
  python examples/shadow_scale_up/trigger_scale_up.py --concurrent 2 --max-tokens 2048
"""

from __future__ import annotations

import argparse
import asyncio
import os
import signal
import sys
import time

import httpx

DEFAULT_URL = os.environ.get("LLM_SERVER_URL", "http://127.0.0.1:8343")
DEFAULT_MODEL = "Qwen/Qwen3-8B"


class Context:
    def __init__(
        self,
        url: str,
        model: str,
        concurrent: int,
        max_tokens: int,
        timeout: float,
        stagger_ms: int,
    ):
        self.url = url.rstrip("/")
        self.model = model
        self.concurrent = concurrent
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.stagger_ms = stagger_ms

    async def run(self) -> None:
        await self.get_health()

        futures: list[asyncio.Future[dict]] = []
        ttfts = []
        tpots = []
        e2es = []
        for i in range(self.concurrent):
            if self.stagger_ms > 0 and i > 0:
                await asyncio.sleep(self.stagger_ms / 1000.0)
            futures.append(asyncio.create_task(self.send_chat(i)))
        for future in asyncio.as_completed(futures):
            result = await future
            elapsed = result["elapsed_s"]
            completion_tokens = result.get("completion_tokens")
            latency = completion_tokens / elapsed
            ttfts.append(result.get("ttft_s"))
            tpots.append(result.get("tpot_s"))
            e2es.append(result.get("e2e_s"))
            print(
                f"  [{result['index']}] HTTP {result['status_code']} in "
                f"{result['elapsed_s']}s finish={result.get('finish_reason')} "
                f"prompt_tokens={result.get('prompt_tokens')}, "
                f"completion_tokens={result.get('completion_tokens')}, "
                f"total_tokens={result.get('total_tokens')}, "
                f"ttft={result.get('ttft_s')}s tpot={result.get('tpot_s')}s "
                f"e2e={result.get('e2e_s')}s"
            )
        print(f"Average ttft: {sum(ttfts) / len(ttfts)}s")
        print(f"Average tpot: {sum(tpots) / len(tpots)}s")
        print(f"Average e2e: {sum(e2es) / len(e2es)}s")

    async def get_health(self) -> None:
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{self.url}/health", timeout=10)
            resp.raise_for_status()

    async def send_chat(self, request_idx: int) -> dict:
        url = f"{self.url}/v1/chat/completions"
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": (
                        f"Request {request_idx}: count slowly from 1 to {self.max_tokens} "
                        "with one number per line."
                    ),
                }
            ],
            "max_tokens": self.max_tokens,
            "temperature": 0.0,
        }
        headers = {"Content-Type": "application/json"}

        t0 = time.perf_counter()
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                url, headers=headers, json=payload, timeout=self.timeout
            )
        elapsed = time.perf_counter() - t0
        out = {
            "index": request_idx,
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
        description="Send concurrent chat requests to trigger SLLM scale-up.",
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
        "--concurrent",
        type=int,
        default=2,
        help="Number of parallel in-flight requests (default: 2 to hit target=1 scaler)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=2048,
        help="max_tokens per request; keep high so requests overlap during scale-up",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=600.0,
        help="Per-request HTTP timeout in seconds",
    )
    parser.add_argument(
        "--stagger-ms",
        type=int,
        default=0,
        help="Delay between starting each request (ms); 0 = all at once",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    context = Context(
        url=args.url,
        model=args.model,
        concurrent=args.concurrent,
        max_tokens=args.max_tokens,
        timeout=args.timeout,
        stagger_ms=args.stagger_ms,
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
