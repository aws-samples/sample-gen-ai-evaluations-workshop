"""
Capacity-aware Amazon Bedrock client for the Capacity Management module.

Three layers, each usable on its own:

    TokenBucket      continuous-refill rate limiter, the algorithm Bedrock uses
    QuotaSimulator   one bucket per model, driven by a hard-coded quota table
    CapacityRouter   quota-aware wrapper around bedrock-runtime Converse

Why simulate quotas at all? Real Bedrock quotas are high enough that hitting
them in a workshop would take thousands of requests and real money. We impose
much smaller artificial limits so the throttling behaviour is observable in
seconds. The mechanics are identical; only the numbers are smaller.

The simulated limiter is also not as artificial as it first looks. Large
platform teams commonly put a per-workload or per-tenant limiter *in front of*
Bedrock so that one workload cannot drain the account's quota. In that setup a
local limiter that knows its own budget is exactly what you run in production.
"""

from __future__ import annotations

import json
import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

import boto3
from botocore.config import Config


# ---------------------------------------------------------------------------
# Clocks
# ---------------------------------------------------------------------------

class RealClock:
    """Wall-clock time. Sleeping actually sleeps."""

    def now(self) -> float:
        return time.monotonic()

    def sleep(self, seconds: float) -> None:
        if seconds > 0:
            time.sleep(seconds)


class VirtualClock:
    """Fake clock that jumps forward instead of waiting.

    Lets us model a 20-minute workload in milliseconds with no Bedrock calls,
    which is how the notebook explores quota scenarios for free.
    """

    def __init__(self, start: float = 0.0):
        self._t = start
        self._lock = threading.Lock()

    def now(self) -> float:
        return self._t

    def sleep(self, seconds: float) -> None:
        if seconds > 0:
            with self._lock:
                self._t += seconds


# ---------------------------------------------------------------------------
# Layer 1: the token bucket
# ---------------------------------------------------------------------------

class TokenBucket:
    """A continuous-refill token bucket.

    This is the important distinction from a fixed window. With a fixed window
    of "5 per minute" you spend 5 and then wait for the top of the next minute.
    With a token bucket the bucket refills *continuously* at ``per_minute / 60``
    tokens per second, so after spending all 5 you get one more roughly every
    12 seconds rather than 5 more all at once a minute later.

    ``capacity`` is the burst size: the most you can spend at once after an
    idle period.
    """

    def __init__(self, per_minute: float, capacity: Optional[float] = None,
                 clock=None):
        if per_minute <= 0:
            raise ValueError("per_minute must be positive")
        self.per_minute = float(per_minute)
        self.refill_per_second = self.per_minute / 60.0
        self.capacity = float(capacity if capacity is not None else per_minute)
        self._clock = clock or RealClock()
        self._tokens = self.capacity
        self._updated = self._clock.now()
        self._lock = threading.Lock()

    # -- internals ---------------------------------------------------------
    def _refill_locked(self) -> None:
        now = self._clock.now()
        elapsed = now - self._updated
        if elapsed > 0:
            self._tokens = min(self.capacity,
                               self._tokens + elapsed * self.refill_per_second)
            self._updated = now

    # -- public API --------------------------------------------------------
    @property
    def tokens(self) -> float:
        """Tokens currently available (after accounting for refill)."""
        with self._lock:
            self._refill_locked()
            return self._tokens

    def try_acquire(self, n: float = 1.0) -> bool:
        """Spend ``n`` tokens if available. Never blocks."""
        with self._lock:
            self._refill_locked()
            if self._tokens >= n:
                self._tokens -= n
                return True
            return False

    def time_until_available(self, n: float = 1.0) -> float:
        """Seconds until ``n`` tokens exist. 0.0 if they already do."""
        with self._lock:
            self._refill_locked()
            if self._tokens >= n:
                return 0.0
            return (n - self._tokens) / self.refill_per_second

    def reset(self) -> None:
        """Refill to full. Call between notebook runs for repeatable timings."""
        with self._lock:
            self._tokens = self.capacity
            self._updated = self._clock.now()


# ---------------------------------------------------------------------------
# Layer 2: per-model quota simulation
# ---------------------------------------------------------------------------

class SimulatedThrottlingException(Exception):
    """Raised when no model in the candidate list has capacity.

    Stands in for the real Bedrock ``ThrottlingException`` (HTTP 429).
    """

    def __init__(self, models: Sequence[str], retry_after: float):
        self.models = list(models)
        self.retry_after = retry_after
        super().__init__(
            f"No capacity on {list(models)}; retry in {retry_after:.1f}s"
        )


class QuotaSimulator:
    """Holds one requests-per-minute bucket per model.

    Only RPM is modelled. Bedrock also enforces tokens-per-minute on most
    endpoints (and some endpoints enforce RPM only), but the routing lesson is
    identical whichever limit binds first. In production, monitor both.
    """

    def __init__(self, quotas: Dict[str, float], clock=None):
        self.clock = clock or RealClock()
        self.quotas = dict(quotas)
        self.buckets = {m: TokenBucket(rpm, clock=self.clock)
                        for m, rpm in quotas.items()}

    def has_capacity(self, model: str) -> bool:
        return self.buckets[model].tokens >= 1

    def try_acquire(self, model: str) -> bool:
        return self.buckets[model].try_acquire(1)

    def time_until_available(self, model: str) -> float:
        return self.buckets[model].time_until_available(1)

    def soonest(self, models: Sequence[str]) -> float:
        """Shortest wait across ``models`` until one of them frees up."""
        return min(self.time_until_available(m) for m in models)

    def total_rpm(self, models: Optional[Sequence[str]] = None) -> float:
        models = models if models is not None else list(self.quotas)
        return sum(self.quotas[m] for m in models)

    def snapshot(self) -> Dict[str, float]:
        return {m: round(b.tokens, 3) for m, b in self.buckets.items()}

    def reset(self) -> None:
        for b in self.buckets.values():
            b.reset()


# ---------------------------------------------------------------------------
# Provider normalisation
# ---------------------------------------------------------------------------

# Models that reject the `temperature` inference parameter. Sending it produces
# a ValidationException, so a multi-provider client has to special-case it.
NO_TEMPERATURE = {
    "us.anthropic.claude-sonnet-5",
    "global.anthropic.claude-sonnet-5",
    "us.openai.gpt-5.6-luna",
    "global.openai.gpt-5.6-luna",
    "us.openai.gpt-5.6-terra",
    "us.openai.gpt-5.6-sol",
}


def extract_text(response: dict) -> str:
    """Pull assistant text out of a Converse response, across providers.

    The obvious ``content[0]["text"]`` is not portable: gpt-oss models put a
    ``reasoningContent`` block first, so index 0 has no ``text`` key at all.
    Scan every block and concatenate the text ones.
    """
    blocks = response["output"]["message"]["content"]
    return "\n".join(b["text"] for b in blocks if "text" in b).strip()


def build_inference_config(model_id: str, max_tokens: int = 512,
                           temperature: Optional[float] = 0.0) -> dict:
    cfg: Dict[str, object] = {"maxTokens": max_tokens}
    if temperature is not None and model_id not in NO_TEMPERATURE:
        cfg["temperature"] = temperature
    return cfg


# ---------------------------------------------------------------------------
# Layer 3: the capacity-aware router
# ---------------------------------------------------------------------------

@dataclass
class InvocationRecord:
    """Everything the router observed about one logical request."""
    model: Optional[str] = None          # model that actually served it
    text: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    latency_s: float = 0.0               # time in the Bedrock call
    queue_wait_s: float = 0.0            # time blocked on simulated quota
    total_s: float = 0.0                 # queue wait + latency + retries
    fallback_depth: int = 0              # 0 = first choice in the list
    throttle_events: int = 0             # times no candidate had capacity
    real_throttles: int = 0              # genuine Bedrock 429s we retried
    attempts: int = 0
    error: Optional[str] = None
    meta: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.error is None


class CapacityRouter:
    """Quota-aware Bedrock client.

    Three ways to call it::

        # 1. one model
        router.invoke(prompt=p, model="us.openai.gpt-5.6-luna")

        # 2. priority-ordered candidates; first with capacity wins
        router.invoke(prompt=p, models=[luna, gpt_oss, haiku])

        # 3. per-model prompts, so each model gets the prompt tuned for it
        router.invoke(models=[luna, gpt_oss], prompts={luna: p1, gpt_oss: p2})

    ``mode="queue"`` waits for quota to refill; ``mode="fail"`` raises
    ``SimulatedThrottlingException`` immediately, reproducing a naive client.
    """

    def __init__(self, quotas: Dict[str, float], region: str = "us-east-1",
                 mode: str = "queue", clock=None, client=None,
                 max_queue_wait_s: float = 300.0,
                 real_throttle_retries: int = 6):
        if mode not in ("queue", "fail"):
            raise ValueError("mode must be 'queue' or 'fail'")
        self.mode = mode
        self.clock = clock or RealClock()
        self.quota = QuotaSimulator(quotas, clock=self.clock)
        self.max_queue_wait_s = max_queue_wait_s
        self.real_throttle_retries = real_throttle_retries
        self.region = region
        self._client = client or boto3.client(
            "bedrock-runtime",
            region_name=region,
            config=Config(retries={"max_attempts": 3, "mode": "standard"},
                          read_timeout=120, connect_timeout=10),
        )
        self._lock = threading.Lock()
        self.history: List[InvocationRecord] = []

    # -- quota bookkeeping -------------------------------------------------
    def reset(self) -> None:
        """Refill all buckets and clear history."""
        self.quota.reset()
        with self._lock:
            self.history = []

    # -- the actual Bedrock call ------------------------------------------
    def _call_bedrock(self, model: str, prompt: str, max_tokens: int,
                      temperature: Optional[float], rec: InvocationRecord) -> dict:
        """Invoke Converse, retrying genuine Bedrock throttles with backoff.

        The simulated bucket governs *our* budget. This retry loop handles the
        real service saying no, which can still happen underneath us.
        """
        for attempt in range(self.real_throttle_retries):
            try:
                started = time.monotonic()
                resp = self._client.converse(
                    modelId=model,
                    messages=[{"role": "user", "content": [{"text": prompt}]}],
                    inferenceConfig=build_inference_config(model, max_tokens, temperature),
                )
                rec.latency_s += time.monotonic() - started
                return resp
            except Exception as exc:                      # noqa: BLE001
                name = type(exc).__name__
                throttled = name in ("ThrottlingException", "TooManyRequestsException") \
                    or "ThrottlingException" in str(exc)
                if throttled and attempt < self.real_throttle_retries - 1:
                    rec.real_throttles += 1
                    backoff = (2 ** attempt) + random.uniform(0, 1)
                    self.clock.sleep(backoff)
                    continue
                raise

    # -- routing -----------------------------------------------------------
    def invoke(self, prompt: Optional[str] = None, model: Optional[str] = None,
               models: Optional[Sequence[str]] = None,
               prompts: Optional[Dict[str, str]] = None,
               max_tokens: int = 512, temperature: Optional[float] = 0.0,
               mode: Optional[str] = None, meta: Optional[dict] = None
               ) -> InvocationRecord:
        mode = mode or self.mode
        candidates = self._resolve_candidates(model, models, prompts)
        rec = InvocationRecord(meta=dict(meta or {}))
        started = self.clock.now()
        deadline = started + self.max_queue_wait_s

        while True:
            # Walk the priority list looking for a model with capacity.
            for depth, candidate in enumerate(candidates):
                if not self.quota.try_acquire(candidate):
                    continue
                rec.model = candidate
                rec.fallback_depth = depth
                rec.attempts += 1
                text = self._prompt_for(candidate, prompt, prompts)
                try:
                    resp = self._call_bedrock(candidate, text, max_tokens,
                                              temperature, rec)
                    rec.text = extract_text(resp)
                    rec.input_tokens = resp["usage"]["inputTokens"]
                    rec.output_tokens = resp["usage"]["outputTokens"]
                except Exception as exc:                  # noqa: BLE001
                    rec.error = f"{type(exc).__name__}: {exc}"
                rec.total_s = self.clock.now() - started
                self._record(rec)
                return rec

            # Nobody had capacity.
            rec.throttle_events += 1
            wait = self.quota.soonest(candidates)
            if mode == "fail":
                rec.error = "SimulatedThrottlingException"
                rec.total_s = self.clock.now() - started
                self._record(rec)
                raise SimulatedThrottlingException(candidates, wait)

            if self.clock.now() + wait > deadline:
                rec.error = (f"Queue wait exceeded max_queue_wait_s="
                             f"{self.max_queue_wait_s}s")
                rec.total_s = self.clock.now() - started
                self._record(rec)
                return rec

            # Sleep a hair past the refill point to avoid a tight spin.
            self.clock.sleep(wait + 0.01)
            rec.queue_wait_s = self.clock.now() - started

    # -- helpers -----------------------------------------------------------
    def _resolve_candidates(self, model, models, prompts) -> List[str]:
        if model and models:
            raise ValueError("pass either `model` or `models`, not both")
        if model:
            candidates = [model]
        elif models:
            candidates = list(models)
        elif prompts:
            candidates = list(prompts)
        else:
            raise ValueError("pass one of `model`, `models`, or `prompts`")
        unknown = [m for m in candidates if m not in self.quota.buckets]
        if unknown:
            raise ValueError(f"no quota configured for {unknown}")
        return candidates

    @staticmethod
    def _prompt_for(model, prompt, prompts) -> str:
        if prompts and model in prompts:
            return prompts[model]
        if prompt is None:
            raise ValueError(f"no prompt available for {model}")
        return prompt

    def _record(self, rec: InvocationRecord) -> None:
        with self._lock:
            self.history.append(rec)


# ---------------------------------------------------------------------------
# Batch execution
# ---------------------------------------------------------------------------

def run_workload(router: CapacityRouter, items: Sequence[dict],
                 prompt_fn: Callable[[dict, str], str],
                 model: Optional[str] = None,
                 models: Optional[Sequence[str]] = None,
                 max_workers: int = 12, max_tokens: int = 512,
                 mode: Optional[str] = None,
                 progress: bool = True) -> dict:
    """Push ``items`` through the router concurrently.

    ``prompt_fn(item, model_id)`` builds the prompt, so callers can vary the
    prompt per model — which is how the optimized-prompt run works.

    Returns a dict with the per-item records and the wall clock for the batch,
    since wall clock is the headline capacity metric.
    """
    candidates = [model] if model else list(models or [])
    results: List[Optional[InvocationRecord]] = [None] * len(items)
    done = {"n": 0}
    lock = threading.Lock()

    def one(idx_item):
        idx, item = idx_item
        prompts = {m: prompt_fn(item, m) for m in candidates}
        try:
            rec = router.invoke(models=candidates, prompts=prompts,
                                max_tokens=max_tokens, mode=mode,
                                meta={"index": idx, "item": item})
        except SimulatedThrottlingException as exc:
            rec = InvocationRecord(error="SimulatedThrottlingException",
                                   throttle_events=1,
                                   meta={"index": idx, "item": item,
                                         "retry_after": exc.retry_after})
        results[idx] = rec
        if progress:
            with lock:
                done["n"] += 1
                if done["n"] % 25 == 0 or done["n"] == len(items):
                    print(f"   {done['n']}/{len(items)} complete", flush=True)

    t0 = time.monotonic()
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        list(pool.map(one, enumerate(items)))
    wall = time.monotonic() - t0

    return {"records": results, "wall_clock_s": wall,
            "candidates": candidates, "n_items": len(items)}


# ---------------------------------------------------------------------------
# Free, instant what-if simulation (no Bedrock calls)
# ---------------------------------------------------------------------------

def simulate_routing(n_requests: int, quotas: Dict[str, float],
                     priority: Optional[Sequence[str]] = None,
                     service_time_s: float = 1.0, concurrency: int = 12,
                     mode: str = "queue") -> dict:
    """Model a workload against a quota table using a virtual clock.

    No Bedrock calls, no cost, runs instantly. Use it to answer "how long would
    50,000 requests take against this routing table?" without paying for it.
    """
    clock = VirtualClock()
    quota = QuotaSimulator(quotas, clock=clock)
    order = list(priority or quotas)

    # Virtual workers pull from a shared queue; each holds a slot for
    # service_time_s of simulated time.
    worker_free_at = [0.0] * concurrency
    assignments: List[dict] = []
    throttled = 0

    for i in range(n_requests):
        w = min(range(concurrency), key=lambda k: worker_free_at[k])
        t = max(worker_free_at[w], clock.now())
        if t > clock.now():
            clock.sleep(t - clock.now())

        chosen, waited = None, 0.0
        while chosen is None:
            for depth, m in enumerate(order):
                if quota.try_acquire(m):
                    chosen, chosen_depth = m, depth
                    break
            if chosen is None:
                wait = quota.soonest(order)
                if mode == "fail":
                    throttled += 1
                    break
                clock.sleep(wait + 1e-6)
                waited += wait
        if chosen is None:
            continue
        start = clock.now()
        worker_free_at[w] = start + service_time_s
        assignments.append({"index": i, "model": chosen, "depth": chosen_depth,
                            "queue_wait_s": waited, "start_s": start,
                            "end_s": start + service_time_s})

    makespan = max((a["end_s"] for a in assignments), default=0.0)
    per_model: Dict[str, int] = {}
    for a in assignments:
        per_model[a["model"]] = per_model.get(a["model"], 0) + 1
    waits = sorted(a["queue_wait_s"] for a in assignments)

    return {
        "n_requests": n_requests,
        "served": len(assignments),
        "throttled": throttled,
        "wall_clock_s": makespan,
        "throughput_rps": (len(assignments) / makespan) if makespan else 0.0,
        "per_model_counts": per_model,
        "total_rpm": quota.total_rpm(order),
        "queue_wait_mean_s": (sum(waits) / len(waits)) if waits else 0.0,
        "queue_wait_p95_s": waits[int(0.95 * (len(waits) - 1))] if waits else 0.0,
        "assignments": assignments,
    }
