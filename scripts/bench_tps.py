#!/usr/bin/env python3
"""Measure decode tok/s of a running sparkrun vLLM/SGLang deployment.

Reads the API key from ~/.sparkrun.env (SGLANG_API_KEY) or from the
LLAMA_API_KEY / VLLM_API_KEY env vars, then drives streaming chat
completions and reports throughput. Uses only the Python standard library.

Two probes:

  default   Short-context coding probe (LRUCache net-decode A/B methodology):
            T=0 greedy, thinking off, streamed. TTFT and decode tok/s are
            reported separately so the number is comparable to the cookbook's
            net-decode figures, not prefill-inclusive. One discarded warmup,
            then N measured runs.

  --depth N Deep-context probe (repeatable): builds a synthetic heterogeneous
            codebase of ~N prompt tokens (varied names, signatures, injected
            bugs — fixed seed, reproducible) and asks for analysis + a fix,
            the workload shape that actually stresses speculative decoding.
            Reports TTFT/prefill, decode tok/s, and the server's
            spec_accept_length / spec_accept_rate before -> after, so a low
            decode number can be attributed to acceptance decay (drafter) vs
            raw KV bandwidth. Prompt sizing is an estimate: on HTTP 400
            (overshot the context window) the probe shrinks 20% and retries.

Numbers from these two probes are NOT comparable: the short probe is the
best case (fresh context, deterministic code continuation); the depth probe
approaches realistic agentic sessions. For DFlash2/DSpark-style drafters,
decode at depth is acceptance-bound — cite both, e.g. qwen3.8-27b-dflash on
GB10: 94 tok/s short, 34 tok/s at 200k hostile depth (accept 12.1 -> 4.1).

Usage:
    python3 scripts/bench_tps.py \
        --url http://omnitron.tago.lan:9000/v1 \
        --model qwen3.8-27b \
        --max-tokens 512 \
        --depth 50000 --depth 200000

Options:
    --url        Base OpenAI URL (default: http://omnitron.tago.lan:9000/v1)
    --model      Served model name (default: Qwen3.8-27B)
    --max-tokens Target output tokens per request (default: 512)
    --runs       Measured requests to average, after warmup (default: 5)
    --depth      Deep-context probe of ~N prompt tokens (repeatable)
    --no-warmup  Skip the discarded warmup request
"""
import argparse
import json
import os
import random
import re
import sys
import time
import urllib.error
import urllib.request
from http.client import HTTPException


class NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    # urllib's default redirect handler re-sends every header except
    # content-length/content-type to the Location target — including the
    # Authorization bearer. Over plaintext http:// that lets a 302 (MITM or a
    # redirecting endpoint) harvest the API key. Return None to refuse 3xx;
    # urllib raises HTTPError, which we surface as a non-leaking error.
    def redirect_request(self, req, fp, code, msg, hdrs, newurl):
        return None


_OPENER = urllib.request.build_opener(NoRedirectHandler)


def _open(req: urllib.request.Request, timeout: float):
    return _OPENER.open(req, timeout=timeout)


def load_api_key() -> str:
    key = os.getenv("SGLANG_API_KEY") or os.getenv("VLLM_API_KEY") or os.getenv("LLAMA_API_KEY")
    if key:
        return key
    env_path = os.path.expanduser("~/.sparkrun.env")
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                m = re.match(r"\s*(?:export\s+)?SGLANG_API_KEY\s*=\s*['\"]?([^'\"\s]+)", line)
                if m:
                    return m.group(1)
    raise SystemExit("no API key (set SGLANG_API_KEY/VLLM_API_KEY/LLAMA_API_KEY or ~/.sparkrun.env)")


def read_models(url: str, key: str) -> dict:
    models_url = url.rstrip("/") + "/models"
    req = urllib.request.Request(models_url, headers={"Authorization": f"Bearer {key}"})
    with _open(req, timeout=60) as r:
        return json.loads(r.read())


def metrics_snapshot(url: str, key: str):
    """Sample spec-decoding gauges from /metrics; None when unavailable.

    Only SGLang exposes these; vLLM deployments get None and the depth
    probe reports throughput without the acceptance attribution.
    """
    try:
        base = url.rstrip("/")
        if base.endswith("/v1"):
            base = base[: -len("/v1")]
        req = urllib.request.Request(base + "/metrics", headers={"Authorization": f"Bearer {key}"})
        out = {}
        with _open(req, timeout=30) as r:
            for raw in r:
                line = raw.decode(errors="replace")
                if line.startswith("sglang:spec_accept_length{"):
                    out["accept_len"] = float(line.rsplit(" ", 1)[1])
                elif line.startswith("sglang:spec_accept_rate{"):
                    out["accept_rate"] = float(line.rsplit(" ", 1)[1])
        return out or None
    except Exception:
        return None


def chat(req: dict, url: str, key: str, timeout: float) -> dict:
    """Stream one chat completion, capturing TTFT and decode timing.

    Returns a dict with: tokens, ttft (s), decode_elapsed (s), total_elapsed (s),
    decode_tps, total_tps, text. Token count comes from the server's usage
    chunk (stream_options.include_usage); a delta-count fallback is used only
    if the server omits usage.
    """
    data = json.dumps(req).encode()
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }
    start = time.perf_counter()
    http = urllib.request.Request(f"{url}/chat/completions", data=data, headers=headers,
                                  method="POST")
    first_token = None
    content_parts = []
    reasoning_parts = []
    usage = {}
    delta_token_count = 0
    with _open(http, timeout=timeout) as r:
        for raw in r:
            line = raw.decode(errors="replace").strip()
            if not line or not line.startswith("data:"):
                continue
            payload = line[len("data:"):].strip()
            if payload == "[DONE]":
                break
            try:
                chunk = json.loads(payload)
            except Exception:
                continue
            delta = (chunk.get("choices") or [{}])[0].get("delta") or {}
            saw_token = False
            if isinstance(delta.get("content"), str):
                if first_token is None:
                    first_token = time.perf_counter()
                content_parts.append(delta["content"])
                saw_token = True
            if isinstance(delta.get("reasoning_content"), str):
                if first_token is None:
                    first_token = time.perf_counter()
                reasoning_parts.append(delta["reasoning_content"])
                saw_token = True
            if saw_token:
                delta_token_count += 1
            if chunk.get("usage"):
                usage = chunk["usage"]
    end = time.perf_counter()

    text = "".join(content_parts)
    reasoning = "".join(reasoning_parts)
    n_tokens = int(usage.get("completion_tokens") or 0)
    if not n_tokens:
        # Fallback only if the server sent no usage chunk: count content+reasoning
        # deltas. A char-estimate is deliberately avoided — it is badly wrong for
        # code (braces, indentation, short tokens).
        n_tokens = max(delta_token_count, 1)

    ttft = (first_token - start) if first_token is not None else None
    total_elapsed = end - start
    # Decode span: first content delta -> last byte. This is the net-decode
    # figure comparable to the README's ndec.py (excludes prefill/TTFT).
    decode_elapsed = (end - first_token) if first_token is not None else total_elapsed
    decode_tps = n_tokens / decode_elapsed if decode_elapsed > 0 else 0.0
    total_tps = n_tokens / total_elapsed if total_elapsed > 0 else 0.0
    return {
        "tokens": n_tokens,
        "prompt_tokens": int(usage.get("prompt_tokens") or 0),
        "ttft": ttft,
        "decode_elapsed": decode_elapsed,
        "total_elapsed": total_elapsed,
        "decode_tps": decode_tps,
        "total_tps": total_tps,
        "text": text,
        "reasoning": reasoning,
    }


def build_payload(model: str, max_tokens: int) -> dict:
    # Coding-style probe that mirrors the README's LRUCache net-decode A/B:
    # deterministic (T=0), thinking off, streamed. DSpark's win is on
    # code/agents (51.5 vs 34.5); prose is its weak case, so don't benchmark a
    # verbose essay if the target workload is coding.
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are an expert Python engineer."},
            {"role": "user", "content": "Implement a thread-safe LRU cache in Python "
                                        "with O(1) get and put, using OrderedDict and "
                                        "a reentrant lock. Include a small test that "
                                        "exercises eviction and concurrent access. "
                                        "Output only the code."},
        ],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "top_p": 1.0,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
    }


# --- Deep-context probe: synthetic heterogeneous codebase -----------------
#
# Why not just repeat one module? A repeated template is the block-diffusion
# drafter's best case — near-deterministic continuation keeps acceptance ~12/16
# at any depth and measures ~94 tok/s at 200k, which overstates reality by ~3x.
# The generator below varies names, signatures, nesting and injects subtle
# bugs, and the task asks for analysis + a fix: the measured 34 tok/s at 200k
# on qwen3.8-27b-dflash matches pangoleen's edit-heavy band (38-41 @ 260k).

_SEED = 20260903
_VERBS = ["validate", "hash", "encode", "collapse", "rotate", "merge", "split", "index",
          "compact", "resolve", "sanitize", "tokenize", "encrypt", "shuffle", "detect"]
_NOUNS = ["config", "payload", "header", "record", "frame", "packet", "cursor", "ledger",
          "buffer", "schema", "token", "segment", "manifest", "digest", "envelope"]
_TYPES = ["dict", "list", "bytes", "str", "int", "tuple", "set", "Optional[list]"]
_STRUCTS = ["class", "def", "async def"]
_CHARS_PER_TOKEN = 3.7  # measured for this generator; sizing is refined on 400


def _block(rng: random.Random, i: int) -> str:
    v1, v2 = rng.choice(_VERBS), rng.choice(_NOUNS)
    v3, v4 = rng.choice(_VERBS), rng.choice(_NOUNS)
    t1, t2 = rng.choice(_TYPES), rng.choice(_TYPES)
    s = rng.choice(_STRUCTS)
    depth = rng.choice([1, 1, 2, 3])
    indent = "    "
    body = []
    body.append(f"{s} {v1}_{v2}_{i}(src: {t1}, mode: {t2} = 'fast') -> {t2}:")
    body.append(indent + "acc = " + rng.choice(["0", "[]", "{}", "None", "''"]))
    for d in range(depth):
        pre = indent * (d + 1)
        body.append(pre + f"for item in {'src' if d == 0 else 'chunk'}:")
        body.append(pre + indent + f"if item is {rng.choice(['None', 'not item', 'isinstance(item, bytes)'])}:")
        body.append(pre + indent * 2 + "continue")
        body.append(pre + indent + f"acc = {v3}_{v4}(acc, item, mode)")
    body.append(indent + "return acc")
    if rng.random() < 0.3:  # subtle falsy-value bug in some modules
        body.append("")
        body.append(f"def {v3}_{v4}(acc, item, mode):")
        body.append(indent + "if mode == 'fast':")
        body.append(indent * 2 + "return acc + item if item else acc  # BUG: falsy-zero dropped")
        body.append(indent + "return acc + item")
    return "\n".join(body) + "\n\n"


def build_depth_prompt(target_tokens: int) -> str:
    """Heterogeneous codebase prompt sized to ~target_tokens (seeded, reproducible)."""
    rng = random.Random(_SEED)
    parts = []
    total_chars = 0
    i = 0
    limit = target_tokens * _CHARS_PER_TOKEN
    while total_chars < limit:
        b = _block(rng, i)
        parts.append(b)
        total_chars += len(b)
        i += 1
    ctx = "".join(parts)
    return (
        "Below is a heterogeneous codebase of unrelated utility functions. "
        "Some modules contain a subtle bug that drops falsy values (0, '', empty).\n"
        "Task: identify the module names that contain the falsy-value bug, then "
        "write a corrected generic helper `merge_strict(acc, item, mode)` that "
        "preserves falsy items, and explain in one sentence why the original "
        "pattern was wrong.\n\n" + ctx)


def run_depth_probe(args, key: str, target_tokens: int) -> None:
    prompt = build_depth_prompt(target_tokens)
    payload = {
        "model": args.model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": args.max_tokens,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
    }
    before = metrics_snapshot(args.url, key)
    for attempt in range(3):
        try:
            r = chat(payload, args.url, key, timeout=max(600, args.max_tokens * 4))
            break
        except urllib.error.HTTPError as e:
            if e.code == 400 and attempt < 2:
                # Sizing overshot the context window: shrink 20% and retry.
                shrink = int(len(payload["messages"][0]["content"]) * 0.8)
                payload["messages"][0]["content"] = payload["messages"][0]["content"][:shrink]
                print(f"  (400: oversized prompt, shrunk to {shrink} chars, retrying)")
                continue
            raise
    after = metrics_snapshot(args.url, key)
    ttft = f"{r['ttft']:.1f}s" if r["ttft"] is not None else "n/a"
    line = (f"depth probe (~{target_tokens} tok target): prompt={r['prompt_tokens']:,} tok | "
            f"out={r['tokens']} tok | TTFT/prefill={ttft} | decode {r['decode_elapsed']:.1f}s "
            f"-> {r['decode_tps']:.1f} tok/s")
    if before and after:
        line += (f" | accept_len {before.get('accept_len', '?')} -> {after.get('accept_len', '?')}"
                 f" (rate {before.get('accept_rate', '?')} -> {after.get('accept_rate', '?')})")
    print(line)


def main() -> int:
    ap = argparse.ArgumentParser(description="Measure decode tok/s of a running inference deployment")
    ap.add_argument("--url", default="http://omnitron.tago.lan:9000/v1")
    ap.add_argument("--model", default="Qwen3.8-27B")
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--runs", type=int, default=5, help="measured runs after warmup (default: 5)")
    ap.add_argument("--depth", type=int, action="append", default=[],
                    help="deep-context probe of ~N prompt tokens (repeatable)")
    ap.add_argument("--no-warmup", action="store_true", help="skip the discarded warmup request")
    args = ap.parse_args()

    key = load_api_key()
    try:
        models = read_models(args.url, key)
        served = [m.get("id") for m in models.get("data", [])]
        print(f"served models on {args.url}: {served or '(none found)'}")
    except Exception as e:
        print(f"warning: could not fetch /models: {e}")

    payload = build_payload(args.model, args.max_tokens)

    if not args.no_warmup:
        print("warmup    : 1 request (discarded) ...", flush=True)
        try:
            chat(payload, args.url, key, timeout=600)
        except Exception as e:
            print(f"warning: warmup failed: {e}")

    results = [chat(payload, args.url, key, timeout=600) for _ in range(args.runs)]

    total_tokens = sum(r["tokens"] for r in results)
    total_decode = sum(r["decode_elapsed"] for r in results)
    total_wall = sum(r["total_elapsed"] for r in results)
    agg_decode = total_tokens / total_decode if total_decode > 0 else 0.0
    agg_total = total_tokens / total_wall if total_wall > 0 else 0.0

    print(f"model      : {args.model}")
    print(f"runs       : {len(results)} x max_tokens={args.max_tokens}, coding probe, thinking off, T=0, streamed")
    for i, r in enumerate(results, 1):
        ttft = f"{r['ttft']:.2f}s" if r["ttft"] is not None else "n/a"
        print(f"  run {i}: {r['tokens']} tok | TTFT {ttft} | decode {r['decode_elapsed']:.2f}s -> "
              f"{r['decode_tps']:.1f} tok/s | wall {r['total_elapsed']:.2f}s -> {r['total_tps']:.1f} tok/s")
    print(f"aggregate  : {total_tokens} tok | decode {total_decode:.2f}s -> {agg_decode:.1f} tok/s | "
          f"wall {total_wall:.2f}s -> {agg_total:.1f} tok/s")
    print("note       : decode tok/s excludes prefill/TTFT; "
          "wall tok/s is prefill-inclusive")

    for target in args.depth:
        run_depth_probe(args, key, target)

    return 0


if __name__ == "__main__":
    sys.exit(main())
