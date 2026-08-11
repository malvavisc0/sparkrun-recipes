# Benchmarks

Throughput benchmarks for every recipe in this repo, run with [llama-benchy](https://github.com/Sean-Vallo/llama-benchy) against a vLLM endpoint on a single DGX Spark (128 GB unified LPDDR5X).

## Layout

Each model gets two artifacts in this folder, named `<model>.{csv,png}`:

| File | Purpose |
|---|---|
| `<model>.csv` | Raw llama-benchy CSV output (source of truth) |
| `<model>.png` | 4-panel chart (pp/tg t/s vs concurrency, ctx_pp/ctx_tg t/s vs depth) |

To add a new model: run the bench command from the project [README](../README.md), then generate the chart:

```bash
uvx --with pandas --with matplotlib python scripts/chart_benchy.py \
  benchmarks/<model>.csv benchmarks/<model>.png
```

## Models

| Model | Recipe | Bench | Chart |
|---|---|---|---|
| `KAT-Coder-V2.5-Dev` | [kat-coder-v2.5-dev-nvfp4.yaml](../kat-coder-v2.5-dev-nvfp4.yaml) | [csv](kat-coder-v2.5-dev-nvfp4.csv) | [png](kat-coder-v2.5-dev-nvfp4.png) |

## KAT-Coder-V2.5-Dev

![pp/tg/ctx throughput](kat-coder-v2.5-dev-nvfp4.png)

### Headline numbers (single stream, concurrency = 1)

| Depth | pp t/s | tg t/s | ttfr (ms) |
|---:|---:|---:|---:|
| 0 | 7 660 | 66 | 354 |
| 4 096 | 3 684 | 65 | 637 |
| 8 192 | 3 442 | 64 | 676 |
| 16 384 | 2 928 | 63 | 781 |
| 32 768 | 2 268 | 60 | 988 |
| 65 535 | 1 896 | 56 | 1 162 |
| 100 000 | 1 150 | 51 | 1 863 |

### Real-life expectations

- **~51–66 tokens/s** sustained generation per interactive coding session (drops with context depth). Thinking mode is on by default, so visible-output latency is higher than raw ttfr suggests — reasoning tokens precede the first user-visible token.
- **Sub-second prefill** for prompts up to ~32k context; ~680 ms ttfr at 8k, rising to ~1.9 s at 100k.
- Concurrency > 1 now degrades aggregate throughput sharply under thinking mode (prefill collapses ~50–70% at c5/c10) — **stay at c=1 for interactive coding**. Scale users by adding replicas, not by raising concurrency on one process.

See the project [README](../README.md) for the exact `llama-benchy` command used to produce these numbers.