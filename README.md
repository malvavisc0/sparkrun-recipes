# sparkrun-recipes

A collection of [sparkrun](https://sparkrun.dev) recipes for serving inference models on NVIDIA DGX Spark (128 GB unified memory) and other Blackwell-class single-GPU hosts. Each recipe is a self-contained YAML file with a known-good vLLM serving configuration.

## Recipes

| Recipe | Model | Params | Dtype | Use case | Port |
|---|---|---|---|---|---|
| [kat-coder](kat-coder-v2.5-dev-nvfp4.yaml) | KAT-Coder V2.5 Dev | 35B A3B MoE | NVFP4 | Agentic coding | 9000 |
| [ornith](ornith-35b-fp8.yaml) | Ornith 1.0 35B + MTP | 35B MoE | FP8 E4M3 | Agentic coding (fast) | 8000 |
| [nemotron-omni](nemotron3-nano-omni-30b-nvfp4.yaml) | Nemotron 3 Nano Omni 30B | 30B A3B MoE | NVFP4 | Multimodal reasoning | 8000 |
| [qwen2.5-coder](qwen2.5-coder-14b-instruct-nvfp4-anima.yaml) | Qwen2.5-Coder-14B-Instruct | 14B | NVFP4 | Autocomplete / FIM | 8100 |
| [nemotron-embed](nemotron-3-embed-1b-nvfp4.yaml) | Nemotron 3 Embed 1B | 1.14B | NVFP4 | Text embeddings / RAG | 8000 |

All recipes use the [vLLM](https://docs.vllm.ai) runtime on the `eugr/spark-vllm:nightly-20260807` container.

## Usage

```bash
sparkrun run kat-coder-v2.5-dev-nvfp4.yaml --solo
sparkrun run ornith-35b-fp8.yaml -o port=9001
sparkrun run nemotron-3-embed-1b-nvfp4.yaml -o gpu_memory_utilization=0.10
```

Override any default with `-o key=value`. See the [Recipe Format](https://sparkrun.dev/recipes/format/) docs for the full schema.

## Authentication

Recipes contain **no API keys**. vLLM reads `VLLM_API_KEY` from the container environment, injected via sparkrun's [cluster-level env](https://sparkrun.dev/clusters/managing/#cluster-level-environment):

```yaml
# ~/.config/sparkrun/clusters/<name>.yaml
env:
  VLLM_API_KEY: "${VLLM_API_KEY}"
env_file: /home/<you>/.sparkrun.env
```

```bash
# ~/.sparkrun.env
VLLM_API_KEY=<your-secret>
```

Clients authenticate with `Authorization: Bearer <your-secret>`. Recipe `env` values are passed through literally (sparkrun 0.3.0+) and will not expand host secrets, so keys must live in the env file, never in the recipe.

## Model details

### KAT-Coder V2.5 Dev — `kat-coder-v2.5-dev-nvfp4.yaml`

NVFP4 quantization of [Kwaipilot/KAT-Coder-V2.5-Dev](https://huggingface.co/sakamakismile/KAT-Coder-V2.5-Dev-NVFP4) by Lna-Lab. A 35B-A3B agentic-coding MoE built on the Qwen3.5 architecture (`qwen3_5_moe`).

- **SWE-bench Verified: 69.4%** (upstream)
- 70 GB bf16 → 21.9 GB NVFP4; fits a single 24 GB Blackwell card
- ~122 tok/s single-stream (2× RTX PRO 2000, 16 GB)
- `qwen3` reasoning parser, `qwen3_coder` tool parser, fixed chat template (via the `fix-qwen3.6-chat-template` mod)
- No vision, no MTP tensors shipped
- License: Apache-2.0

### Ornith 1.0 35B — `ornith-35b-fp8.yaml`

[Ornith-1.0-35B](https://huggingface.co/deepreinforce-ai/Ornith-1.0-35B) is a self-improving agentic-coding MoE family from DeepReinforce, post-trained on Qwen 3.5. This recipe serves the [FP8 E4M3 build](https://huggingface.co/kyr0/Ornith-35B-FP8-E4M3-MTP) by kyr0 with a grafted MTP speculative-decoding sidecar.

- **SWE-bench Verified: 75.6%**, SWE-bench Pro 50.4%, Terminal-Bench 2.1 (Terminus-2) 64.2%
- FP8 weights (35.8 GB) + MTP sidecar (1.6 GB); KV cache FP8
- ~18% faster with MTP: 751 tok/s vs 635 tok/s baseline (H200); 69.2% avg draft acceptance, mean acceptance length 2.38
- `qwen3` reasoning parser, `qwen3_xml` tool parser
- License: MIT

### Nemotron 3 Nano Omni 30B — `nemotron3-nano-omni-30b-nvfp4.yaml`

[Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4](https://huggingface.co/lactroiii/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4) — a 31B Mamba2-Transformer hybrid MoE (~3B active) with integrated video, audio, image, and text understanding. The only multimodal recipe here.

- Modalities in: video/audio/image/text; out: text. 256K context, 34-language English-focused support
- OSWorld 47.4, VideoMME 72.2, OCRBenchV2 (EN) 67.0, Charxiv Reasoning 63.6
- NVFP4 stays within 0.38 mean pts of BF16 across 9 multimodal benchmarks (20.9 GB)
- `nemotron_v3` reasoning parser, `qwen3_coder` tool parser
- `pre_exec` installs `vllm[audio]`; `--video-pruning-rate 0.60`, `--media-io-kwargs` for 2 FPS / 256 frames
- License: NVIDIA Open Model Agreement (NOMA)

### Qwen2.5-Coder-14B-Instruct — `qwen2.5-coder-14b-instruct-nvfp4-anima.yaml`

[Qwen2.5-Coder-14B-Instruct-NVFP4-anima](https://huggingface.co/ilessio-aiflowlab/Qwen2.5-Coder-14B-Instruct-NVFP4-anima) — an NVFP4 quantization by RobotFlow Labs, built for the Jetson AGX Thor (Blackwell `sm_110a`).

- Sized for **autocomplete / fill-in-the-middle** via `/v1/completions`, not agentic chat
- 14B, ~3.5× smaller than bf16; `--attention-backend TRITON_ATTN`, FP8 KV cache
- Instruct model — **no thinking mode**, no tool/reasoning parsers
- Short context (`max_model_len 8192`) and high concurrency (`max_num_seqs 128`) tuned for short FIM requests
- License: Apache-2.0 (base Qwen2.5-Coder-14B-Instruct)

### Nemotron 3 Embed 1B — `nemotron-3-embed-1b-nvfp4.yaml`

[Nemotron-3-Embed-1B-NVFP4](https://huggingface.co/nvidia/Nemotron-3-Embed-1B-NVFP4) — a 1.14B text-embedding model for multilingual retrieval/RAG, built on a Ministral-3 pruned encoder with bidirectional attention + average pooling.

- 2048-dim embeddings (sliceable to 1024/512 with re-normalization), max seq 32768, 34 languages
- **RTEB 72.00** (BF16 72.38, −0.38); QAD-recovered for long sequences
- vLLM auto-detects NVFP4 + pooling metadata — no quantization/runner flags
- Sparse `--cudagraph-capture-sizes` profile; set a persistent `VLLM_CACHE_ROOT` to reuse FP4 GEMM autotuning across restarts (first startup takes several minutes)
- Requires vLLM ≥ 0.25.0 (0.23.x/0.24.x have known issues with this NVFP4 family)
- Recommended endpoint: `/v2/embed` with `input_type: "query"` / `"document"` (OpenAI `/v1/embeddings` works with manual `query:` / `passage:` prefixes)
- License: OpenMDW-1.1

## GPU memory utilization

Recipes target a 128 GB DGX Spark (unified LPDDR5X). `gpu_memory_utilization` is the fraction of total memory each vLLM process reserves; on unified memory, over-reserving starves the OS and can stall weight prefetch.

| Recipe | gpu_memory_utilization | Weights (approx) |
|---|---|---|
| kat-coder | 0.65 | ~22 GB |
| ornith | 0.60 | ~37 GB |
| nemotron-omni | 0.70 | ~21 GB |
| qwen2.5-coder | 0.20 | ~11 GB |
| nemotron-embed | 0.07 | ~1 GB |

When running multiple recipes on the same host, their utilizations must sum to ≤ ~0.88 (leaving headroom for the OS). Override per launch with `-o gpu_memory_utilization=0.3`.

## Benchmark

```bash
export SPARK_RECIPE='recipe'
export SPARK_MODEL='model'
export SPARK_BASE_URL='http://server:9000/v1'
export SPARK_TOKENIZER='hf-org/hf-model'   # must match the served model's tokenizer
export LLAMA_API_KEY='api-key'
mkdir -p ./benchmarks
uvx llama-benchy \
      --base-url "$SPARK_BASE_URL" \
      --api-key "$LLAMA_API_KEY" \
      --model "$SPARK_MODEL" \
      --tokenizer Kwaipilot/KAT-Coder-V2.5-Dev \
      --pp 2048 \
      --tg 128 \
      --exact-tg \
      --depth 0 4096 8192 16384 32768 65535 100000 \
      --enable-prefix-caching \
      --concurrency 1 2 5 10 \
      --latency-mode generation \
      --exit-on-first-fail \
      --format csv --save-result ./benchmarks/$SPARK_RECIPE.csv
```

Then plot it:

```bash
uvx --with pandas --with matplotlib python scripts/chart_benchy.py \
  ./benchmarks/$SPARK_RECIPE.csv ./benchmarks/$SPARK_RECIPE.png
```

## Notes

- **KAT chat-template mod**: `kat-coder` references `mods/fix-qwen3.6-chat-template`, which resolves via sparkrun's mod lookup (adjacent dir, same registry, or `@eugr` fallback). If it isn't available, `sparkrun run` will fail with the paths it tried.
- **nemotron-omni context**: the model card's single-GPU example uses 131072; this recipe sets 262144. If you hit OOM on unified memory, lower `max_model_len` and/or `gpu_memory_utilization`.
- **vLLM versions**: qwen2.5-coder and nemotron-embed have specific version requirements (0.23 and ≥0.25 respectively). Verify the shared container satisfies the model you're serving.
