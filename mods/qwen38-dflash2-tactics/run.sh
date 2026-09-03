#!/usr/bin/env bash
# Install the measured FlashInfer autotune tactic draws for Qwen3.8-27B DFlash2
# on DGX Spark (GB10) into the persistent SGLang runtime cache.
#
# Source: pangoleen/qwen3.8-27b-dgx-spark-dflash2 (RESULTS.md, 2026-09-01/02),
# measured on the same engine lineage this recipe runs (SGLang day-0 qwen38-27b
# image + upstream PRs #35371/#35496 = the pinned dev-cu13-qwen38-27b-dflash2
# digest), draft budget 16, MAX_RUNNING=4, thinking off.
#
# FlashInfer's autotuner re-times candidate kernels for every GEMM shape at
# boot. Timing noise picks different tactics per boot (the "boot lottery" —
# boots land up to 20% apart, and the kept draw is worth ~+16% over a median
# fresh draw). Autotuning is therefore left ON and a known-good draw is
# installed so every boot replays the same kernel plan instead of re-tuning.
#
# The autotuner caches to
#   $SGLANG_CACHE_DIR/flashinfer/autotune/<flashinfer-version>/<arch>/<key>/rank_tp0_pp0_dp0.json
# where <key> is sha256(model path|dtype|quant|moe backend|tp|pp|dp|ep|hf config
# class|skip_ops|draft_quant)[:16] — the FI *version is NOT part of the key*,
# only of the directory — and <arch> is sm<compute-capability> (sm121 on GB10).
# sparkrun mounts a persistent runtime cache at /cache/runtime and exports
# SGLANG_CACHE_DIR=/cache/runtime/sglang, so installed draws survive restarts.
#
# The two shipped draws (keys measured on the reference profile):
#   6affbca9eddbd34b — the NVFP4 target model (49 entries)
#   772fea630fdf214a — the NVFP4 DFlash2 drafter
# An unmatched key means this launch's configuration differs from the
# reference and that boot re-tunes from scratch — slower first boot, still
# correct.
#
# OWNERSHIP (the subtle part): this mod runs as root (docker exec --user
# root) but the serve command runs as the container's --user (rootless
# auto_user = host uid). Any root-owned file left in the shared runtime cache
# breaks serve — e.g. importing flashinfer as root makes its JIT logger
# create flashinfer_jit.log root-owned, and the non-root serve process then
# dies at import with PermissionError. So: never import heavy packages here
# (package metadata and nvidia-smi have no import side effects), and hand the
# whole cache to the serve user on every boot.
set -euo pipefail

MOD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CACHE_ROOT="${SGLANG_CACHE_DIR:-/cache/runtime/sglang}"
DEST_BASE="$CACHE_ROOT/flashinfer/autotune"
RUNTIME_CACHE_ROOT="${XDG_CACHE_HOME:-/cache/runtime}"
HF_HUB_ROOT="${HF_HUB_CACHE:-${HF_HOME:-/cache/huggingface}/hub}"

# Serve-user uid/gid = the user of PID 1 (sleep infinity runs as the
# container's --user; the serve docker exec inherits it).
SERVE_UID=$(stat -c %u /proc/1 2>/dev/null || echo 0)
SERVE_GID=$(stat -c %g /proc/1 2>/dev/null || echo 0)

fix_ownership() {
  if [ "$SERVE_UID" != "0" ]; then
    chown -R "$SERVE_UID:$SERVE_GID" "$RUNTIME_CACHE_ROOT" "$HF_HUB_ROOT" 2>/dev/null \
      || chmod -R a+rwX "$RUNTIME_CACHE_ROOT" "$HF_HUB_ROOT" 2>/dev/null \
      || echo "dflash2-tactics: WARNING: could not fix cache ownership; serve may fail to write caches" >&2
  fi
}
# Fix leftovers from previous boots first (a root-owned file here is what
# crashes serve at flashinfer import time), and again after installing.
fix_ownership

# --- Offline SHA resolution: write refs/main for the pinned snapshots ---
#
# sparkrun downloads checkpoints pinned by commit SHA, which materializes
# snapshots/<sha>/ but writes no refs/main. The serve container runs with
# HF_HUB_OFFLINE=1, and several revision-less lookups inside sglang (e.g. the
# speculative-algorithm alias resolver's get_config(draft_path) call, which
# predates revision threading) fall back to refs/main and die with
# "couldn't find ... in the cached files" when it's absent. Pangoleen's repo
# documents the same trap. Writing refs/main -> <pinned SHA> for both
# checkpoints makes every offline lookup resolve to the pinned snapshot
# (the same thing a branch download would have produced).
write_ref() {
  repo="$1"; sha="$2"
  d="$HF_HUB_ROOT/models--${repo//\//--}"
  if [ -d "$d/snapshots/$sha" ]; then
    mkdir -p "$d/refs"
    # NO trailing newline: huggingface_hub 1.28's try_to_load_from_cache reads
    # refs/<rev> with f.read() and does not strip — a trailing \n makes the
    # resolved snapshot dir name "sha\n" and every offline lookup misses.
    printf '%s' "$sha" > "$d/refs/main"
    echo "dflash2-tactics: refs/main -> $sha for $repo"
  else
    echo "dflash2-tactics: NOTE: no snapshot $sha for $repo in $HF_HUB_ROOT (distribution will download it; refs written on next boot)" >&2
  fi
}
write_ref RadixArk/Qwen3.8-27B-NVFP4 319f741cce68d7914884900c138a1fbb70a42f30
write_ref maurienne-ai/Qwen3.8-27B-DFlash2-NVFP4-RTNcal bd7a934213c47a9e7ef69eef36bb3325f47fd1f1

FI_VERSION=$(python3 - <<'PY' 2>/dev/null || echo unknown
import importlib.metadata as md
for name in ("flashinfer-python", "flashinfer_python", "flashinfer"):
    try:
        print(md.version(name))
        break
    except md.PackageNotFoundError:
        pass
else:
    print("unknown")
PY
)

# Arch WITHOUT importing torch, for the same reason. torch reports
# (major, minor) from get_device_capability() -> sm121 on GB10.
CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n1 | tr -d '[:space:]')
ARCH="sm$(echo "${CAP%%[a-z]*}" | tr -d '.')"

if [ "$FI_VERSION" = "unknown" ] || [ -z "$CAP" ]; then
  echo "dflash2-tactics: WARNING: cannot detect flashinfer/arch (fi=$FI_VERSION cap='$CAP') — draws not installed" >&2
  exit 0
fi
echo "dflash2-tactics: flashinfer=$FI_VERSION arch=$ARCH"

for draw in "$MOD_DIR"/draws/*.json; do
  key=$(basename "$draw" .json)
  dest="$DEST_BASE/$FI_VERSION/$ARCH/$key"
  mkdir -p "$dest"
  install -m 644 "$draw" "$dest/rank_tp0_pp0_dp0.json"
  echo "dflash2-tactics: installed draw $key -> $dest/rank_tp0_pp0_dp0.json"
done
fix_ownership

# The draws were captured under FlashInfer 0.6.18 (see _metadata in each
# file); this image ships 0.6.17. The cache *path* is version-keyed but the
# draw *key* is not, so the draws install and replay under this image's
# version directory — but kernel selection can differ between FI versions:
# verify the first boot against the reference band (~64-78 tok/s
# short-context decode, ~7-8 accepted tokens/pass of the 16 budget) before
# trusting the replayed draw.
if [ "$FI_VERSION" != "0.6.18" ]; then
  echo "dflash2-tactics: NOTE: draws measured on FlashInfer 0.6.18; this image reports $FI_VERSION — verify the replayed draw performs." >&2
fi

ls -l "$DEST_BASE/$FI_VERSION/$ARCH"/*/rank_tp0_pp0_dp0.json
