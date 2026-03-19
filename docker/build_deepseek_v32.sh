#!/bin/bash
# Build a vLLM image optimized for DeepSeek V3.2 on MI355X (gfx950).
#
# Uses a pre-built ROCm base image and builds vLLM from the current branch.
# Includes:
#   - RMSNorm+Quant fusion fix (matcher_utils.py + rocm_aiter_fusion.py)
#   - AITER mla_asm.csv fix for gfx950 gqa=32
#
# Usage:
#   ./docker/build_deepseek_v32.sh [--base-image IMAGE] [--tag TAG]

set -euo pipefail

BASE_IMAGE="${BASE_IMAGE:-rocm/vllm-dev:base_torch2.10_triton3.6_rocm7.2_torch_build_20260216}"
TAG="${TAG:-vllm-deepseek-v32:latest}"
ROCM_ARCH="${ROCM_ARCH:-gfx942;gfx950}"

while [[ $# -gt 0 ]]; do
    case $1 in
        --base-image) BASE_IMAGE="$2"; shift 2;;
        --tag) TAG="$2"; shift 2;;
        --rocm-arch) ROCM_ARCH="$2"; shift 2;;
        *) echo "Unknown option: $1"; exit 1;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "=== Building DeepSeek V3.2 vLLM image ==="
echo "  Base image:  $BASE_IMAGE"
echo "  Tag:         $TAG"
echo "  ROCM arch:   $ROCM_ARCH"
echo "  Source:      $REPO_ROOT"
echo ""

docker build \
    -f "$REPO_ROOT/docker/Dockerfile.rocm" \
    --build-arg BASE_IMAGE="$BASE_IMAGE" \
    --build-arg ARG_PYTORCH_ROCM_ARCH="$ROCM_ARCH" \
    --target final \
    -t "$TAG" \
    "$REPO_ROOT"
