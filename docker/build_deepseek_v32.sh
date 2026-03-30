#!/bin/bash
# Build a vLLM image optimized for DeepSeek V3.2 on MI355X (gfx950).
#
# Uses a pre-built ROCm base image and builds vLLM from the current branch.
# Includes:
#   - RMSNorm+Quant fusion fix (matcher_utils.py + rocm_aiter_fusion.py)
#   - AITER mla_asm.csv fix for gfx950 gqa=32
#
# Usage:
#   ./docker/build_deepseek_v32.sh [--base-image IMAGE] [--tag TAG] [--pull] [--target STAGE]
#
# STAGE: final (default) or test — "test" includes dev deps + full tree for CI-style pytest
#        but builds much longer (RIXL/DeepEP, etc.).

set -euo pipefail

BASE_IMAGE="${BASE_IMAGE:-rocm/vllm-dev:base_custom_rocm_7.2.1_torch_triton_20260326_full_fix}"
TAG="${TAG:-vllm-deepseek-v32:latest}"
ROCM_ARCH="${ROCM_ARCH:-gfx942;gfx950}"
DOCKER_TARGET="${DOCKER_TARGET:-final}"
PULL_BASE=0

while [[ $# -gt 0 ]]; do
    case $1 in
        --base-image) BASE_IMAGE="$2"; shift 2;;
        --tag) TAG="$2"; shift 2;;
        --rocm-arch) ROCM_ARCH="$2"; shift 2;;
        --target) DOCKER_TARGET="$2"; shift 2;;
        --pull) PULL_BASE=1; shift;;
        -h|--help)
            grep '^#' "$0" | head -20
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "=== Building DeepSeek V3.2 vLLM image ==="
echo "  Base image:  $BASE_IMAGE"
echo "  Tag:         $TAG"
echo "  ROCM arch:   $ROCM_ARCH"
echo "  Docker target: $DOCKER_TARGET"
echo "  Source:      $REPO_ROOT"
echo ""

if [[ "$PULL_BASE" -eq 1 ]]; then
    echo "=== docker pull $BASE_IMAGE ==="
    docker pull "$BASE_IMAGE"
fi

docker build \
    -f "$REPO_ROOT/docker/Dockerfile.rocm" \
    --build-arg BASE_IMAGE="$BASE_IMAGE" \
    --build-arg ARG_PYTORCH_ROCM_ARCH="$ROCM_ARCH" \
    --target "$DOCKER_TARGET" \
    -t "$TAG" \
    "$REPO_ROOT"

echo ""
echo "=== Done. Run tests (Linux + ROCm Docker), e.g.: ==="
echo "  ./docker/run_deepseek_v32_tests.sh $TAG"
