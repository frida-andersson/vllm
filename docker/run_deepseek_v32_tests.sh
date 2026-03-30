#!/bin/bash
# Run ROCm-relevant tests inside an existing DeepSeek V3.2 vLLM image.
# Mounts the repo so pytest uses current sources (including local patches).
#
# Requires: Docker on Linux with ROCm (/dev/kfd, /dev/dri) or equivalent.
#
# Usage:
#   ./docker/run_deepseek_v32_tests.sh [IMAGE] [PYTEST_ARGS...]
#
# Example:
#   ./docker/run_deepseek_v32_tests.sh vllm-deepseek-v32:latest \
#     tests/kernels/attention/test_deepgemm_attention.py -v --tb=short

set -euo pipefail

IMAGE="${1:-vllm-deepseek-v32:latest}"
shift || true

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

if [[ $# -eq 0 ]]; then
  set -- tests/kernels/attention/test_deepgemm_attention.py -v --tb=short
fi

echo "=== Running tests in $IMAGE ==="
echo "  Repo: $REPO_ROOT"
echo "  Args: $*"
echo ""

EXTRA_DEVICES=()
if [[ -e /dev/kfd ]]; then EXTRA_DEVICES+=(--device /dev/kfd); fi
if [[ -e /dev/dri ]]; then EXTRA_DEVICES+=(--device /dev/dri); fi

docker run --rm -i \
  "${EXTRA_DEVICES[@]}" \
  --group-add video \
  -e HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}" \
  -v "$REPO_ROOT:/src/vllm" \
  -w /src/vllm \
  "$IMAGE" \
  bash -c 'set -euo pipefail
    export AITER_ENABLE_AOT_GLUON_PA_MQA_LOGITS=1
    export VLLM_ROCM_USE_AITER=1
    python3 -m pip install -q -e .
    exec python3 -m pytest "$@"
  ' bash "$@"
