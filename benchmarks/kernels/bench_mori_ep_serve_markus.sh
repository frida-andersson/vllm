#!/bin/bash
# ========================================================================
# WRAPPER: bench_mori_ep_serve_markus.sh
# WRAPS: vllm serve + vllm bench serve
# PURPOSE: Benchmark MORI-EP AITER backend for DeepSeek R1 style workloads
# USAGE: ./scripts/bench_mori_ep_serve_markus.sh [OPTIONS]
# ARGUMENTS:
#   --model MODEL       Model name or path (default: deepseek-ai/DeepSeek-R1)
#   --isl N             Input sequence length (default: 70000)
#   --osl N             Output sequence length (default: 300)
#   --num-prompts N     Number of concurrent prompts (default: 10)
#   --tp N              Tensor parallel size (default: 8)
#   --backend BACKEND   All2All backend: mori_ep, deepep_high_throughput, etc.
#                       Use "none" or "tp_only" for basic TP without EP
#   --server-only       Only start the server, don't run benchmark
#   --bench-only        Only run benchmark (server must be running)
#   --port N            Server port (default: 8000)
#   --host HOST         Server host (default: localhost)
# EXAMPLES:
#   ./scripts/bench_mori_ep_serve_markus.sh --backend mori_ep
#   ./scripts/bench_mori_ep_serve_markus.sh --isl 70000 --osl 300 --num-prompts 10
#   ./scripts/bench_mori_ep_serve_markus.sh --server-only --backend mori_ep
#   ./scripts/bench_mori_ep_serve_markus.sh --bench-only --port 8000
# DEPENDENCIES: vllm, torch, mori (for mori_ep backend), aiter
# ========================================================================

set -e

# Default values matching DeepSeek R1 benchmark spec
MODEL="deepseek-ai/DeepSeek-R1"
ISL=70000          # Input Sequence Length
OSL=300            # Output Sequence Length
NUM_PROMPTS=10     # Concurrency
TP_SIZE=8          # Tensor Parallel (EP uses TP ranks for expert distribution)
BACKEND="mori_ep"  # All2All backend
HOST="localhost"
PORT=8000
SERVER_ONLY=false
BENCH_ONLY=false
MAX_MODEL_LEN=72000  # ISL + some buffer

# Enable AITER for ROCm
export VLLM_ROCM_USE_AITER=1
export VLLM_ROCM_USE_AITER_MOE=1

# MORI-EP requires a large symmetric heap for All-to-All communication buffers
# With max_num_batched_tokens=8192 and hidden_dim=7168, each rank needs ~940MB
# The heap is shared across all 8 GPUs, so total needed = 940MB * 8 = 7.5GB
# Set to 12GB to be safe with headroom for other allocations.
export MORI_SHMEM_HEAP_SIZE="${MORI_SHMEM_HEAP_SIZE:-12G}"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL="$2"
            shift 2
            ;;
        --isl)
            ISL="$2"
            # Auto-adjust max_model_len
            MAX_MODEL_LEN=$((ISL + 2000))
            shift 2
            ;;
        --osl)
            OSL="$2"
            shift 2
            ;;
        --num-prompts)
            NUM_PROMPTS="$2"
            shift 2
            ;;
        --tp)
            TP_SIZE="$2"
            shift 2
            ;;
        --backend)
            BACKEND="$2"
            shift 2
            ;;
        --host)
            HOST="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --server-only)
            SERVER_ONLY=true
            shift
            ;;
        --bench-only)
            BENCH_ONLY=true
            shift
            ;;
        --max-model-len)
            MAX_MODEL_LEN="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Benchmark MORI-EP AITER backend for DeepSeek R1 style workloads."
            echo ""
            echo "Options:"
            echo "  --model MODEL       Model name/path (default: deepseek-ai/DeepSeek-R1)"
            echo "  --isl N             Input sequence length (default: 70000)"
            echo "  --osl N             Output sequence length (default: 300)"
            echo "  --num-prompts N     Concurrent prompts (default: 10)"
            echo "  --tp N              Tensor parallel size (default: 8)"
            echo "  --backend BACKEND   All2All backend (default: mori_ep)"
            echo "                      Options: mori_ep, deepep_high_throughput,"
            echo "                               deepep_low_latency, pplx, naive,"
            echo "                               none/tp_only (basic TP, no EP)"
            echo "  --server-only       Only start server"
            echo "  --bench-only        Only run benchmark (server must be running)"
            echo "  --host HOST         Server host (default: localhost)"
            echo "  --port N            Server port (default: 8000)"
            echo ""
            echo "Examples:"
            echo "  # Full benchmark with MORI-EP"
            echo "  $0 --backend mori_ep"
            echo ""
            echo "  # Compare backends"
            echo "  $0 --backend mori_ep && $0 --bench-only  # baseline with MORI"
            echo "  # restart server with different backend, then:"
            echo "  $0 --backend deepep_high_throughput"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use -h or --help for usage information"
            exit 1
            ;;
    esac
done

echo "=========================================="
echo "MORI-EP AITER Serving Benchmark"
echo "=========================================="
echo "Model:        $MODEL"
echo "ISL:          $ISL"
echo "OSL:          $OSL"
echo "Concurrency:  $NUM_PROMPTS"
echo "TP Size:      $TP_SIZE"
echo "Backend:      $BACKEND"
echo "Host:         $HOST:$PORT"
echo "Max Model Len: $MAX_MODEL_LEN"
echo "=========================================="

cd "$(dirname "$0")/.."

# Function to start the server
start_server() {
    echo ""
    echo "Starting vLLM server with $BACKEND backend..."
    echo ""
    
    # Build server command as array (handles quoting properly)
    SERVER_ARGS=(
        "vllm" "serve" "$MODEL"
        "--host" "$HOST"
        "--port" "$PORT"
        "--tensor-parallel-size" "$TP_SIZE"
        "--max-model-len" "$MAX_MODEL_LEN"
        "--trust-remote-code"
    )
    
    # Add expert parallel flags unless using basic TP
    if [[ "$BACKEND" != "none" && "$BACKEND" != "tp_only" ]]; then
        SERVER_ARGS+=("--enable-expert-parallel" "--all2all-backend" "$BACKEND")
        echo "Mode: Expert Parallel (EP) with $BACKEND"
        
        # MORI-EP dispatch-free: CUDAGraph-compatible (no dynamic shapes).
        # No special compilation flags needed.
    else
        echo "Mode: Basic Tensor Parallel (TP$TP_SIZE, no EP)"
    fi
    
    # For DeepSeek R1, use FP8 KV cache for memory efficiency
    SERVER_ARGS+=("--kv-cache-dtype" "fp8")
    
    echo "Command: ${SERVER_ARGS[*]}"
    echo ""
    
    # Run server
    exec "${SERVER_ARGS[@]}"
}

# Function to run benchmark
run_benchmark() {
    echo ""
    echo "Running vllm bench serve..."
    echo ""
    
    # Build benchmark command
    BENCH_CMD="vllm bench serve"
    BENCH_CMD="$BENCH_CMD --model $MODEL"
    BENCH_CMD="$BENCH_CMD --host $HOST"
    BENCH_CMD="$BENCH_CMD --port $PORT"
    BENCH_CMD="$BENCH_CMD --dataset-name random"
    BENCH_CMD="$BENCH_CMD --input-len $ISL"
    BENCH_CMD="$BENCH_CMD --output-len $OSL"
    BENCH_CMD="$BENCH_CMD --num-prompts $NUM_PROMPTS"
    BENCH_CMD="$BENCH_CMD --request-rate inf"  # Send all at once for concurrency test
    
    echo "Command: $BENCH_CMD"
    echo ""
    
    # Run benchmark
    $BENCH_CMD
}

# Main logic
if $BENCH_ONLY; then
    run_benchmark
elif $SERVER_ONLY; then
    start_server
else
    echo ""
    echo "NOTE: This script starts the server in foreground."
    echo "For full benchmark, use two terminals:"
    echo "  Terminal 1: $0 --server-only --backend $BACKEND"
    echo "  Terminal 2: $0 --bench-only"
    echo ""
    echo "Or run server in background:"
    echo "  $0 --server-only --backend $BACKEND &"
    echo "  sleep 60  # wait for server to start"
    echo "  $0 --bench-only"
    echo ""
    
    # Start server (blocking)
    start_server
fi

