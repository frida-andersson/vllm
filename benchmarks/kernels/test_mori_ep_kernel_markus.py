#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Benchmark: dispatch-free MORI EP (TP8+EP8) vs AITER-only (TP8).

In TP8+EP8 mode every GPU already holds the full token batch after
TP all-reduce.  The dispatch-free optimisation therefore:

  1. Filters to tokens that activate any local expert   (cheap index_select)
  2. Remaps global expert IDs → local IDs               (arithmetic)
  3. Runs AITER fused-expert compute on the subset       (GPU kernel)
  4. Scatters results back into a full-size buffer       (index_copy)
  5. All-reduces across EP group                         (NCCL)

Sub-commands
============

kernel   Micro-benchmark of the MoE kernel (dispatch-free vs AITER-only).
serve    End-to-end serving benchmark via ``vllm serve`` + ``vllm bench serve``.

Kernel mode
-----------
  # dispatch-free EP8  (8 processes, one per GPU)
  torchrun --nproc_per_node=8 benchmarks/kernels/test_mori_ep_kernel_markus.py kernel

  # AITER-only baseline (single GPU, all 256 experts local)
  python benchmarks/kernels/test_mori_ep_kernel_markus.py kernel --no-ep

Serve mode
----------
  # TP8 baseline
  python benchmarks/kernels/test_mori_ep_kernel_markus.py serve --mode tp

  # TP8+EP8 dispatch-free
  python benchmarks/kernels/test_mori_ep_kernel_markus.py serve --mode ep

  # Both (sequential: start TP8, bench, stop, start EP8, bench, stop)
  python benchmarks/kernels/test_mori_ep_kernel_markus.py serve --mode both
"""

import argparse
import gc
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass

os.environ.setdefault("VLLM_ROCM_USE_AITER", "1")
os.environ.setdefault("VLLM_ROCM_USE_AITER_MOE", "1")

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)


# ── dataclass ───────────────────────────────────────────────────────
@dataclass
class BenchmarkResult:
    label: str
    num_tokens: int
    hidden_size: int
    num_experts: int
    topk: int
    ep_size: int
    total_us: float
    prepare_us: float | None = None
    compute_us: float | None = None
    allreduce_us: float | None = None


# ── helpers ─────────────────────────────────────────────────────────
def _create_data(
    num_tokens: int,
    hidden_size: int,
    num_experts: int,
    topk: int,
    num_local_experts: int,
    intermediate_size: int = 2048,
    dtype: torch.dtype = torch.bfloat16,
):
    """Produce input activations, routing decisions, and expert weights."""
    x = torch.randn(num_tokens, hidden_size, dtype=dtype, device="cuda")
    gating = torch.randn(num_tokens, num_experts, dtype=torch.float32, device="cuda")
    topk_weights, topk_ids = torch.topk(gating, topk, dim=-1)
    topk_weights = torch.softmax(topk_weights, dim=-1).to(torch.float32)
    topk_ids = topk_ids.to(torch.int32)

    w1 = torch.randn(
        num_local_experts, 2 * intermediate_size, hidden_size, dtype=dtype, device="cuda"
    )
    w2 = torch.randn(
        num_local_experts, hidden_size, intermediate_size, dtype=dtype, device="cuda"
    )
    return x, topk_weights, topk_ids, w1, w2


def _timer(num_iters, warmup, fn):
    """Run *fn* for warmup + measured iterations and return per-iter µs."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(num_iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return (start.elapsed_time(end) / num_iters) * 1000  # → µs


# ── dispatch-free benchmark ────────────────────────────────────────
def bench_dispatch_free(
    x: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    rank: int,
    num_local_experts: int,
    ep_group: torch.distributed.ProcessGroup,
    num_iters: int,
    warmup: int,
    num_experts: int = 256,
) -> tuple[float, float, float, float]:
    """
    Benchmark the dispatch-free TP+EP pipeline.

    The prepare phase is a pure pass-through (no filtering, no ID
    remapping).  AITER's ``expert_mask`` skips GEMMs for non-local
    experts, so each GPU only computes the ~1 local expert per token.

    Returns (total_us, prepare_us, compute_us, allreduce_us).
    """
    from vllm.model_executor.layers.fused_moe.rocm_aiter_fused_moe import (
        rocm_aiter_fused_experts,
    )

    # Build expert_mask: 1 for local experts, 0 for others + sentinel
    rank_offset = rank * num_local_experts
    expert_mask = torch.zeros(num_experts + 1, dtype=torch.int32, device=x.device)
    expert_mask[rank_offset : rank_offset + num_local_experts] = 1

    # ── Step 1: prepare (pass-through, measure overhead) ──────
    def _prepare():
        # In vllm serve mode this is a no-op; measure the baseline
        return x, topk_weights, topk_ids

    prepare_us = _timer(num_iters, warmup, _prepare)

    # ── Step 2: AITER compute with expert_mask ────────────────
    def _compute():
        return rocm_aiter_fused_experts(
            x, w1, w2, topk_weights, topk_ids,
            expert_map=expert_mask,
        )

    compute_us = _timer(num_iters, warmup, _compute)
    expert_out = _compute()

    # ── Step 3: all-reduce ────────────────────────────────────
    def _allreduce():
        buf = expert_out.clone()
        torch.distributed.all_reduce(buf, op=torch.distributed.ReduceOp.SUM, group=ep_group)
        return buf

    allreduce_us = _timer(num_iters, warmup, _allreduce)

    total_us = prepare_us + compute_us + allreduce_us
    return total_us, prepare_us, compute_us, allreduce_us


# ── AITER-only baseline ────────────────────────────────────────────
def bench_aiter_only(
    x: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    num_iters: int,
    warmup: int,
) -> float:
    """Benchmark AITER fused MoE with all experts local (TP8 proxy)."""
    from vllm.model_executor.layers.fused_moe.rocm_aiter_fused_moe import (
        rocm_aiter_fused_experts,
    )

    def _run():
        return rocm_aiter_fused_experts(x, w1, w2, topk_weights, topk_ids)

    return _timer(num_iters, warmup, _run)


# ── printing ────────────────────────────────────────────────────────
def _print_results(results: list[BenchmarkResult]):
    print()
    print("=" * 115)
    print("Dispatch-Free MORI EP  Benchmark Results")
    print("=" * 115)
    hdr = (
        f"{'Label':<30} {'Tokens':<8} {'EP':<4} "
        f"{'Total (µs)':<14} {'Prepare (µs)':<14} "
        f"{'Compute (µs)':<14} {'AR (µs)':<14} "
        f"{'Tput (tok/s)':<16}"
    )
    print(hdr)
    print("-" * 115)
    for r in results:
        f = f"{r.prepare_us:.1f}" if r.prepare_us is not None else "N/A"
        c = f"{r.compute_us:.1f}" if r.compute_us is not None else "N/A"
        s = f"{r.allreduce_us:.1f}" if r.allreduce_us is not None else "N/A"
        tput = r.num_tokens / r.total_us * 1e6
        print(
            f"{r.label:<30} {r.num_tokens:<8} {r.ep_size:<4} "
            f"{r.total_us:<14.1f} {f:<14} {c:<14} {s:<14} {tput:<16.0f}"
        )
    print("=" * 115)


# ── serve-mode helpers ──────────────────────────────────────────────
def _wait_for_server(port: int, timeout: int = 600) -> bool:
    """Poll server health endpoint until ready or timeout."""
    import urllib.request
    url = f"http://localhost:{port}/health"
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with urllib.request.urlopen(url, timeout=2) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(5)
    return False


_SERVER_LOG = "/tmp/vllm_bench_server.log"


def _start_server(
    model: str,
    tp: int,
    port: int,
    max_model_len: int,
    enable_ep: bool,
) -> subprocess.Popen:
    """Launch ``vllm serve`` as a background process and return the Popen."""
    env = os.environ.copy()
    env["VLLM_ROCM_USE_AITER"] = "1"
    env["VLLM_ROCM_USE_AITER_MOE"] = "1"
    env["VLLM_RPC_TIMEOUT"] = "1800000"

    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model,
        "--tensor-parallel-size", str(tp),
        "--port", str(port),
        "--max-model-len", str(max_model_len),
        "--trust-remote-code",
        "--disable-log-requests",
    ]
    if enable_ep:
        cmd += ["--enable-expert-parallel", "--all2all-backend", "mori"]

    label = "TP+EP (dispatch-free)" if enable_ep else "TP-only"
    print(f"\n>>> Starting vllm serve [{label}]")
    print(f">>> Command: {' '.join(cmd)}")
    print(f">>> Server log: {_SERVER_LOG}\n", flush=True)

    log_fh = open(_SERVER_LOG, "w")
    proc = subprocess.Popen(
        cmd, env=env, stdout=log_fh, stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    proc._log_fh = log_fh  # type: ignore[attr-defined]
    return proc


def _stop_server(proc: subprocess.Popen) -> None:
    """Gracefully stop a server process and all its children."""
    # Close log file handle if we attached one
    log_fh = getattr(proc, "_log_fh", None)
    if log_fh:
        try:
            log_fh.close()
        except Exception:
            pass

    if proc.poll() is not None:
        return
    # Kill the whole process group
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except (ProcessLookupError, OSError):
        pass
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, OSError):
            pass
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            pass
    # Also clean up any stray workers
    subprocess.run(
        "pkill -9 -f 'VLLM::Worker|VLLM::Engine' 2>/dev/null; sleep 3",
        shell=True, capture_output=True,
    )


def _run_bench(model: str, port: int, isl: int, osl: int, num_prompts: int,
               request_rate: float) -> str:
    """Run ``vllm bench serve`` and return stdout."""
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.cli.main",
        "bench", "serve",
        "--base-url", f"http://localhost:{port}",
        "--model", model,
        "--dataset-name", "random",
        "--random-input-len", str(isl),
        "--random-output-len", str(osl),
        "--num-prompts", str(num_prompts),
        "--request-rate", str(request_rate),
    ]
    print(f">>> Benchmark: {' '.join(cmd)}\n")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=1200)
    output = result.stdout + result.stderr
    print(output)
    return output


def cmd_serve(args: argparse.Namespace) -> None:
    """Sub-command: end-to-end serving benchmark."""
    modes = []
    if args.mode in ("tp", "both"):
        modes.append(("TP8 baseline", False))
    if args.mode in ("ep", "both"):
        modes.append(("TP8+EP8 dispatch-free", True))

    for label, enable_ep in modes:
        print(f"\n{'='*60}")
        print(f" {label}")
        print(f"{'='*60}")

        proc = _start_server(
            model=args.model,
            tp=args.tp,
            port=args.port,
            max_model_len=args.max_model_len,
            enable_ep=enable_ep,
        )
        try:
            print(">>> Waiting for server to become ready ...")
            if not _wait_for_server(args.port, timeout=2400):
                print(">>> ERROR: server did not become ready in 2400s")
                print(f">>> Last 80 lines of server log ({_SERVER_LOG}):")
                try:
                    with open(_SERVER_LOG) as f:
                        lines = f.readlines()
                        print("".join(lines[-80:]))
                except Exception:
                    pass
                continue

            print(">>> Server is ready. Running benchmark ...\n")
            _run_bench(
                model=args.model,
                port=args.port,
                isl=args.isl,
                osl=args.osl,
                num_prompts=args.num_prompts,
                request_rate=args.request_rate,
            )
        finally:
            print(f"\n>>> Stopping server [{label}] ...")
            _stop_server(proc)
            # Wait for GPU memory to be freed
            time.sleep(10)


# ── kernel sub-command ──────────────────────────────────────────────
def cmd_kernel(args: argparse.Namespace) -> None:
    """Sub-command: micro-benchmark of MoE kernel."""
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)

    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    is_rank0 = rank == 0

    if world > 1 and not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="nccl")

    results: list[BenchmarkResult] = []

    for num_tokens in args.num_tokens:
        torch.manual_seed(42)
        if args.no_ep or world == 1:
            # ── AITER-only (all 256 experts local) ──────────────
            num_local = args.num_experts // args.ep_size
            x, tw, ti, w1, w2 = _create_data(
                num_tokens, args.hidden_size, args.num_experts,
                args.topk, num_local,
            )
            lat = bench_aiter_only(x, tw, ti, w1, w2, args.num_iters, args.warmup)
            results.append(BenchmarkResult(
                label="aiter (TP8 proxy)", num_tokens=num_tokens,
                hidden_size=args.hidden_size, num_experts=args.num_experts,
                topk=args.topk, ep_size=1, total_us=lat,
            ))
            del x, tw, ti, w1, w2
        else:
            # ── dispatch-free EP8 ───────────────────────────────
            ep_group = torch.distributed.new_group(backend="nccl")
            num_local = args.num_experts // world
            x, tw, ti, w1, w2 = _create_data(
                num_tokens, args.hidden_size, args.num_experts,
                args.topk, num_local,
            )
            total, prep, comp, ar = bench_dispatch_free(
                x, tw, ti, w1, w2,
                rank=rank, num_local_experts=num_local, ep_group=ep_group,
                num_iters=args.num_iters, warmup=args.warmup,
                num_experts=args.num_experts,
            )
            results.append(BenchmarkResult(
                label="dispatch-free EP8", num_tokens=num_tokens,
                hidden_size=args.hidden_size, num_experts=args.num_experts,
                topk=args.topk, ep_size=world,
                total_us=total, prepare_us=prep,
                compute_us=comp, allreduce_us=ar,
            ))
            del x, tw, ti, w1, w2

        gc.collect()
        torch.cuda.empty_cache()

    if is_rank0:
        _print_results(results)

    if torch.distributed.is_initialized():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


# ── main ────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Dispatch-free MORI EP benchmark (kernel + serve modes)",
    )
    sub = parser.add_subparsers(dest="command")

    # ── kernel sub-command ──────────────────────────────────────
    p_kernel = sub.add_parser("kernel", help="Micro-benchmark MoE kernel")
    p_kernel.add_argument("--num-tokens", type=int, nargs="+", default=[128, 512, 1024, 4096])
    p_kernel.add_argument("--hidden-size", type=int, default=7168)
    p_kernel.add_argument("--num-experts", type=int, default=256)
    p_kernel.add_argument("--topk", type=int, default=8)
    p_kernel.add_argument("--ep-size", type=int, default=8)
    p_kernel.add_argument("--num-iters", type=int, default=100)
    p_kernel.add_argument("--warmup", type=int, default=20)
    p_kernel.add_argument("--no-ep", action="store_true",
                          help="Run AITER-only baseline (single GPU).")

    # ── serve sub-command ──────────────────────────────────────
    p_serve = sub.add_parser("serve", help="E2E serving benchmark via vllm serve")
    p_serve.add_argument("--model", type=str, default="deepseek-ai/DeepSeek-R1")
    p_serve.add_argument("--tp", type=int, default=8)
    p_serve.add_argument("--port", type=int, default=8100)
    p_serve.add_argument("--max-model-len", type=int, default=4096)
    p_serve.add_argument("--isl", type=int, default=512, help="Input sequence length")
    p_serve.add_argument("--osl", type=int, default=128, help="Output sequence length")
    p_serve.add_argument("--num-prompts", type=int, default=200)
    p_serve.add_argument("--request-rate", type=float, default=10.0)
    p_serve.add_argument("--mode", choices=["tp", "ep", "both"], default="both",
                         help="tp=baseline, ep=dispatch-free, both=run sequentially")

    args = parser.parse_args()

    if args.command == "kernel":
        cmd_kernel(args)
    elif args.command == "serve":
        cmd_serve(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
