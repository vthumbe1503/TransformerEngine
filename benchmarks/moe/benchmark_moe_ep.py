# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""BF16 MoE-block benchmark: Dispatch, GroupedMLP (FC1+SwiGLU+FC2), Combine.

Designed so GroupedMLP work is constant from 1 node (4 GPUs) to 8 nodes (32 GPUs):

    gpus_per_node = 4
    num_local_experts = 8
    topk = num_nodes
    tokens_in_per_gpu = target_recv / num_nodes     # 32768 / num_nodes
    tokens_after_dispatch ≈ target_recv             # 32768 (GroupedMLP)

1-node:  32 experts, topk=1,  32768 tokens in,  32768 recv (4-GPU A2A)
8-node: 256 experts, topk=8,   4096 tokens in,  32768 recv (32-GPU A2A)

Routing is a round-robin over expert ids so every local expert gets the same
m_splits. Launch with torchrun (1 node) or srun (multi-node); see run_moe_ep.sh.

    torchrun --nproc_per_node=4 benchmarks/moe/benchmark_moe_ep.py
    torchrun --nproc_per_node=4 benchmarks/moe/benchmark_moe_ep.py --profile
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass

import torch
import torch.distributed as dist

from transformer_engine.pytorch import ops as te_ops
from transformer_engine.pytorch.ep import (
    EpBuffer,
    ep_bootstrap,
    ep_combine,
    ep_dispatch,
    ep_finalize,
    release_symm_mem_pool,
)

# Must come after the transformer_engine import so libtransformer_engine.so is loaded.
import transformer_engine_torch as tex  # noqa: F401

# DeepSeek-V3 MoE expert dims (SwiGLU: FC1 out = 2 * intermediate).
DEFAULT_HIDDEN = 7168
DEFAULT_INTERMEDIATE = 2048
DEFAULT_LOCAL_EXPERTS = 8
DEFAULT_GPUS_PER_NODE = 4
DEFAULT_TARGET_RECV = 32768


@dataclass
class BenchConfig:
    rank: int
    world_size: int
    local_rank: int
    gpus_per_node: int
    num_nodes: int
    ep_size: int
    num_local_experts: int
    num_experts: int
    topk: int
    tokens_per_rank: int
    target_recv: int
    hidden: int
    intermediate: int
    recv_capacity: int
    device: torch.device
    warmup: int
    iters: int
    profile: bool
    mode: str


def _env_int(name: str, default: int | None = None) -> int | None:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return int(value)


def init_distributed() -> tuple[int, int, int]:
    """Init NCCL from torchrun or Slurm env vars. Returns rank, world_size, local_rank."""
    if "SLURM_PROCID" in os.environ and "RANK" not in os.environ:
        os.environ["RANK"] = os.environ["SLURM_PROCID"]
        os.environ["WORLD_SIZE"] = os.environ["SLURM_NTASKS"]
        os.environ["LOCAL_RANK"] = os.environ.get(
            "SLURM_LOCALID", os.environ.get("SLURM_PROCID", "0")
        )
        if "MASTER_ADDR" not in os.environ:
            node_list = os.environ.get("SLURM_NODELIST") or os.environ.get("SLURM_JOB_NODELIST")
            if node_list:
                import subprocess

                os.environ["MASTER_ADDR"] = (
                    subprocess.check_output(["scontrol", "show", "hostnames", node_list])
                    .decode()
                    .splitlines()[0]
                )
            else:
                os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ.setdefault("MASTER_PORT", "29500")

    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    try:
        from torch.distributed import _symmetric_memory as _symm_mem

        _symm_mem.set_backend("NCCL")
    except (ImportError, RuntimeError):
        pass
    return dist.get_rank(), dist.get_world_size(), local_rank


def build_config(args: argparse.Namespace, rank: int, world_size: int, local_rank: int) -> BenchConfig:
    gpus_per_node = args.gpus_per_node
    if world_size % gpus_per_node != 0:
        raise ValueError(
            f"world_size={world_size} is not divisible by gpus_per_node={gpus_per_node}"
        )
    num_nodes = world_size // gpus_per_node
    if args.target_recv % num_nodes != 0:
        raise ValueError(
            f"target_recv={args.target_recv} must be divisible by num_nodes={num_nodes} "
            "so tokens_in * topk stays equal to target_recv"
        )
    tokens_per_rank = args.target_recv // num_nodes
    topk = num_nodes
    num_experts = args.num_local_experts * world_size
    # Exact uniform recv is target_recv; keep a little slack for EP padding.
    recv_capacity = max(args.target_recv, tokens_per_rank) + args.num_local_experts
    return BenchConfig(
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
        gpus_per_node=gpus_per_node,
        num_nodes=num_nodes,
        ep_size=world_size,
        num_local_experts=args.num_local_experts,
        num_experts=num_experts,
        topk=topk,
        tokens_per_rank=tokens_per_rank,
        target_recv=args.target_recv,
        hidden=args.hidden,
        intermediate=args.intermediate,
        recv_capacity=recv_capacity,
        device=torch.device("cuda", local_rank),
        warmup=args.warmup,
        iters=args.iters,
        profile=args.profile,
        mode=args.mode,
    )


def make_uniform_routing(
    rank: int,
    tokens_per_rank: int,
    topk: int,
    num_experts: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Round-robin expert ids so every expert receives the same token count.

    Slot ``s`` on this rank maps to expert ``(rank * T * K + s) % E``. Per-token
    top-k ids are ``K`` consecutive experts, so they do not collide (K << E).
    Weights are uniform ``1/K``; destination is entirely from ``topk_idx``.
    """
    slots = torch.arange(tokens_per_rank * topk, device=device, dtype=torch.int64)
    base = rank * tokens_per_rank * topk
    topk_idx = ((base + slots) % num_experts).view(tokens_per_rank, topk)
    topk_weights = torch.full(
        (tokens_per_rank, topk),
        1.0 / topk,
        dtype=torch.float32,
        device=device,
    )
    return topk_idx, topk_weights


def _build_ep_group() -> dist.ProcessGroup:
    world_pg = dist.distributed_c10d._get_default_group()
    return dist.new_group(ranks=list(range(world_pg.size())), backend="nccl")


def make_mlp(cfg: BenchConfig) -> te_ops.Sequential:
    fc1 = te_ops.GroupedLinear(
        cfg.num_local_experts,
        cfg.hidden,
        2 * cfg.intermediate,
        bias=False,
        device=cfg.device,
        dtype=torch.bfloat16,
    )
    activation = te_ops.ScaledSwiGLU()
    fc2 = te_ops.GroupedLinear(
        cfg.num_local_experts,
        cfg.intermediate,
        cfg.hidden,
        bias=False,
        device=cfg.device,
        dtype=torch.bfloat16,
    )
    return te_ops.Sequential(fc1, activation, fc2)


def make_moe_block(cfg: BenchConfig, buffer: EpBuffer) -> te_ops.Sequential:
    """Unfused Dispatch/Combine so m_splits is returned; FC1+SwiGLU+FC2 can still fuse."""
    dispatch = te_ops.Dispatch(buffer)
    fc1 = te_ops.GroupedLinear(
        cfg.num_local_experts,
        cfg.hidden,
        2 * cfg.intermediate,
        bias=False,
        device=cfg.device,
        dtype=torch.bfloat16,
    )
    activation = te_ops.ScaledSwiGLU()
    fc2 = te_ops.GroupedLinear(
        cfg.num_local_experts,
        cfg.intermediate,
        cfg.hidden,
        bias=False,
        device=cfg.device,
        dtype=torch.bfloat16,
    )
    combine = te_ops.Combine(buffer, num_local_tokens=cfg.tokens_per_rank)
    dispatch.set_extra_output_channel(0, "tokens_per_expert", output_to_caller=True)
    dispatch.set_extra_output_channel(1, "routing_weights", output_to_caller=True)
    fc1.set_extra_input_channel(0, "tokens_per_expert")
    activation.set_extra_input_channel(0, "routing_weights")
    fc2.set_extra_input_channel(0, "tokens_per_expert")
    return te_ops.Sequential(dispatch, fc1, activation, fc2, combine)


def _mlp_fwd_flops(cfg: BenchConfig, recv_tokens: int) -> int:
    # FC1: M x H x 2I, FC2: M x I x H. Ignore SwiGLU elementwise.
    return 2 * recv_tokens * cfg.hidden * cfg.intermediate * (2 + 1)


@torch.no_grad()
def _zero_mlp_grads(mlp: torch.nn.Module) -> None:
    for param in mlp.parameters():
        if param.grad is not None:
            param.grad.zero_()


def time_cuda(fn, warmup: int, iters: int, *, sync_group: dist.ProcessGroup) -> float:
    """Average GPU time in milliseconds. Barriers keep ranks aligned."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    dist.barrier(group=sync_group)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    end.synchronize()
    dist.barrier(group=sync_group)
    return start.elapsed_time(end) / iters


def _allgather_min_max_mean(value: float, group: dist.ProcessGroup) -> tuple[float, float, float]:
    tensor = torch.tensor([value], dtype=torch.float64, device="cuda")
    gathered = [torch.zeros_like(tensor) for _ in range(dist.get_world_size(group))]
    dist.all_gather(gathered, tensor, group=group)
    host = torch.stack(gathered).cpu()
    return float(host.min()), float(host.max()), float(host.mean())


def _log(rank: int, message: str) -> None:
    if rank == 0:
        print(message, flush=True)


def print_m_splits(cfg: BenchConfig, m_splits: torch.Tensor, group: dist.ProcessGroup) -> None:
    """Print every rank's m_splits and whether the split is uniform."""
    local = m_splits.detach().to(dtype=torch.int64, device="cpu").tolist()
    packed = torch.tensor(local, dtype=torch.int64, device=cfg.device)
    gathered = [torch.empty_like(packed) for _ in range(cfg.world_size)]
    dist.all_gather(gathered, packed, group=group)
    if cfg.rank != 0:
        return
    print("\n=== m_splits (tokens per local expert, after dispatch) ===", flush=True)
    expected = cfg.target_recv // cfg.num_local_experts
    all_uniform = True
    for src, row in enumerate(gathered):
        values = row.tolist()
        total = sum(values)
        uniform = all(v == expected for v in values) and total == cfg.target_recv
        all_uniform = all_uniform and uniform
        flag = "uniform" if uniform else "NOT uniform"
        print(f"  rank {src:3d}: {values}  sum={total}  [{flag}]", flush=True)
    print(
        f"  expected per expert={expected}  expected sum={cfg.target_recv}  "
        f"all_ranks_uniform={all_uniform}",
        flush=True,
    )


def run_benchmark(cfg: BenchConfig) -> None:
    if cfg.ep_size < 2:
        raise RuntimeError("NCCL EP needs at least 2 GPUs")

    ep_group = _build_ep_group()
    ep_bootstrap(
        ep_group,
        num_experts=cfg.num_experts,
        max_tokens_per_rank=cfg.tokens_per_rank,
        hidden_dim=cfg.hidden,
        num_topk=cfg.topk,
        recv_capacity_per_rank=cfg.recv_capacity,
    )
    buffer = EpBuffer(
        top_k=cfg.topk,
        max_tokens_per_rank=cfg.tokens_per_rank,
        hidden_dim=cfg.hidden,
        num_local_experts=cfg.num_local_experts,
        recv_capacity_per_rank=cfg.recv_capacity,
    )

    topk_idx, topk_weights = make_uniform_routing(
        cfg.rank, cfg.tokens_per_rank, cfg.topk, cfg.num_experts, cfg.device
    )
    tokens = torch.randn(
        cfg.tokens_per_rank, cfg.hidden, device=cfg.device, dtype=torch.bfloat16
    )
    grad_out = torch.randn_like(tokens)

    mlp = make_mlp(cfg)
    moe = make_moe_block(cfg, buffer)

    # One unfused e2e step so we can print m_splits before timing.
    with torch.no_grad():
        _out, m_splits, _recv_w = moe(tokens, topk_idx, topk_weights)
    print_m_splits(cfg, m_splits, ep_group)
    recv_rows = int(m_splits.sum().item())

    _log(cfg.rank, "\n=== config ===")
    _log(
        cfg.rank,
        f"  nodes={cfg.num_nodes}  gpus_per_node={cfg.gpus_per_node}  "
        f"world={cfg.world_size}  EP={cfg.ep_size}",
    )
    _log(
        cfg.rank,
        f"  num_experts={cfg.num_experts}  local_experts={cfg.num_local_experts}  "
        f"topk={cfg.topk}",
    )
    _log(
        cfg.rank,
        f"  hidden={cfg.hidden}  intermediate={cfg.intermediate}  "
        f"fc1_out={2 * cfg.intermediate}",
    )
    _log(
        cfg.rank,
        f"  tokens_in/gpu={cfg.tokens_per_rank}  tokens_recv/gpu={recv_rows}  "
        f"target_recv={cfg.target_recv}",
    )
    _log(cfg.rank, f"  mode={cfg.mode}  warmup={cfg.warmup}  iters={cfg.iters}")

    # Persistent recv views for isolated-stage timing (same routing every iter).
    recv_tokens, recv_weights, splits = ep_dispatch(buffer, tokens, topk_idx, topk_weights)
    # Warm the MLP once so first-iter autotune is outside the timed region.
    with torch.no_grad():
        expert_out = mlp(recv_tokens, splits, recv_weights, splits)
    expert_for_combine = expert_out.detach().requires_grad_(True)
    mlp_dy = torch.randn_like(expert_out)

    def dispatch_fwd():
        with torch.cuda.nvtx.range("dispatch_fwd"):
            return ep_dispatch(buffer, tokens, topk_idx, topk_weights)

    def mlp_fwd():
        with torch.cuda.nvtx.range("grouped_mlp_fwd"):
            return mlp(recv_tokens, splits, recv_weights, splits)

    def combine_fwd():
        with torch.cuda.nvtx.range("combine_fwd"):
            return ep_combine(buffer, expert_out, num_local_tokens=cfg.tokens_per_rank)

    def e2e_fwd():
        with torch.cuda.nvtx.range("moe_e2e_fwd"):
            return moe(tokens, topk_idx, topk_weights)

    def e2e_fwd_bwd():
        moe.zero_grad(set_to_none=False)
        tokens_p = tokens.detach().requires_grad_(True)
        weights_p = topk_weights.detach().requires_grad_(True)
        with torch.cuda.nvtx.range("moe_e2e_fwd_bwd"):
            out, _splits_i, _rw = moe(tokens_p, topk_idx, weights_p)
            out.backward(grad_out)

    def dispatch_fwd_bwd():
        tokens_p = tokens.detach().requires_grad_(True)
        weights_p = topk_weights.detach().requires_grad_(True)
        with torch.cuda.nvtx.range("dispatch_fwd_bwd"):
            recv_t, recv_w, _ = ep_dispatch(buffer, tokens_p, topk_idx, weights_p)
            (recv_t.float().square().mean() + recv_w.float().square().mean()).backward()

    def mlp_fwd_bwd():
        _zero_mlp_grads(mlp)
        recv_p = recv_tokens.detach().requires_grad_(True)
        weights_p = recv_weights.detach().requires_grad_(True)
        with torch.cuda.nvtx.range("grouped_mlp_fwd_bwd"):
            out = mlp(recv_p, splits, weights_p, splits)
            out.backward(mlp_dy)

    def combine_fwd_bwd():
        if expert_for_combine.grad is not None:
            expert_for_combine.grad.zero_()
        with torch.cuda.nvtx.range("combine_fwd_bwd"):
            out = ep_combine(buffer, expert_for_combine, num_local_tokens=cfg.tokens_per_rank)
            out.backward(grad_out)

    if cfg.profile:
        torch.cuda.profiler.start()

    timings: dict[str, float] = {}
    timings["dispatch_fwd_ms"] = time_cuda(dispatch_fwd, cfg.warmup, cfg.iters, sync_group=ep_group)
    timings["grouped_mlp_fwd_ms"] = time_cuda(mlp_fwd, cfg.warmup, cfg.iters, sync_group=ep_group)
    timings["combine_fwd_ms"] = time_cuda(combine_fwd, cfg.warmup, cfg.iters, sync_group=ep_group)
    timings["moe_e2e_fwd_ms"] = time_cuda(e2e_fwd, cfg.warmup, cfg.iters, sync_group=ep_group)

    if cfg.mode == "fwd_bwd":
        timings["dispatch_fwd_bwd_ms"] = time_cuda(
            dispatch_fwd_bwd, cfg.warmup, cfg.iters, sync_group=ep_group
        )
        timings["grouped_mlp_fwd_bwd_ms"] = time_cuda(
            mlp_fwd_bwd, cfg.warmup, cfg.iters, sync_group=ep_group
        )
        timings["combine_fwd_bwd_ms"] = time_cuda(
            combine_fwd_bwd, cfg.warmup, cfg.iters, sync_group=ep_group
        )
        timings["moe_e2e_fwd_bwd_ms"] = time_cuda(
            e2e_fwd_bwd, cfg.warmup, cfg.iters, sync_group=ep_group
        )

    if cfg.profile:
        torch.cuda.profiler.stop()

    _log(cfg.rank, "\n=== timings (ms / iter, min/max/mean over ranks) ===")
    for name, local_ms in timings.items():
        lo, hi, mean = _allgather_min_max_mean(local_ms, ep_group)
        _log(cfg.rank, f"  {name:24s}  min={lo:8.3f}  max={hi:8.3f}  mean={mean:8.3f}")

    mlp_ms = timings["grouped_mlp_fwd_ms"]
    flops = _mlp_fwd_flops(cfg, recv_rows)
    tokens_in_per_s = cfg.tokens_per_rank / (timings["moe_e2e_fwd_ms"] / 1e3)
    recv_per_s = recv_rows / (mlp_ms / 1e3)
    _log(cfg.rank, "\n=== throughput (rank-0 local, using mean stage times) ===")
    lo, hi, mean_e2e = _allgather_min_max_mean(timings["moe_e2e_fwd_ms"], ep_group)
    lo_m, hi_m, mean_mlp = _allgather_min_max_mean(mlp_ms, ep_group)
    _log(
        cfg.rank,
        f"  tokens_in/gpu/s (e2e fwd, mean ms)   = {cfg.tokens_per_rank / (mean_e2e / 1e3):.1f}",
    )
    _log(
        cfg.rank,
        f"  tokens_recv/gpu/s (mlp fwd, mean ms) = {recv_rows / (mean_mlp / 1e3):.1f}",
    )
    _log(
        cfg.rank,
        f"  grouped_mlp fwd TFLOP/s (mean ms)    = {flops / (mean_mlp / 1e3) / 1e12:.2f}",
    )
    _log(
        cfg.rank,
        f"  (local rank0: e2e {tokens_in_per_s:.1f} tok/s, mlp {recv_per_s:.1f} tok/s)",
    )
    del lo, hi, lo_m, hi_m

    dist.barrier(group=ep_group)
    ep_finalize()
    release_symm_mem_pool()
    dist.destroy_process_group()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gpus-per-node", type=int, default=DEFAULT_GPUS_PER_NODE)
    parser.add_argument("--num-local-experts", type=int, default=DEFAULT_LOCAL_EXPERTS)
    parser.add_argument("--hidden", type=int, default=DEFAULT_HIDDEN)
    parser.add_argument("--intermediate", type=int, default=DEFAULT_INTERMEDIATE)
    parser.add_argument(
        "--target-recv",
        type=int,
        default=DEFAULT_TARGET_RECV,
        help="Tokens each GPU should receive after dispatch (GroupedMLP M). "
        "tokens_in = target_recv / num_nodes, topk = num_nodes.",
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--mode", choices=("fwd", "fwd_bwd"), default="fwd_bwd")
    parser.add_argument(
        "--profile",
        action="store_true",
        help="NVTX ranges + cudaProfiler start/stop for "
        "`nsys profile --capture-range=cudaProfilerApi`.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    rank, world_size, local_rank = init_distributed()
    cfg = build_config(args, rank, world_size, local_rank)
    run_benchmark(cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
