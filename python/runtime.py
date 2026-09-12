"""Distributed execution, precision and training configuration."""
from __future__ import annotations
import contextlib
import os
import subprocess
from typing import Any, Mapping
import numpy as np
import torch
from torch import distributed as dist
from torch.utils.data import Sampler
from dataset import BatchDimensions
from model import ModelConfig
from pid_targets import SIGNED_RECO_PID_TARGETS

def nested(value: Mapping[str, Any], path: str, default: Any) -> Any:
    for part in path.split("."):
        if not isinstance(value, Mapping) or part not in value: return default
        value = value[part]
    return value

def distributed_context(device_name: str) -> tuple[int, int, int, torch.device]:
    world = int(os.environ.get("WORLD_SIZE", "1")); rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if device_name == "cuda":
        if not torch.cuda.is_available(): raise RuntimeError("--device cuda requested but CUDA is unavailable")
        torch.cuda.set_device(local_rank); device = torch.device("cuda", local_rank)
    else: device = torch.device("cpu")
    if world > 1: dist.init_process_group(backend="nccl" if device.type == "cuda" else "gloo", init_method="env://")
    return rank, world, local_rank, device

def reduce_sums(values: torch.Tensor) -> torch.Tensor:
    if dist.is_initialized(): dist.all_reduce(values, op=dist.ReduceOp.SUM)
    return values

def make_model_config(config: Mapping[str, Any], dims: BatchDimensions, metadata: Mapping[str, Any]) -> ModelConfig:
    model = config.get("model", {})
    bin_counts_explicit = "pid_bin_counts" in model
    bin_counts = model.get("pid_bin_counts", metadata.get("pid_bin_counts", []))
    if not bin_counts_explicit and not bin_counts and metadata.get("pid_target_names"):
        bin_counts = [int(nested(config, "objectives.masked_pid.n_bins", 32))] * len(metadata["pid_target_names"])
    architecture = str(model.get("architecture", "heterogeneous_particle_transformer"))
    return ModelConfig(
        particle_dim=dims.particle_features, pid_dim=dims.pid_features,
        architecture=architecture,
        global_dim=dims.global_features, vertex_dim=dims.vertex_features,
        candidate_dim=dims.candidate_features, pairwise_dim=dims.pairwise_features,
        use_polarity_conditioning=bool(model.get("use_polarity_conditioning", False)),
        d_model=int(model.get("d_model", 128)), n_heads=int(model.get("n_heads", 8)),
        n_layers=int(model.get("n_layers", 4)), dim_feedforward=int(model.get("dim_feedforward", 512)),
        dropout=float(model.get("dropout", 0.1)), pid_num_classes=int(model.get("pid_num_classes", 6)),
        pid_balanced_auxiliary=bool(model.get("pid_balanced_auxiliary", False)),
        pid_natural_head_detach_encoder=bool(
            model.get("pid_natural_head_detach_encoder", False)
        ),
        pid_bin_counts=[int(v) for v in bin_counts], pid_continuous_dim=int(model.get("pid_continuous_dim", 0)),
        origin_num_classes=int(model.get("origin_num_classes", 4)),
        vertex_num_classes=int(model.get("vertex_num_classes", 0)),
        candidate_num_classes=int(model.get("candidate_num_classes", 0)),
        flavour_num_classes=int(model.get("flavour_num_classes", 3)),
        flavour_heads_affine=bool(model.get("flavour_heads_affine", False)),
        beauty_sign_num_classes=int(model.get("beauty_sign_num_classes", 0)),
        charm_sign_num_classes=int(model.get("charm_sign_num_classes", 0)),
        b_vs_c_num_classes=int(model.get("b_vs_c_num_classes", 0)),
        b_vs_light_num_classes=int(model.get("b_vs_light_num_classes", 0)),
        c_vs_light_num_classes=int(model.get("c_vs_light_num_classes", 0)),
        b_vs_bbar_num_classes=int(model.get("b_vs_bbar_num_classes", 0)),
        c_vs_cbar_num_classes=int(model.get("c_vs_cbar_num_classes", 0)),
        topology_target_dim=int(model.get("topology_target_dim", metadata.get("topology_target_dim", 0))),
        enable_semantic_token=bool(model.get("enable_semantic_token", False)),
        enable_completeness_token=bool(model.get("enable_completeness_token", False)),
    )

def differentiable_zero(outputs: Mapping[str, Any]) -> torch.Tensor:
    """Return zero, retaining the graph during training and working in eval."""
    tensors = []
    for value in outputs.values():
        values = value if isinstance(value, (list, tuple)) else (value,)
        tensors.extend(item for item in values if isinstance(item, torch.Tensor) and item.is_floating_point())
    if not tensors:
        raise RuntimeError("The model emitted no floating-point tensors")
    graph_terms = [item.sum() for item in tensors if item.requires_grad]
    if graph_terms:
        return torch.stack(graph_terms).sum() * 0.0
    return tensors[0].sum() * 0.0

class ExactDistributedEvalSampler(Sampler[int]):
    """Disjoint, no-padding evaluation indices for one distributed rank."""

    def __init__(self, size: int, rank: int, world_size: int) -> None:
        self.size, self.rank, self.world_size = int(size), int(rank), int(world_size)
        if self.size < 0 or not 0 <= self.rank < self.world_size:
            raise ValueError(f"Invalid sampler parameters size={size}, rank={rank}, world={world_size}")

    def __iter__(self):
        return iter(range(self.rank, self.size, self.world_size))

    def __len__(self) -> int:
        if self.rank >= self.size:
            return 0
        return (self.size - 1 - self.rank) // self.world_size + 1

def distributed_weighted_mean(
    local_sum: torch.Tensor, local_weight: torch.Tensor, *, synchronize: bool = True,
) -> torch.Tensor:
    """Mean whose DDP-averaged gradient is weighted over all ranks."""
    denominator = local_weight.detach().to(device=local_sum.device, dtype=torch.float32).clone()
    world = 1
    if synchronize and dist.is_initialized():
        dist.all_reduce(denominator, op=dist.ReduceOp.SUM)
        world = dist.get_world_size()
    if denominator.item() <= 0:
        return local_sum * 0.0
    return local_sum * (world / denominator.to(local_sum.dtype))

def resolved_pid_species_weights(
    metadata: Mapping[str, Any], n_classes: int, *, power: float,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Build uncapped, mean-one inverse-frequency weights from train only."""
    if not 0.0 <= power <= 1.0:
        raise ValueError(f"PID species balance power must be in [0, 1], got {power}")
    counts = torch.tensor(
        metadata.get("marginal_target_statistics", {}).get("species_counts", []),
        dtype=torch.float64,
    )
    if counts.numel() != n_classes:
        raise ValueError(
            "Cannot balance PID species: training-manifest counts do not match "
            f"the model classes ({counts.numel()} != {n_classes})"
        )
    supported = counts > 0
    if not supported.any():
        raise ValueError("Cannot balance PID species: every training count is zero")
    weights = torch.zeros_like(counts)
    raw = counts[supported].pow(-power)
    # Normalize E_data[w(class)] to one. For power=1 this is exactly
    # N / (K n_c), so every supported class has identical total weight.
    normalization = counts[supported].sum() / (counts[supported] * raw).sum()
    weights[supported] = raw * normalization
    class_order = list(metadata.get("pid_class_order", []))
    names = class_order if len(class_order) == n_classes else [str(i) for i in range(n_classes)]
    resolved = {
        "definition": (
            "uncapped inverse-frequency power law normalized to mean event "
            "weight one; power=1 gives N/(K*n_class)"
        ),
        "source": "training-manifest reconstructed-species counts",
        "power": float(power),
        "capped": False,
        "counts": {name: int(count) for name, count in zip(names, counts.tolist())},
        "weights": {name: float(weight) for name, weight in zip(names, weights.tolist())},
        "zero_support_classes_ignored": [
            name for name, count in zip(names, counts.tolist()) if count <= 0
        ],
    }
    return weights.float(), resolved

def autocast_context(device: torch.device, precision: str):
    if precision == "fp32": return contextlib.nullcontext()
    dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    return torch.autocast(device_type=device.type, dtype=dtype)

def resolve_precision(requested: str, device: torch.device) -> str:
    if device.type == "cpu":
        if requested not in {"auto", "fp32"}:
            raise RuntimeError(f"{requested} mixed precision is only supported by this trainer on CUDA; use fp32 on CPU")
        return "fp32"
    native_bf16 = torch.cuda.is_bf16_supported() and torch.cuda.get_device_capability(device)[0] >= 8
    if requested == "auto":
        return "bf16" if native_bf16 else "fp16"
    if requested == "bf16" and not native_bf16:
        raise RuntimeError("bf16 requires a compute-capability 8+ GPU; use --precision fp16")
    return requested

def git_hash() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None
