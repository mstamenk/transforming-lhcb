#!/usr/bin/env python3
"""Train reconstructed-only deletion-denoising and missing-track completion.

The encoder sees
only a genuinely compacted residual jet: no removed slot, removed count,
corruption label, generator label or exclusive-decay label is an encoder input.
The ordinary masked-PID objective remains an auxiliary on surviving particles.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import itertools
import json
import math
import os
import platform
import random
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import yaml
from torch import distributed as dist
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Subset

from dataset import (
    BatchDimensions,
    ShardLocalDistributedSampler,
    deterministic_pid_mask,
    move_to_device,
    pid_eligibility,
    stable_sample_ids,
)
from corruption import (
    CORRUPTION_TYPE_NAMES,
    PARTICLE_TARGET_NAMES,
    REMOVED_SUMMARY_NAMES,
    TOPOLOGY_TARGET_NAMES,
    MissingTrackCorruptionConfig,
    MissingTrackCorruptionDataset,
    MissingTrackSelectionDataset,
    collate_missing_track_pairs,
    corruption_contract,
)
from model import ModelConfig, build_model
from losses import MISSING_LOSS_NAMES, missing_weights, compute_objective, compute_missing_objective
from pid_targets import SIGNED_RECO_PID_TARGETS
from runtime import (
    ExactDistributedEvalSampler,
    autocast_context,
    differentiable_zero,
    distributed_context,
    distributed_weighted_mean,
    git_hash,
    make_model_config,
    reduce_sums,
    resolve_precision,
    resolved_pid_species_weights,
)


DEPLOYABLE_ENCODER_INPUT_KEYS = frozenset({
    "particle_features", "particle_feature_valid", "particle_mask",
    "pid_features", "pid_available", "global_features", "polarity",
    "pairwise_inputs", "pairwise_features",
})
ENCODER_INPUT_KEYS = DEPLOYABLE_ENCODER_INPUT_KEYS
FORBIDDEN_ENCODER_KEYS = frozenset({
    "missing_count_target", "corruption_type_target", "removed_summary_target",
    "corruption_feasible_max", "corruption_feasible_types",
    "missing_set_valid", "missing_particle_target", "missing_particle_valid",
    "missing_delta_phi_period", "missing_pid_target",
    "missing_charge_target", "missing_topology_target", "missing_original_index",
    "target_reco_id", "pid_class", "pid_bins", "pid_target_valid",
    "jet_flavour", "analysis_jet_mc_flavour", "sample_id", "event_uid",
    "source_id", "source_entry", "jet_index", "split",
})
RESIDUAL_DEVICE_KEYS = ENCODER_INPUT_KEYS | frozenset({
    # Deterministic masked-PID selection and validation row identity.
    "sample_index", "event_uid", "event_id", "source_id", "source_entry",
    "jet_index",
    # Surviving-particle auxiliary targets.  None is an encoder input.
    "pid_class", "pid_bins", "pid_values", "pid_target_valid",
    "pid_eligible", "particle_charge",
    "vertex_mask", "vertex_track_mask", "candidate_mask", "candidate_track_mask",
})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/training.yaml")
    parser.add_argument("--train", required=True)
    parser.add_argument("--val", required=True)
    parser.add_argument("--output", default="outputs/checkpoints/missing_track_v1")
    parser.add_argument("--resume", help="Checkpoint path, or 'auto' for output/last.pt")
    parser.add_argument(
        "--init-checkpoint",
        help="Optional reconstructed-only masked-PID checkpoint used to initialize shared weights",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--workers", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--grad-accumulation", type=int)
    parser.add_argument("--precision", choices=("auto", "fp32", "fp16", "bf16"))
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--max-batches", type=int,
        help="Diagnostic smoke only: cap both train and validation batches",
    )
    parser.add_argument("--local-rank", type=int, default=None, help=argparse.SUPPRESS)
    return parser.parse_args()


def nested(value: Mapping[str, Any], path: str, default: Any = None) -> Any:
    current: Any = value
    for key in path.split("."):
        if not isinstance(current, Mapping) or key not in current:
            return default
        current = current[key]
    return current


def canonical_sha256(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


def missing_count_balance_contract(config: Mapping[str, Any]) -> dict[str, Any]:
    """Resolve the optional train-census count-balancing contract.

    Absence preserves the v1 natural-frequency cross entropy.  The only
    production balancing mode deliberately supported is full, uncapped
    inverse-frequency weighting from the exact frozen train corruption
    schedule.  This keeps validation/test labels out of loss construction.
    """

    raw = nested(config, "training.missing_count_balance", None)
    if raw is None or raw is False:
        return {"enabled": False, "source": "natural_frequency"}
    if not isinstance(raw, Mapping):
        raise ValueError("training.missing_count_balance must be a mapping")
    enabled = bool(raw.get("enabled", True))
    if not enabled:
        return {"enabled": False, "source": "natural_frequency"}
    source = str(raw.get("source", ""))
    power = float(raw.get("power", 1.0))
    uncapped = bool(raw.get("uncapped", True))
    if source != "exact_frozen_train_corruption_census":
        raise ValueError(
            "Balanced missing-count CE requires source: "
            "exact_frozen_train_corruption_census"
        )
    if power != 1.0 or not uncapped:
        raise ValueError(
            "Balanced missing-count CE is fixed to full uncapped inverse "
            "frequency (power: 1.0, uncapped: true)"
        )
    return {
        "enabled": True,
        "source": source,
        "power": 1.0,
        "uncapped": True,
    }


def make_missing_model_config(
    config: Mapping[str, Any], dims: BatchDimensions, metadata: Mapping[str, Any],
) -> ModelConfig:
    base = make_model_config(config, dims, metadata).to_dict()
    model = config.get("model", {})
    completion_fields = (
        "enable_semantic_token", "enable_completeness_token",
        "missing_track_max_queries", "missing_track_count_classes",
        "missing_track_count_balanced_auxiliary",
        "missing_track_count_natural_head_detach_encoder",
        "missing_track_particle_dim", "missing_track_pid_classes",
        "missing_track_pid_balanced_auxiliary",
        "missing_track_pid_natural_head_enabled",
        "missing_track_pid_natural_head_detach_encoder",
        "missing_track_charge_classes", "missing_track_summary_dim",
        "missing_track_corruption_classes", "missing_track_topology_dim",
        "missing_track_decoder_layers",
    )
    for name in completion_fields:
        if name in model:
            base[name] = model[name]
    return ModelConfig.from_dict(base)


def validate_contracts(
    config: Mapping[str, Any], train: MissingTrackCorruptionDataset,
    val: MissingTrackCorruptionDataset, model_config: ModelConfig,
) -> dict[str, Any]:
    if config.get("run_mode") != "missing_track_pretrain":
        raise ValueError("Dedicated trainer requires run_mode: missing_track_pretrain")
    objective = nested(config, "objectives.missing_track", {})
    if not isinstance(objective, Mapping) or not objective.get("enabled", False):
        raise ValueError("objectives.missing_track.enabled must be true")
    if train.base.metadata.get("preprocessing_sha256") != val.base.metadata.get("preprocessing_sha256"):
        raise ValueError("Train and validation preprocessing fingerprints differ")
    if train.base.metadata.get("pid_target_sha256") != val.base.metadata.get("pid_target_sha256"):
        raise ValueError("Train and validation signed-PID target fingerprints differ")
    target = train.base.metadata.get("pid_target_contract", {})
    if not isinstance(target, Mapping) or target.get("name") not in SIGNED_RECO_PID_TARGETS:
        raise ValueError("Missing-track training requires signed reconstructed-ID manifests")

    count_order = [int(value) for value in objective.get("count_class_order", ())]
    expected_count = list(range(train.config.max_missing + 1))
    if count_order != expected_count:
        raise ValueError(f"Count target order {count_order} != {expected_count}")
    if list(objective.get("removed_summary_names", ())) != list(REMOVED_SUMMARY_NAMES):
        raise ValueError("Configured removed-summary order differs from the corruption dataset")
    if list(nested(config, "corruption.missing_particle_feature_names", ())) != list(PARTICLE_TARGET_NAMES):
        raise ValueError("Configured missing-particle target order differs from the dataset")
    if list(nested(config, "corruption.missing_topology_names", ())) != list(TOPOLOGY_TARGET_NAMES):
        raise ValueError("Configured missing-topology target order differs from the dataset")
    configured_types = list(nested(config, "corruption.types", ()))
    if configured_types != list(CORRUPTION_TYPE_NAMES):
        raise ValueError(
            "Configured corruption-type order differs from the dataset: "
            f"config={configured_types}, dataset={list(CORRUPTION_TYPE_NAMES)}"
        )
    probabilities = [float(v) for v in nested(config, "corruption.count_probabilities", ())]
    if probabilities != list(train.config.count_probabilities):
        raise ValueError(
            "Configured count probabilities differ from the generated views: "
            f"config={probabilities}, dataset={list(train.config.count_probabilities)}"
        )
    if train.config != val.config:
        raise ValueError("Train and validation corruption configurations differ")
    if train.config.topology_mode != "drop_all":
        raise ValueError(
            "MC training requires topology_mode=drop_all"
        )

    expected_dims = {
        "missing_track_max_queries": train.config.max_missing,
        "missing_track_count_classes": len(count_order),
        "missing_track_particle_dim": len(PARTICLE_TARGET_NAMES),
        "missing_track_pid_classes": len(train.base.metadata.get("pid_class_order", ())),
        "missing_track_charge_classes": 3,
        "missing_track_summary_dim": len(REMOVED_SUMMARY_NAMES),
        "missing_track_corruption_classes": len(CORRUPTION_TYPE_NAMES),
        "missing_track_topology_dim": len(TOPOLOGY_TARGET_NAMES),
    }
    mismatch = {
        name: (getattr(model_config, name), expected)
        for name, expected in expected_dims.items()
        if int(getattr(model_config, name)) != int(expected)
    }
    if mismatch:
        raise ValueError(f"Model and missing-track target dimensions differ: {mismatch}")
    if not model_config.enable_completeness_token:
        raise ValueError("Missing-track heads require the dedicated completeness token")
    contract = corruption_contract(train.config)
    configured_sanitized = list(nested(config, "corruption.sanitized_global_names", ()))
    executable_sanitized = list(
        contract["global_policy"]["sanitized_to_training_center"]
    )
    if configured_sanitized != executable_sanitized:
        raise ValueError(
            "Configured/executable sanitized global lists differ: "
            f"{configured_sanitized} != {executable_sanitized}"
        )
    return {
        "contract": contract,
        "sha256": canonical_sha256(contract),
        "count_class_order": count_order,
        "count_probabilities": probabilities,
    }


def encoder_batch(batch: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    accidental = FORBIDDEN_ENCODER_KEYS.intersection(ENCODER_INPUT_KEYS)
    if accidental:
        raise AssertionError(f"Forbidden keys entered encoder allowlist: {sorted(accidental)}")
    result = {key: value for key, value in batch.items() if key in ENCODER_INPUT_KEYS}
    required = {"particle_features", "particle_mask", "pid_features", "polarity"}
    missing = required - set(result)
    if missing:
        raise KeyError(f"Residual batch lacks encoder inputs {sorted(missing)}")
    return result


def move_pair_to_device(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, Mapping):
            # The corruption dataset retains audit and reconstruction fields in
            # host memory.  Filter before the PCIe transfer, not after it, so
            # forbidden labels and dropped topology tensors never reach CUDA.
            selected = {
                name: tensor for name, tensor in value.items()
                if name in RESIDUAL_DEVICE_KEYS
            }
            result[key] = move_to_device(selected, device)
        else:
            result[key] = value.to(device, non_blocking=True)
    return result


def _slice_model_outputs(
    value: Any, selection: slice, combined_batch: int,
) -> Any:
    """Slice every batch-shaped tensor in the heterogeneous model output."""

    if isinstance(value, torch.Tensor):
        return value[selection] if value.ndim and value.shape[0] == combined_batch else value
    if isinstance(value, list):
        return [_slice_model_outputs(item, selection, combined_batch) for item in value]
    if isinstance(value, tuple):
        return tuple(_slice_model_outputs(item, selection, combined_batch) for item in value)
    if isinstance(value, Mapping):
        return {
            key: _slice_model_outputs(item, selection, combined_batch)
            for key, item in value.items()
        }
    return value


def _concatenate_encoder_batches(
    batches: list[Mapping[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    """Pad variable particle axes and concatenate several views once for DDP."""

    if not batches:
        raise ValueError("Cannot concatenate an empty encoder-batch list")
    keys = set(batches[0])
    if any(set(batch) != keys for batch in batches[1:]):
        raise ValueError("Clean and residual encoder views expose different keys")
    combined: dict[str, torch.Tensor] = {}
    for key in sorted(keys):
        values = [batch[key] for batch in batches]
        ndim = values[0].ndim
        if any(value.ndim != ndim for value in values):
            raise ValueError(f"Encoder view rank differs for {key!r}")
        tail = tuple(max(value.shape[axis] for value in values) for axis in range(1, ndim))
        padded: list[torch.Tensor] = []
        for value in values:
            if value.shape[1:] == tail:
                padded.append(value)
                continue
            target = value.new_zeros((value.shape[0], *tail))
            selection = (slice(None),) + tuple(slice(0, size) for size in value.shape[1:])
            target[selection] = value
            padded.append(target)
        combined[key] = torch.cat(padded, dim=0)
    return combined


def _pad_and_concatenate_pid_masks(
    masks: list[torch.Tensor], particle_width: int,
) -> torch.Tensor:
    padded: list[torch.Tensor] = []
    for mask in masks:
        if mask.shape[1] == particle_width:
            padded.append(mask)
            continue
        target = torch.zeros(
            mask.shape[0], particle_width, dtype=mask.dtype, device=mask.device,
        )
        target[:, :mask.shape[1]] = mask
        padded.append(target)
    return torch.cat(padded, dim=0)


def run_epoch(
    model: torch.nn.Module, loader: DataLoader, device: torch.device, *, epoch: int,
    seed: int, precision: str, weights: Mapping[str, float],
    pid_species_weights: torch.Tensor, missing_pid_species_weights: torch.Tensor,
    missing_count_class_weights: torch.Tensor | None,
    optimizer=None, scheduler=None, scaler=None,
    grad_accumulation: int = 1, max_grad_norm: float = 1.0,
    mask_fraction: float = 0.15, max_batches: int | None = None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    module = model.module if hasattr(model, "module") else model
    forward_model = model if training else module
    steps = len(loader) if max_batches is None else min(len(loader), max_batches)
    if steps <= 0:
        raise ValueError("Missing-track epoch has no batches")
    sums: dict[str, torch.Tensor] = {}
    if training:
        optimizer.zero_grad(set_to_none=True)
    for step, cpu_batch in enumerate(loader):
        if step >= steps:
            break
        batch = move_pair_to_device(cpu_batch, device)
        residual = batch["residual"]
        sample_ids = stable_sample_ids(residual)
        pid_mask = deterministic_pid_mask(
            pid_eligibility(residual), sample_ids, fraction=mask_fraction,
            seed=seed + 401, epoch=epoch if training else 0,
        )
        completion_pid_mask = torch.zeros_like(pid_mask)
        residual_inputs = encoder_batch(residual)
        sync_step = (step + 1) % grad_accumulation == 0 or step + 1 == steps
        sync_context = (
            contextlib.nullcontext()
            if not training or sync_step or not hasattr(model, "no_sync")
            else model.no_sync()
        )
        with sync_context, autocast_context(device, precision):
            pid_auxiliary = (
                weights.get("pid", 0.0) > 0
                or weights.get("pid_species_balanced", 0.0) > 0
            )
            view_inputs = [residual_inputs]
            view_masks = [completion_pid_mask]
            roles = ["completion"]
            if pid_auxiliary:
                view_inputs.append(residual_inputs)
                view_masks.append(pid_mask)
                roles.append("pid")
            # One joint DDP forward avoids reducer ambiguity from reusing the
            # shared encoder before a common backward.  Variable clean/residual
            # particle axes are padded only to the maximum width in this batch.
            if len(view_inputs) == 1:
                outputs = forward_model(residual_inputs, completion_pid_mask)
                pid_outputs = None
            else:
                combined_inputs = _concatenate_encoder_batches(view_inputs)
                particle_width = combined_inputs["particle_features"].shape[1]
                combined_mask = _pad_and_concatenate_pid_masks(view_masks, particle_width)
                combined_outputs = forward_model(combined_inputs, combined_mask)
                batch_size = completion_pid_mask.shape[0]
                combined_batch = len(roles) * batch_size
                sliced = {
                    role: _slice_model_outputs(
                        combined_outputs,
                        slice(index * batch_size, (index + 1) * batch_size),
                        combined_batch,
                    )
                    for index, role in enumerate(roles)
                }
                outputs = sliced["completion"]
                pid_outputs = sliced.get("pid")
            missing_loss, metrics = compute_missing_objective(
                outputs, batch, weights, missing_pid_species_weights,
                missing_count_class_weights,
                synchronize=training,
            )
            pid_weights = {
                "pid": weights.get("pid", 0.0),
                "pid_species_balanced": weights.get("pid_species_balanced", 0.0),
            }
            if pid_outputs is not None:
                pid_loss, pid_metrics = compute_objective(
                    pid_outputs, residual, pid_mask, pid_weights,
                    pid_species_class_weights=pid_species_weights,
                    synchronize_losses=training,
                )
            else:
                pid_loss = differentiable_zero(outputs)
                pid_metrics = {}
            total = missing_loss + pid_loss

            metrics.update({f"pidaux_{key}": value for key, value in pid_metrics.items()})
            metrics["loss_total"] = total.detach()
            scaled = total / grad_accumulation
        if training:
            scaler.scale(scaled).backward()
            if sync_step:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                if scheduler is not None:
                    scheduler.step()
        for key, value in metrics.items():
            if isinstance(value, torch.Tensor) and value.numel() == 1:
                sums[key] = sums.get(key, torch.zeros((), device=device)) + value.detach().float()
        sums["batches"] = sums.get("batches", torch.zeros((), device=device)) + 1

    keys = sorted(sums)
    if dist.is_initialized():
        gathered: list[list[str] | None] = [None] * dist.get_world_size()
        dist.all_gather_object(gathered, keys)
        keys = sorted(set().union(*(value or [] for value in gathered)))
    packed = torch.stack([sums.get(key, torch.zeros((), device=device)) for key in keys])
    reduced = reduce_sums(packed)
    result = dict(zip(keys, reduced.cpu().tolist()))
    batches = max(result.pop("batches", 1.0), 1.0)
    for key in [name for name in result if name.startswith("loss_")]:
        result[key] /= batches

    ratio_specs = {
        "missing_count_nll": ("missing_count_nll_sum", "missing_count_count"),
        "missing_count_balanced_nll": (
            "missing_count_balanced_nll_sum",
            "missing_count_balanced_weight_sum",
        ),
        "corruption_type_nll": ("corruption_type_nll_sum", "corruption_type_count"),
        "removed_summary_nll": ("removed_summary_nll_sum", "removed_summary_count"),
        "missing_existence_nll": ("missing_existence_nll_sum", "missing_existence_count"),
        "missing_feature_nll": ("missing_feature_nll_sum", "missing_feature_count"),
        "missing_charge_nll": ("missing_charge_nll_sum", "missing_charge_count"),
        "missing_topology_nll": ("missing_topology_nll_sum", "missing_topology_count"),
        "missing_pid_natural_nll": ("missing_pid_natural_nll_sum", "missing_pid_natural_count"),
        "missing_pid_balanced_nll": ("missing_pid_balanced_nll_sum", "missing_pid_balanced_count"),
        "surviving_pid_nll": ("pidaux_pid_nll_sum", "pidaux_masked_particles"),
        "surviving_pid_balanced_nll": (
            "pidaux_pid_balanced_nll_sum", "pidaux_pid_balanced_weight_sum",
        ),
    }
    for name, (numerator, denominator) in ratio_specs.items():
        if result.get(denominator, 0) > 0:
            result[name] = result[numerator] / result[denominator]
    if result.get("missing_count_count", 0):
        result["missing_count_accuracy"] = result["missing_count_correct"] / result["missing_count_count"]
    if result.get("missing_count_balanced_weight_sum", 0):
        result["missing_count_balanced_accuracy"] = (
            result["missing_count_balanced_correct"]
            / result["missing_count_count"]
        )
    for prefix in ("missing_count", "missing_count_balanced"):
        recalls = []
        for truth in range(int(module.config.missing_track_count_classes)):
            support = sum(
                result.get(f"{prefix}_confusion_{truth}_{predicted}", 0.0)
                for predicted in range(int(module.config.missing_track_count_classes))
            )
            if support > 0:
                recalls.append(
                    result.get(f"{prefix}_confusion_{truth}_{truth}", 0.0)
                    / support
                )
        if recalls:
            result[f"{prefix}_macro_recall"] = float(np.mean(recalls))
    if result.get("corruption_type_count", 0):
        result["corruption_type_accuracy"] = result["corruption_type_correct"] / result["corruption_type_count"]
    if result.get("missing_existence_count", 0):
        result["missing_existence_accuracy"] = result["missing_existence_correct"] / result["missing_existence_count"]
    if result.get("missing_charge_count", 0):
        result["missing_charge_accuracy"] = result["missing_charge_correct"] / result["missing_charge_count"]
    for name in ("missing_pid_natural", "missing_pid_balanced"):
        if result.get(f"{name}_count", 0):
            # The weighted denominator is used for NLL; raw correctness support
            # is the number of true missing objects.
            support = result.get("missing_charge_count", 0)
            if support:
                result[f"{name}_accuracy"] = result[f"{name}_correct"] / support

    set_components = [
        result[name] for name in (
            "missing_existence_nll", "missing_feature_nll", "missing_charge_nll",
        ) if name in result
    ]
    if set_components:
        result["missing_set_nll"] = float(np.mean(set_components))
    # Reconstruct the validation selection metric from globally reduced raw
    # sums, avoiding bias from unequal exact-evaluation rank tails.
    component_for_weight = {
        "missing_count": "missing_count_nll",
        "missing_count_balanced": "missing_count_balanced_nll",
        "corruption_type": "corruption_type_nll",
        "removed_summary": "removed_summary_nll",
        "missing_set": "missing_set_nll",
        "missing_topology": "missing_topology_nll",
        "missing_pid_natural": "missing_pid_natural_nll",
        "missing_pid_balanced": "missing_pid_balanced_nll",
    }
    selection = 0.0
    for loss_name, metric_name in component_for_weight.items():
        if weights.get(loss_name, 0) > 0:
            if metric_name not in result:
                raise RuntimeError(f"Active loss {loss_name} has no epoch metric {metric_name}")
            selection += weights[loss_name] * result[metric_name]
    # PID selection terms are logged by the inherited objective but are not
    # mixed into the primary completeness checkpoint criterion.
    result["missing_track_validation_nll"] = selection
    joint = selection
    for loss_name, metric_name in (
        ("pid", "surviving_pid_nll"),
        ("pid_species_balanced", "surviving_pid_balanced_nll"),
    ):
        if weights.get(loss_name, 0) > 0:
            if metric_name not in result:
                raise RuntimeError(
                    f"Active joint loss {loss_name} has no epoch metric {metric_name}"
                )
            joint += weights[loss_name] * result[metric_name]
    result["joint_validation_nll"] = joint
    return result


def initialize_from_checkpoint(
    model: torch.nn.Module, checkpoint: Mapping[str, Any], provenance: Mapping[str, Any],
) -> dict[str, Any]:
    old_provenance = checkpoint.get("provenance", {})
    if old_provenance.get("preprocessing_sha256") != provenance.get("preprocessing_sha256"):
        raise ValueError("Initialization checkpoint uses different preprocessing")
    if old_provenance.get("pid_target_sha256") != provenance.get("pid_target_sha256"):
        raise ValueError("Initialization checkpoint uses a different PID target contract")
    if checkpoint.get("config", {}).get("run_mode") not in {"pretrain", "missing_track_pretrain"}:
        raise ValueError("Initialization must come from reconstructed-only self-supervision")
    current = model.state_dict()
    source = checkpoint.get("model", {})
    incompatible_shape = {
        key: (tuple(value.shape), tuple(current[key].shape))
        for key, value in source.items()
        if key in current and tuple(value.shape) != tuple(current[key].shape)
    }
    if incompatible_shape:
        raise ValueError(f"Shared checkpoint parameter shapes differ: {incompatible_shape}")
    compatible = {key: value for key, value in source.items() if key in current}
    unknown = sorted(set(source) - set(current))
    if unknown:
        raise ValueError(f"Initialization checkpoint has unknown model parameters: {unknown[:12]}")
    missing = sorted(set(current) - set(compatible))
    model.load_state_dict(compatible, strict=False)
    return {"loaded_parameters": len(compatible), "new_parameters": missing}


def atomic_torch_save(payload: Mapping[str, Any], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def expected_count_prior_from_feasibility(
    feasible_max_counts: Sequence[int], design_probabilities: Sequence[float],
) -> list[float]:
    """Expected marginal K prior for deterministic epoch-resampled views.

    Each jet draws from the configured K proposal truncated and renormalized
    to its maximum feasible deletion count.  The feasibility census is fixed
    by the input jets, so this expectation is exact for the resampling design
    without pretending that one epoch's finite draw is the all-epoch prior.
    """

    counts = np.asarray(feasible_max_counts, dtype=np.float64)
    design = np.asarray(design_probabilities, dtype=np.float64)
    if counts.ndim != 1 or design.ndim != 1 or len(counts) != len(design):
        raise ValueError("Feasibility counts and count design must have equal length")
    if np.any(counts < 0) or counts.sum() <= 0:
        raise ValueError("Feasibility counts must be nonnegative with positive total")
    if np.any(design < 0) or not np.isclose(design.sum(), 1.0, rtol=0, atol=1e-12):
        raise ValueError("Count design probabilities must be nonnegative and normalized")
    expected = np.zeros_like(design)
    for maximum, population in enumerate(counts):
        allowed = design[:maximum + 1]
        normalization = allowed.sum()
        if population and normalization <= 0:
            raise ValueError(f"No supported K proposal for feasible maximum {maximum}")
        if population:
            expected[:maximum + 1] += population * allowed / normalization
    expected /= counts.sum()
    if not np.isclose(expected.sum(), 1.0, rtol=0, atol=1e-12):
        raise RuntimeError("Expected resampled count prior is not normalized")
    return [float(value) for value in expected]


def validate_train_corruption_census(
    payload: Mapping[str, Any], *, dataset: MissingTrackCorruptionDataset,
    n_pid_classes: int,
) -> dict[str, Any]:
    """Validate the train-only epoch-0 reference corruption census."""

    if not isinstance(payload, Mapping) or payload.get("diagnostic_skipped", False):
        raise ValueError("A production resume requires a complete corruption census")
    resampled = bool(dataset.config.resample_each_epoch)
    expected_schedule = "epoch_resampled" if resampled else "fixed"
    schedule = str(payload.get("schedule", "fixed"))
    if schedule != expected_schedule:
        raise ValueError(
            f"Corruption census schedule {schedule!r} != {expected_schedule!r}"
        )
    if bool(payload.get("frozen_schedule", True)) != (not resampled):
        raise ValueError("Corruption census frozen-schedule flag is inconsistent")
    if int(payload.get("reference_epoch", 0)) != 0:
        raise ValueError("Corruption balancing reference must be epoch 0")

    def integer_vector(name: str, length: int) -> list[int]:
        raw = payload.get(name)
        if (
            not isinstance(raw, Sequence)
            or isinstance(raw, (str, bytes))
            or len(raw) != length
        ):
            raise ValueError(f"Corruption census {name} must have length {length}")
        result = [int(value) for value in raw]
        if any(float(value) != result[index] or result[index] < 0 for index, value in enumerate(raw)):
            raise ValueError(f"Corruption census {name} is not nonnegative integer data")
        return result

    rows = int(payload.get("rows", -1))
    if rows != len(dataset):
        raise ValueError(
            f"Corruption census rows {rows} != current train views {len(dataset)}"
        )
    count_counts = integer_vector(
        "count_counts", dataset.config.max_missing + 1,
    )
    type_counts = integer_vector("type_counts", len(CORRUPTION_TYPE_NAMES))
    feasible_max_counts = integer_vector(
        "feasible_max_counts", dataset.config.max_missing + 1,
    )
    feasible_type_opportunities = integer_vector(
        "feasible_type_opportunities", len(CORRUPTION_TYPE_NAMES),
    )
    pid_counts = integer_vector("missing_pid_counts", n_pid_classes)
    if min(count_counts) <= 0:
        raise ValueError(f"Every missing-count class needs train support: {count_counts}")
    if min(pid_counts) <= 0:
        raise ValueError(f"Every missing-PID class needs train support: {pid_counts}")
    if sum(count_counts) != rows or sum(type_counts) != rows:
        raise ValueError("Corruption census count/type totals differ from train rows")
    if sum(feasible_max_counts) != rows:
        raise ValueError("Corruption census feasible-maximum totals differ from train rows")
    if type_counts[0] != count_counts[0]:
        raise ValueError("Complete corruption/type counts differ in the census")
    expected_missing = sum(
        count * examples for count, examples in enumerate(count_counts)
    )
    if sum(pid_counts) != expected_missing:
        raise ValueError(
            f"Corruption census PID support {sum(pid_counts)} != {expected_missing}"
        )
    if int(payload.get("missing_pid_total", expected_missing)) != expected_missing:
        raise ValueError("Corruption census missing_pid_total is inconsistent")
    if list(payload.get("type_names", ())) != list(CORRUPTION_TYPE_NAMES):
        raise ValueError("Corruption census type-name order differs from the trainer")

    realized_probabilities = [value / rows for value in count_counts]
    probabilities = (
        expected_count_prior_from_feasibility(
            feasible_max_counts, dataset.config.count_probabilities,
        )
        if resampled else realized_probabilities
    )
    recorded_probabilities = payload.get("count_probabilities")
    if (
        not isinstance(recorded_probabilities, Sequence)
        or isinstance(recorded_probabilities, (str, bytes))
        or len(recorded_probabilities) != len(probabilities)
        or not np.allclose(
            np.asarray(recorded_probabilities, dtype=np.float64),
            np.asarray(probabilities, dtype=np.float64),
            rtol=0.0,
            atol=1e-15,
        )
    ):
        raise ValueError("Corruption census score prior does not match its schedule")
    return {
        **dict(payload),
        "rows": rows,
        "count_counts": count_counts,
        "count_probabilities": probabilities,
        "type_counts": type_counts,
        "type_names": list(CORRUPTION_TYPE_NAMES),
        "feasible_max_counts": feasible_max_counts,
        "feasible_type_opportunities": feasible_type_opportunities,
        "missing_pid_counts": pid_counts,
        "missing_pid_total": expected_missing,
        "schedule": expected_schedule,
        "reference_epoch": 0,
        "reference_count_probabilities": realized_probabilities,
        "prior_kind": (
            "expected_resampled_train_schedule"
            if resampled else "exact_fixed_train_schedule"
        ),
        "frozen_schedule": not resampled,
    }


def scan_train_corruption_census(
    dataset: MissingTrackCorruptionDataset, *, n_pid_classes: int,
    rank: int, world: int, workers: int,
    batch_size: int,
) -> dict[str, Any]:
    """Measure the exact train-only epoch-0 reference corruption frequencies."""

    if int(dataset.epoch) != 0:
        raise ValueError("The train corruption reference census must use epoch 0")
    size = len(dataset)
    start = size * rank // world
    stop = size * (rank + 1) // world
    selection = MissingTrackSelectionDataset(dataset)
    local = Subset(selection, range(start, stop))
    loader = DataLoader(
        local,
        batch_size=max(1, int(batch_size)),
        shuffle=False,
        num_workers=max(0, int(workers)),
        persistent_workers=False,
        # torchrun initializes CUDA/NCCL before this production census.  The
        # default Linux ``fork`` context can clone a live CUDA/NCCL process and
        # corrupt the parent's communicator (observed as an asynchronous
        # illegal-memory-access in the subsequent all_reduce).  Spawned workers
        # never inherit CUDA/NCCL state; the census itself remains CPU-only.
        multiprocessing_context="spawn" if int(workers) > 0 else None,
    )
    count_counts = torch.zeros(dataset.config.max_missing + 1, dtype=torch.long)
    type_counts = torch.zeros(len(CORRUPTION_TYPE_NAMES), dtype=torch.long)
    feasible_max_counts = torch.zeros(dataset.config.max_missing + 1, dtype=torch.long)
    feasible_type_counts = torch.zeros(len(CORRUPTION_TYPE_NAMES), dtype=torch.long)
    pid_counts = torch.zeros(n_pid_classes, dtype=torch.long)
    rows = 0
    for batch in loader:
        count_counts += torch.bincount(
            batch["missing_count"].long(), minlength=len(count_counts),
        )
        type_counts += torch.bincount(
            batch["corruption_type"].long(), minlength=len(type_counts),
        )
        feasible_max_counts += torch.bincount(
            batch["feasible_max"].long(), minlength=len(feasible_max_counts),
        )
        feasible_type_counts += batch["feasible_types"].long().sum(0)
        pid_counts += batch["missing_pid_counts"].long().sum(0)
        rows += int(batch["missing_count"].numel())
    packed = torch.cat((
        torch.tensor([rows], dtype=torch.long), count_counts, type_counts,
        feasible_max_counts, feasible_type_counts, pid_counts,
    ))
    if dist.is_initialized():
        # The census is CPU bookkeeping.  Keep it off CUDA entirely, including
        # its one collective, so startup cannot consume or diagnose errors on
        # the training CUDA stream.  The default process group remains NCCL for
        # DDP; this short-lived auxiliary group only reduces a few int64s.
        census_group = (
            dist.new_group(backend="gloo")
            if str(dist.get_backend()).lower() == "nccl"
            else None
        )
        try:
            dist.all_reduce(
                packed, op=dist.ReduceOp.SUM, group=census_group,
            )
        finally:
            if census_group is not None:
                dist.destroy_process_group(census_group)
    cursor = 1
    total_rows = int(packed[0])
    count_counts = packed[cursor:cursor + len(count_counts)]; cursor += len(count_counts)
    type_counts = packed[cursor:cursor + len(type_counts)]; cursor += len(type_counts)
    feasible_max_counts = packed[cursor:cursor + len(feasible_max_counts)]; cursor += len(feasible_max_counts)
    feasible_type_counts = packed[cursor:cursor + len(feasible_type_counts)]; cursor += len(feasible_type_counts)
    pid_counts = packed[cursor:cursor + n_pid_classes]
    if total_rows != size or int(count_counts.sum()) != size:
        raise RuntimeError(
            f"Corruption census lost rows: rows={total_rows}, counts={int(count_counts.sum())}, "
            f"expected={size}"
        )
    if (pid_counts <= 0).any():
        missing = torch.nonzero(pid_counts <= 0, as_tuple=False).flatten().tolist()
        raise RuntimeError(f"Missing-target census has zero support for PID classes {missing}")
    total_missing = int(pid_counts.sum())
    expected_missing = int(sum(index * int(value) for index, value in enumerate(count_counts)))
    if total_missing != expected_missing:
        raise RuntimeError(
            f"PID census support {total_missing} != missing-count total {expected_missing}"
        )
    realized_count_probability = (
        count_counts.double() / count_counts.sum()
    ).tolist()
    count_probability = (
        expected_count_prior_from_feasibility(
            [int(value) for value in feasible_max_counts],
            dataset.config.count_probabilities,
        )
        if dataset.config.resample_each_epoch
        else [float(value) for value in realized_count_probability]
    )
    result = {
        "rows": total_rows,
        "count_counts": [int(value) for value in count_counts],
        "count_probabilities": [float(value) for value in count_probability],
        "reference_count_probabilities": [
            float(value) for value in realized_count_probability
        ],
        "type_counts": [int(value) for value in type_counts],
        "type_names": list(CORRUPTION_TYPE_NAMES),
        "feasible_max_counts": [int(value) for value in feasible_max_counts],
        "feasible_type_opportunities": [int(value) for value in feasible_type_counts],
        "missing_pid_counts": [int(value) for value in pid_counts],
        "missing_pid_total": total_missing,
        "schedule": (
            "epoch_resampled"
            if dataset.config.resample_each_epoch else "fixed"
        ),
        "reference_epoch": 0,
        "prior_kind": (
            "expected_resampled_train_schedule"
            if dataset.config.resample_each_epoch
            else "exact_fixed_train_schedule"
        ),
        "frozen_schedule": bool(not dataset.config.resample_each_epoch),
    }
    return validate_train_corruption_census(
        result, dataset=dataset, n_pid_classes=n_pid_classes,
    )


def exact_inverse_frequency_weights(counts: Sequence[int]) -> torch.Tensor:
    values = torch.tensor(counts, dtype=torch.float64)
    if values.ndim != 1 or not len(values) or (values <= 0).any():
        raise ValueError(f"Exact balancing needs positive class counts, got {counts}")
    return (values.sum() / (len(values) * values)).float()


def save_checkpoint(
    path: Path, model, optimizer, scheduler, scaler, *, epoch: int,
    global_step: int, best: float, history: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any], model_config: ModelConfig,
    provenance: Mapping[str, Any],
    selection_metric: str = "missing_track_validation_nll",
) -> None:
    module = model.module if hasattr(model, "module") else model
    atomic_torch_save({
        "format_version": 2,
        "epoch": epoch,
        "global_step": global_step,
        "selection_metric": selection_metric,
        "best_validation_metric": best,
        "model": module.state_dict(),
        "model_config": model_config.to_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "history": list(history),
        "config": dict(config),
        "preprocessing_metadata": provenance["preprocessing_metadata"],
        "provenance": dict(provenance),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
    }, path)


def main() -> None:
    args = parse_args()
    if args.max_batches is not None and args.max_batches <= 0:
        raise ValueError("--max-batches must be positive")
    if args.resume and args.init_checkpoint:
        raise ValueError("--resume and --init-checkpoint are mutually exclusive")
    if args.force and args.resume:
        raise ValueError("--force cannot replace a resumed run")
    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text()) or {}
    weights = missing_weights(config)
    count_balance = missing_count_balance_contract(config)
    balanced_count_active = weights.get("missing_count_balanced", 0.0) > 0
    if balanced_count_active != bool(count_balance["enabled"]):
        raise ValueError(
            "missing_count_balanced loss and training.missing_count_balance "
            "must be enabled together"
        )
    seed = int(args.seed if args.seed is not None else config.get("seed", 2026))
    rank, world, _, device = distributed_context(args.device)
    random.seed(seed + rank)
    np.random.seed(seed + rank)
    torch.manual_seed(seed + rank)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed + rank)

    output = Path(args.output)
    output_state = torch.zeros((), dtype=torch.int32, device=device)
    if rank == 0:
        if output.exists() and not output.is_dir():
            output_state.fill_(2)
        elif output.exists() and any(output.iterdir()) and not args.resume and not args.force:
            output_state.fill_(1)
        elif output.exists() and args.force:
            shutil.rmtree(output)
        if int(output_state.item()) != 2:
            output.mkdir(parents=True, exist_ok=True)
    if dist.is_initialized():
        dist.broadcast(output_state, 0)
        dist.barrier()
    if int(output_state.item()) == 1:
        raise FileExistsError(f"Output directory is non-empty: {output}")
    if int(output_state.item()) == 2:
        raise NotADirectoryError(output)

    corruption_config = MissingTrackCorruptionConfig.from_mapping(config.get("corruption", {}))
    cache_size = int(nested(config, "data.cache_size", 1))
    include_clean = False
    train_data = MissingTrackCorruptionDataset(
        args.train, config=corruption_config, cache_size=cache_size,
        include_clean=include_clean,
    )
    val_data = MissingTrackCorruptionDataset(
        args.val, config=corruption_config, cache_size=cache_size,
        include_clean=include_clean,
    )
    val_data.set_epoch(0)

    training = config.get("training", {})
    workers = int(args.workers if args.workers is not None else training.get("workers", 4))
    batch_size = int(args.batch_size if args.batch_size is not None else training.get("batch_size_per_gpu", 128))
    epochs = int(args.epochs if args.epochs is not None else training.get("epochs", 100))
    accumulation = int(args.grad_accumulation if args.grad_accumulation is not None else training.get("gradient_accumulation", 1))
    precision = resolve_precision(args.precision or training.get("precision", "auto"), device)
    if min(batch_size, epochs, accumulation) <= 0 or workers < 0:
        raise ValueError("Batch, epochs and accumulation must be positive; workers nonnegative")

    # Shape discovery must not start the persistent training worker pool before
    # the one-off census.  Apart from temporarily doubling the worker count, an
    # early iterator used to fork workers from a live CUDA/NCCL parent.  These
    # dimensions are part of the validated manifest contract, so no shard (and
    # therefore no main-process shard cache) needs to be opened here.
    feature_names = train_data.base.metadata.get("feature_names", {})
    dimensions = {
        name: len(feature_names.get(name, ()))
        for name in ("particle", "pid", "global", "vertex", "candidate")
    }
    pairwise_features = len(
        train_data.base.metadata.get("pairwise_features", ())
    )
    if min(dimensions["particle"], dimensions["pid"]) <= 0 or pairwise_features <= 0:
        raise ValueError(
            "Training manifest lacks particle, PID, or pairwise feature dimensions"
        )
    dims = BatchDimensions(
        particle_features=dimensions["particle"],
        pid_features=dimensions["pid"],
        global_features=dimensions["global"],
        vertex_features=dimensions["vertex"],
        candidate_features=dimensions["candidate"],
        pairwise_features=pairwise_features,
    )
    model_config = make_missing_model_config(config, dims, train_data.base.metadata)
    if (
        bool(model_config.missing_track_count_balanced_auxiliary)
        != balanced_count_active
    ):
        raise ValueError(
            "model.missing_track_count_balanced_auxiliary must match the "
            "missing_count_balanced loss"
        )
    if (
        model_config.missing_track_count_natural_head_detach_encoder
        and weights.get("missing_count", 0.0) <= 0
    ):
        raise ValueError(
            "A detached natural missing-count head still requires a positive "
            "missing_count calibration loss"
        )
    natural_missing_pid_active = weights.get("missing_pid_natural", 0.0) > 0
    if (
        bool(model_config.missing_track_pid_natural_head_enabled)
        != natural_missing_pid_active
    ):
        raise ValueError(
            "model.missing_track_pid_natural_head_enabled must match the "
            "missing_pid_natural loss"
        )
    if (
        weights.get("missing_pid_balanced", 0.0) > 0
        and not model_config.missing_track_pid_balanced_auxiliary
    ):
        raise ValueError(
            "An active missing_pid_balanced loss requires its auxiliary head"
        )
    resolved_contract = validate_contracts(config, train_data, val_data, model_config)
    # Keep the model on CPU through the CPU-only census.  This avoids occupying
    # GPU memory or launching unrelated CUDA work while startup accounting and
    # its spawned workers run.
    model = build_model(model_config)

    provenance: dict[str, Any] = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_hash(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "seed": seed,
        "world_size": world,
        "train_source": str(Path(args.train).resolve()),
        "validation_source": str(Path(args.val).resolve()),
        "preprocessing_sha256": train_data.base.metadata.get("preprocessing_sha256"),
        "pid_target_sha256": train_data.base.metadata.get("pid_target_sha256"),
        "pid_target_contract": train_data.base.metadata.get("pid_target_contract"),
        "preprocessing_metadata": train_data.base.metadata,
        "missing_track_contract": resolved_contract["contract"],
        "missing_track_contract_sha256": resolved_contract["sha256"],
        "encoder_input_keys": sorted(DEPLOYABLE_ENCODER_INPUT_KEYS),
        "trainer_only_selector_keys": [],
        "generator_labels_are_encoder_inputs": False,
        "exclusive_decay_labels_used": False,
        "event_met_target": False,
    }

    resume_path = (
        output / "last.pt" if args.resume == "auto"
        else Path(args.resume) if args.resume else None
    )
    resume_checkpoint: Mapping[str, Any] | None = None
    if resume_path is not None:
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint is absent: {resume_path}")
        resume_checkpoint = torch.load(
            resume_path, map_location="cpu", weights_only=False,
        )
        checkpoint_provenance = resume_checkpoint.get("provenance", {})
        if not isinstance(checkpoint_provenance, Mapping):
            raise ValueError("Resume checkpoint has no provenance mapping")
        if (
            checkpoint_provenance.get("missing_track_contract_sha256")
            != resolved_contract["sha256"]
        ):
            raise ValueError("Resume checkpoint has a different corruption contract")
        if resume_checkpoint.get("model_config") != model_config.to_dict():
            raise ValueError("Resume checkpoint model configuration differs")
        for name in (
            "train_source", "validation_source", "preprocessing_sha256",
            "pid_target_sha256", "seed",
        ):
            if checkpoint_provenance.get(name) != provenance.get(name):
                raise ValueError(
                    f"Resume checkpoint provenance differs for {name}: "
                    f"{checkpoint_provenance.get(name)!r} != {provenance.get(name)!r}"
                )

    pid_species_weights, balance = resolved_pid_species_weights(
        train_data.base.metadata,
        model_config.pid_num_classes,
        power=float(nested(config, "training.pid_species_balance.power", 1.0)),
    )
    provenance["pid_species_balance"] = balance
    checkpoint_census = (
        resume_checkpoint.get("provenance", {}).get("missing_track_census", {})
        if resume_checkpoint is not None else None
    )
    if (
        resume_checkpoint is not None
        and isinstance(checkpoint_census, Mapping)
        and not checkpoint_census.get("diagnostic_skipped", False)
    ):
        census = validate_train_corruption_census(
            checkpoint_census,
            dataset=train_data,
            n_pid_classes=model_config.missing_track_pid_classes,
        )
        if rank == 0:
            print(json.dumps({
                "stage": "reuse_train_corruption_census",
                "checkpoint": str(resume_path.resolve()),
                "rows": census["rows"],
            }))
        missing_pid_species_weights = exact_inverse_frequency_weights(
            census["missing_pid_counts"]
        )
    elif args.max_batches is None:
        if resume_checkpoint is not None:
            raise ValueError(
                "A production resume cannot reuse a diagnostic/skipped census"
            )
        if rank == 0:
            print(json.dumps({
                "stage": "train_corruption_census",
                "rows": len(train_data),
                "purpose": "exact train-only count prior and missing-PID balance",
            }))
        census = scan_train_corruption_census(
            train_data,
            n_pid_classes=model_config.missing_track_pid_classes,
            rank=rank,
            world=world,
            workers=int(training.get("census_workers", workers)),
            batch_size=int(training.get("census_batch_size", max(batch_size, 512))),
        )
        missing_pid_species_weights = exact_inverse_frequency_weights(
            census["missing_pid_counts"]
        )
    else:
        # A bounded smoke must remain bounded; production refuses this shortcut
        # because --max-batches is absent there.
        census = {
            "diagnostic_skipped": True,
            "reason": "--max-batches bounds the smoke run",
            "count_probabilities": list(corruption_config.count_probabilities),
            "missing_pid_counts": [
                int(balance["counts"][name])
                for name in train_data.base.metadata.get("pid_class_order", ())
            ],
            "frozen_schedule": bool(not corruption_config.resample_each_epoch),
        }
        missing_pid_species_weights = pid_species_weights.clone()
    provenance["missing_track_census"] = census
    if count_balance["enabled"]:
        count_source = (
            census["count_counts"]
            if not census.get("diagnostic_skipped", False)
            else census["count_probabilities"]
        )
        missing_count_class_weights = exact_inverse_frequency_weights(
            count_source
        )
    else:
        missing_count_class_weights = None
    provenance["missing_count_balance"] = {
        **count_balance,
        "balanced_auxiliary_head": bool(
            model_config.missing_track_count_balanced_auxiliary
        ),
        "natural_head_detached_from_encoder": bool(
            model_config.missing_track_count_natural_head_detach_encoder
        ),
        "deployable_score_uses_natural_head_only": True,
        "counts": (
            list(census["count_counts"])
            if "count_counts" in census else None
        ),
        "weights": (
            [float(value) for value in missing_count_class_weights]
            if missing_count_class_weights is not None else None
        ),
        "diagnostic_fallback": bool(census.get("diagnostic_skipped", False)),
    }
    provenance["missing_pid_species_balance"] = {
        "source": (
            "exact fixed train corruption targets"
            if not corruption_config.resample_each_epoch
            else "deterministic epoch-0 train corruption reference"
        ),
        "power": 1.0,
        "uncapped": True,
        "counts": list(census["missing_pid_counts"]),
        "weights": [float(value) for value in missing_pid_species_weights],
        "diagnostic_fallback": bool(census.get("diagnostic_skipped", False)),
    }
    if rank == 0:
        (output / "corruption_census.json").write_text(
            json.dumps({
                **census,
                "missing_pid_weights": [
                    float(value) for value in missing_pid_species_weights
                ],
                "missing_count_weights": (
                    [float(value) for value in missing_count_class_weights]
                    if missing_count_class_weights is not None else None
                ),
            }, indent=2) + "\n"
        )

    # Only create the persistent epoch loaders after the one-off census workers
    # have exited.  This keeps the peak worker count at workers-per-rank rather
    # than twice that number during startup.
    train_sampler = ShardLocalDistributedSampler(
        train_data, num_replicas=world, rank=rank, shuffle=True, seed=seed,
    )
    val_sampler = (
        ExactDistributedEvalSampler(len(val_data), rank, world)
        if world > 1 else None
    )
    loader_options = dict(
        batch_size=batch_size,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_missing_track_pairs,
        # CUDA/NCCL is already initialized by distributed_context().  Never
        # fork DataLoader children from that process: even CPU-only workers can
        # poison the inherited NCCL communicator and make the next collective
        # fail with an illegal CUDA memory access.
        multiprocessing_context="spawn" if workers > 0 else None,
        # A frozen production corruption schedule is identical in every epoch,
        # so retaining workers avoids repeatedly spawning processes and losing
        # their shard caches.  Resampled diagnostic schedules still require new
        # worker copies to inherit dataset.set_epoch.
        persistent_workers=(
            workers > 0 and not corruption_config.resample_each_epoch
        ),
    )
    train_loader = DataLoader(train_data, sampler=train_sampler, **loader_options)
    val_loader = DataLoader(
        val_data, sampler=val_sampler, shuffle=False, **loader_options,
    )

    initialization = None
    if args.init_checkpoint:
        initial = torch.load(args.init_checkpoint, map_location="cpu", weights_only=False)
        initialization = initialize_from_checkpoint(model, initial, provenance)
        initialization["checkpoint"] = str(Path(args.init_checkpoint).resolve())
        provenance["initialization"] = initialization

    model = model.to(device)
    if world > 1:
        model = DistributedDataParallel(
            model,
            device_ids=[device.index] if device.type == "cuda" else None,
            find_unused_parameters=bool(nested(config, "training.ddp.find_unused_parameters", True)),
        )
    optimizer = torch.optim.AdamW(
        (value for value in model.parameters() if value.requires_grad),
        lr=float(training.get("learning_rate", 6e-4)),
        weight_decay=float(training.get("weight_decay", 1e-2)),
    )
    steps_per_epoch = len(train_loader) if args.max_batches is None else min(len(train_loader), args.max_batches)
    total_steps = max(1, math.ceil(steps_per_epoch / accumulation) * epochs)
    warmup = int(training.get("warmup_steps", max(1, total_steps // 20)))

    def schedule(step: int) -> float:
        rising = min((step + 1) / warmup, 1.0)
        falling = 0.5 * (1 + math.cos(
            math.pi * max(0, step - warmup) / max(1, total_steps - warmup)
        ))
        return rising * falling

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    scaler = torch.amp.GradScaler(
        "cuda", enabled=device.type == "cuda" and precision == "fp16",
    )
    start_epoch, global_step, best, best_joint, history = 0, 0, math.inf, math.inf, []
    if resume_checkpoint is not None:
        checkpoint = resume_checkpoint
        (model.module if hasattr(model, "module") else model).load_state_dict(
            checkpoint["model"]
        )
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint.get("scaler", {}))
        start_epoch = int(checkpoint["epoch"]) + 1
        global_step = int(checkpoint.get("global_step", 0))
        history = list(checkpoint.get("history", []))
        primary_values = [
            float(record.get("validation", {}).get(
                "missing_track_validation_nll", math.inf,
            ))
            for record in history
        ]
        joint_values = [
            float(record.get("validation", {}).get("joint_validation_nll", math.inf))
            for record in history
        ]
        best = min(primary_values, default=math.inf)
        best_joint = min(joint_values, default=math.inf)
        selection_metric = checkpoint.get("selection_metric")
        checkpoint_best = float(
            checkpoint.get("best_validation_metric", math.inf)
        )
        if selection_metric == "missing_track_validation_nll":
            best = min(best, checkpoint_best)
        elif selection_metric == "joint_validation_nll":
            best_joint = min(best_joint, checkpoint_best)
        else:
            raise ValueError(
                f"Resume checkpoint has unknown selection metric {selection_metric!r}"
            )

    if rank == 0:
        startup = {
            "run_mode": config["run_mode"],
            "world_size": world,
            "device": str(device),
            "precision": precision,
            "train_views": len(train_data),
            "validation_views": len(val_data),
            "batch_per_rank": batch_size,
            "workers_per_rank": workers,
            "model": model_config.to_dict(),
            "loss_weights": weights,
            "missing_count_balance": provenance["missing_count_balance"],
            "missing_track_contract_sha256": resolved_contract["sha256"],
            "initialization": initialization,
        }
        print(json.dumps(startup, indent=2))
        (output / "training_contract.json").write_text(json.dumps({
            **startup,
            "corruption_contract": resolved_contract["contract"],
            "preprocessing_sha256": provenance["preprocessing_sha256"],
            "pid_target_sha256": provenance["pid_target_sha256"],
        }, indent=2) + "\n")
        priors = np.asarray(census["count_probabilities"], dtype=np.float64)
        (output / "constant_predictor_baseline.json").write_text(json.dumps({
            "count_probabilities": priors.tolist(),
            "count_cross_entropy": float(-(priors * np.log(priors)).sum()),
            "zero_missing_probability": float(priors[0]),
            "primary_score_for_constant_predictor": 0.0,
            "design_probabilities": list(resolved_contract["count_probabilities"]),
            "definition": (
                "train-schedule count predictor; primary log odds are corrected "
                "with the recorded fixed-schedule realized prior or the exact "
                "feasibility-averaged resampling prior"
            ),
            "prior_kind": census.get("prior_kind"),
            "schedule": census.get("schedule", "fixed"),
        }, indent=2) + "\n")

    patience = int(training.get("early_stopping_patience", 12))
    stale = 0
    for epoch in range(start_epoch, epochs):
        started = time.time()
        train_data.set_epoch(epoch)
        train_sampler.set_epoch(epoch)
        train_metrics = run_epoch(
            model, train_loader, device, epoch=epoch, seed=seed,
            precision=precision, weights=weights,
            pid_species_weights=pid_species_weights,
            missing_pid_species_weights=missing_pid_species_weights,
            missing_count_class_weights=missing_count_class_weights,
            optimizer=optimizer, scheduler=scheduler, scaler=scaler,
            grad_accumulation=accumulation,
            max_grad_norm=float(training.get("max_grad_norm", 1.0)),
            mask_fraction=float(training.get("mask_fraction", 0.15)),
            max_batches=args.max_batches,
        )
        global_step += math.ceil(steps_per_epoch / accumulation)
        val_data.set_epoch(0)
        with torch.no_grad():
            validation_metrics = run_epoch(
                model, val_loader, device, epoch=0, seed=seed + 1000003,
                precision=precision, weights=weights,
                pid_species_weights=pid_species_weights,
                missing_pid_species_weights=missing_pid_species_weights,
                missing_count_class_weights=missing_count_class_weights,
                mask_fraction=float(training.get("mask_fraction", 0.15)),
                max_batches=args.max_batches,
            )
        metric = float(validation_metrics["missing_track_validation_nll"])
        joint_metric = float(validation_metrics["joint_validation_nll"])
        record = {
            "epoch": epoch,
            "seconds": time.time() - started,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "selection_metric": "missing_track_validation_nll",
            "train": train_metrics,
            "validation": validation_metrics,
        }
        improved = metric < best - float(training.get("min_delta", 1e-4))
        if improved:
            best, stale = metric, 0
        else:
            stale += 1
        improved_joint = joint_metric < best_joint - float(training.get("min_delta", 1e-4))
        if improved_joint:
            best_joint = joint_metric
        if rank == 0:
            history.append(record)
            print(json.dumps(record))
            (output / "training_history.json").write_text(json.dumps(history, indent=2) + "\n")
            save_checkpoint(
                output / "last.pt", model, optimizer, scheduler, scaler,
                epoch=epoch, global_step=global_step, best=best, history=history,
                config=config, model_config=model_config, provenance=provenance,
            )
            if improved:
                save_checkpoint(
                    output / "best.pt", model, optimizer, scheduler, scaler,
                    epoch=epoch, global_step=global_step, best=best, history=history,
                    config=config, model_config=model_config, provenance=provenance,
                )
            if improved_joint:
                save_checkpoint(
                    output / "best_joint.pt", model, optimizer, scheduler, scaler,
                    epoch=epoch, global_step=global_step, best=best_joint,
                    history=history, config=config, model_config=model_config,
                    provenance=provenance, selection_metric="joint_validation_nll",
                )
        stop = torch.tensor(
            stale >= patience and bool(training.get("early_stopping", True)),
            device=device,
        )
        if dist.is_initialized():
            dist.broadcast(stop, 0)
        if stop.item():
            break
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
