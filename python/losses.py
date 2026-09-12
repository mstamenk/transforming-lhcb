"""Masked-PID and missing-particle objectives, including query matching."""
from __future__ import annotations
import itertools
import math
from typing import Any, Mapping, Sequence
import torch
from torch import distributed as dist
from torch.nn import functional as F
from dataset import pid_eligibility
from corruption import PARTICLE_TARGET_NAMES, TOPOLOGY_TARGET_NAMES, REMOVED_SUMMARY_NAMES
from runtime import nested, differentiable_zero, distributed_weighted_mean

MISSING_LOSS_NAMES = (
    "missing_count", "missing_count_balanced", "corruption_type",
    "removed_summary", "missing_set",
    "missing_topology", "missing_pid_natural", "missing_pid_balanced",
)

def compute_objective(
    outputs: Mapping[str, Any], batch: Mapping[str, torch.Tensor], pid_mask: torch.Tensor,
    weights: Mapping[str, float], flavour_class_weights: torch.Tensor | None = None,
    pid_species_class_weights: torch.Tensor | None = None,
    supervised_class_weights: Mapping[str, torch.Tensor] | None = None,
    *, synchronize_losses: bool = True,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    score, score_valid = masked_pid_score(outputs, batch)
    selected = pid_mask & pid_eligibility(batch) & score_valid
    losses: dict[str, torch.Tensor] = {}
    if weights.get("pid", 0) > 0:
        local_pid_sum = score[selected].sum() if selected.any() else differentiable_zero(outputs)
        losses["pid"] = distributed_weighted_mean(local_pid_sum, selected.sum(), synchronize=synchronize_losses)
    metrics = {
        "masked_particles": selected.sum().float(),
        "pid_nll_sum": score[selected].sum().detach(),
    }
    if "pid_logits" in outputs and "pid_class" in batch:
        target = batch["pid_class"].long(); valid = selected & (target >= 0)
        if valid.any():
            metrics["pid_class_count"] = valid.sum().float()
            prediction = outputs["pid_logits"].argmax(-1)[valid]
            metrics["pid_correct"] = (prediction == target[valid]).sum().float()
            n_classes = outputs["pid_logits"].shape[-1]
            confusion = torch.bincount(target[valid] * n_classes + prediction, minlength=n_classes*n_classes)
            for truth in range(n_classes):
                for predicted in range(n_classes): metrics[f"pid_confusion_{truth}_{predicted}"] = confusion[truth*n_classes+predicted].float()
    if weights.get("pid_species_balanced", 0) > 0:
        if pid_species_class_weights is None:
            raise ValueError("Balanced PID-species loss is enabled without resolved class weights")
        if "pid_balanced_logits" not in outputs or "pid_class" not in batch:
            raise ValueError("Balanced PID-species loss requires the auxiliary species head and pid_class")
        logits = outputs["pid_balanced_logits"]
        target = batch["pid_class"].long()
        n_classes = logits.shape[-1]
        class_weights = pid_species_class_weights.to(logits.device)
        in_range = (target >= 0) & (target < n_classes)
        target_index = target.clamp(0, n_classes - 1)
        valid = selected & in_range & (class_weights[target_index] > 0)
        per_particle = F.cross_entropy(
            logits.transpose(1, 2), target_index, reduction="none",
        )
        particle_weight = class_weights[target_index] * valid
        local_sum = (per_particle * particle_weight).sum()
        losses["pid_species_balanced"] = distributed_weighted_mean(
            local_sum, particle_weight.sum(), synchronize=synchronize_losses,
        )
        metrics["pid_balanced_nll_sum"] = local_sum.detach()
        metrics["pid_balanced_weight_sum"] = particle_weight.sum().detach()
        if valid.any():
            metrics["pid_balanced_class_count"] = valid.sum().float()
            prediction = logits.argmax(-1)[valid]
            metrics["pid_balanced_correct"] = (
                prediction == target[valid]
            ).sum().float()
            confusion = torch.bincount(
                target[valid] * n_classes + prediction,
                minlength=n_classes * n_classes,
            )
            for truth in range(n_classes):
                for predicted in range(n_classes):
                    metrics[f"pid_balanced_confusion_{truth}_{predicted}"] = (
                        confusion[truth * n_classes + predicted].float()
                    )
    if "pid_bin_logits" in outputs and "pid_bins" in batch:
        target_valid = batch.get("pid_target_valid", batch["pid_bins"] >= 0).bool()
        for channel, logits in enumerate(outputs["pid_bin_logits"]):
            target = batch["pid_bins"][..., channel].long(); valid = selected & target_valid[..., channel] & (target >= 0)
            if valid.any():
                metrics[f"pid_bin_count_{channel}"] = valid.sum().float()
                metrics[f"pid_bin_correct_{channel}"] = (logits.argmax(-1)[valid] == target[valid]).sum().float()
    weighted = [weights.get(name, 0.0) * loss for name, loss in losses.items() if weights.get(name, 0.0) > 0]
    total = sum(weighted, differentiable_zero(outputs))
    metrics.update({f"loss_{name}": loss.detach() for name, loss in losses.items()})
    metrics["loss"] = total.detach()
    return total, metrics

def missing_weights(config: Mapping[str, Any]) -> dict[str, float]:
    raw = nested(config, "training.loss_weights", {})
    if not isinstance(raw, Mapping):
        raise ValueError("training.loss_weights must be a mapping")
    names = ("pid", "pid_species_balanced", *MISSING_LOSS_NAMES)
    result = {name: float(raw.get(name, 0.0)) for name in names}
    if any(not math.isfinite(value) or value < 0 for value in result.values()):
        raise ValueError(f"Loss weights must be finite and nonnegative: {result}")
    for required in ("missing_set", "removed_summary", "corruption_type"):
        if result[required] <= 0:
            raise ValueError(f"Missing-track training requires a positive {required!r} loss")
    return result

def _gaussian_nll_sum(
    mean: torch.Tensor, log_scale: torch.Tensor, target: torch.Tensor,
) -> torch.Tensor:
    delta = (target - mean) * torch.exp(-log_scale)
    return (0.5 * delta.square() + log_scale + 0.5 * math.log(2.0 * math.pi)).sum()

def _missing_feature_validity(batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
    """Per-query continuous-target validity from the frozen target contract."""

    if "missing_particle_valid" not in batch:
        raise KeyError("Batch lacks exact missing_particle_valid target mask")
    valid = batch["missing_particle_valid"].bool()
    if valid.shape != batch["missing_particle_target"].shape:
        raise ValueError("missing_particle_valid shape differs from target shape")
    if (valid & ~batch["missing_set_valid"][..., None]).any():
        raise ValueError("A padded missing query has a valid continuous target")
    return valid

def _wrap_standardized_phi(delta: torch.Tensor, period: torch.Tensor) -> torch.Tensor:
    """Wrap a standardized delta-phi residual to its shortest signed branch."""

    return torch.remainder(delta + 0.5 * period, period) - 0.5 * period

def _query_cost(
    outputs: Mapping[str, torch.Tensor], batch: Mapping[str, torch.Tensor], *,
    include_missing_pid: bool = True, include_topology: bool = True,
) -> torch.Tensor:
    """Detached assignment cost with shape [batch, query, target-slot]."""

    mean = outputs["missing_query_features_mean"].float()
    target = batch["missing_particle_target"].float()
    expanded_mean = mean[:, :, None, :].expand(-1, -1, target.shape[1], -1)
    expanded_target = target[:, None, :, :].expand(-1, mean.shape[1], -1, -1)
    difference = expanded_target - expanded_mean
    period = batch["missing_delta_phi_period"].float()[:, None, None]
    difference[..., 3] = _wrap_standardized_phi(difference[..., 3], period)
    element_cost = F.smooth_l1_loss(
        difference, torch.zeros_like(difference), reduction="none",
    )
    feature_valid = _missing_feature_validity(batch)
    expanded_valid = feature_valid[:, None].expand_as(element_cost)
    cost = (element_cost * expanded_valid).sum(-1) / expanded_valid.sum(-1).clamp_min(1)
    pid_logits = (
        outputs.get("missing_pid_balanced_logits", outputs.get("missing_pid_logits"))
        if include_missing_pid else None
    )
    if pid_logits is not None:
        log_prob = pid_logits.float().log_softmax(-1)
        pid = batch["missing_pid_target"].clamp_min(0)
        pid_cost = -log_prob[:, :, None, :].expand(-1, -1, pid.shape[1], -1).gather(
            -1, pid[:, None, :, None].expand(-1, log_prob.shape[1], -1, 1),
        ).squeeze(-1)
        cost = cost + 0.5 * pid_cost
    if "missing_charge_logits" in outputs:
        log_prob = outputs["missing_charge_logits"].float().log_softmax(-1)
        charge = batch["missing_charge_target"].clamp_min(0)
        charge_cost = -log_prob[:, :, None, :].expand(-1, -1, charge.shape[1], -1).gather(
            -1, charge[:, None, :, None].expand(-1, log_prob.shape[1], -1, 1),
        ).squeeze(-1)
        cost = cost + 0.25 * charge_cost
    if include_topology and "missing_topology_prediction" in outputs:
        prediction = outputs["missing_topology_prediction"].float()
        topology = batch["missing_topology_target"].float()
        topology_cost = F.binary_cross_entropy_with_logits(
            prediction[:, :, None, :].expand(-1, -1, topology.shape[1], -1),
            topology[:, None, :, :].expand(-1, prediction.shape[1], -1, -1),
            reduction="none",
        ).mean(-1)
        cost = cost + 0.25 * topology_cost
    return cost.detach()

def match_missing_queries(
    outputs: Mapping[str, torch.Tensor], batch: Mapping[str, torch.Tensor],
    *, include_missing_pid: bool = True, include_topology: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Permutation-invariant exact matching for at most four query slots.

    Returns ``query_for_target [B,Q]`` (unused targets are -1) and the binary
    existence target for every query.  Exhaustive permutations are cheaper and
    avoid a CPU synchronization/Scipy dependency at Q<=4.
    """

    cost = _query_cost(
        outputs, batch, include_missing_pid=include_missing_pid,
        include_topology=include_topology,
    )
    bsz, n_queries, n_targets = cost.shape
    if n_queries != n_targets:
        raise ValueError(f"Query/target slot mismatch: {n_queries} vs {n_targets}")
    counts = batch["missing_count_target"].long()
    query_for_target = torch.full((bsz, n_targets), -1, device=cost.device, dtype=torch.long)
    existence = torch.zeros((bsz, n_queries), device=cost.device, dtype=torch.float32)
    for k in range(1, n_targets + 1):
        rows = torch.nonzero(counts == k, as_tuple=False).flatten()
        if not len(rows):
            continue
        permutations = list(itertools.permutations(range(n_queries), k))
        candidate_costs = []
        selected_cost = cost.index_select(0, rows)
        for permutation in permutations:
            candidate_costs.append(sum(
                selected_cost[:, query, target_index]
                for target_index, query in enumerate(permutation)
            ))
        best = torch.stack(candidate_costs, 1).argmin(1)
        permutation_tensor = torch.tensor(permutations, device=cost.device, dtype=torch.long)
        chosen = permutation_tensor.index_select(0, best)
        query_for_target[rows, :k] = chosen
        existence[rows[:, None], chosen] = 1.0
    if not torch.equal(existence.sum(1).long(), counts):
        raise RuntimeError("Hungarian-style query matching lost a missing object")
    return query_for_target, existence

def _gather_matched(
    prediction: torch.Tensor, query_for_target: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    valid = query_for_target >= 0
    rows, targets = torch.nonzero(valid, as_tuple=True)
    queries = query_for_target[rows, targets]
    return prediction[rows, queries], valid

def _weighted_mean(
    numerator: torch.Tensor, denominator: torch.Tensor | int,
    *, synchronize: bool,
) -> torch.Tensor:
    if not isinstance(denominator, torch.Tensor):
        denominator = torch.tensor(float(denominator), device=numerator.device)
    return distributed_weighted_mean(numerator, denominator, synchronize=synchronize)

def compute_missing_objective(
    outputs: Mapping[str, Any], batch: Mapping[str, Any], weights: Mapping[str, float],
    missing_pid_species_weights: torch.Tensor,
    missing_count_class_weights: torch.Tensor | None, *, synchronize: bool,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    residual = batch["residual"]
    metrics: dict[str, torch.Tensor] = {}
    losses: dict[str, torch.Tensor] = {}
    zero = differentiable_zero(outputs)

    count_target = batch["missing_count_target"].long()
    count_per = F.cross_entropy(outputs["missing_count_logits"], count_target, reduction="none")
    count_sum = count_per.sum()
    losses["missing_count"] = _weighted_mean(
        count_sum, count_target.numel(), synchronize=synchronize,
    )
    metrics["missing_count_nll_sum"] = count_sum.detach()
    metrics["missing_count_count"] = torch.tensor(count_target.numel(), device=count_sum.device, dtype=torch.float32)
    count_prediction = outputs["missing_count_logits"].argmax(-1)
    metrics["missing_count_correct"] = (count_prediction == count_target).sum().float()
    n_count = outputs["missing_count_logits"].shape[-1]
    confusion = torch.bincount(count_target * n_count + count_prediction, minlength=n_count * n_count)
    for truth in range(n_count):
        for predicted in range(n_count):
            metrics[f"missing_count_confusion_{truth}_{predicted}"] = confusion[truth*n_count+predicted].float()

    if weights.get("missing_count_balanced", 0) > 0:
        if missing_count_class_weights is None:
            raise ValueError(
                "missing_count_balanced requires exact train-census class weights"
            )
        balanced_logits = outputs.get("missing_count_balanced_logits")
        if balanced_logits is None:
            raise KeyError(
                "Active missing_count_balanced loss requires its auxiliary head"
            )
        class_weights = missing_count_class_weights.to(count_per.device)
        if class_weights.numel() != balanced_logits.shape[-1]:
            raise ValueError(
                "Missing-count class-weight length differs from balanced logits"
            )
        balanced_per = F.cross_entropy(
            balanced_logits, count_target, reduction="none",
        )
        count_event_weight = class_weights[count_target]
        balanced_sum = (balanced_per * count_event_weight).sum()
        balanced_weight_sum = count_event_weight.sum()
        losses["missing_count_balanced"] = _weighted_mean(
            balanced_sum, balanced_weight_sum, synchronize=synchronize,
        )
        balanced_prediction = balanced_logits.argmax(-1)
        metrics["missing_count_balanced_nll_sum"] = balanced_sum.detach()
        metrics["missing_count_balanced_weight_sum"] = (
            balanced_weight_sum.detach()
        )
        metrics["missing_count_balanced_correct"] = (
            balanced_prediction == count_target
        ).sum().float()
        balanced_confusion = torch.bincount(
            count_target * n_count + balanced_prediction,
            minlength=n_count * n_count,
        )
        for truth in range(n_count):
            for predicted in range(n_count):
                metrics[
                    f"missing_count_balanced_confusion_{truth}_{predicted}"
                ] = balanced_confusion[truth*n_count+predicted].float()

    corruption_target = batch["corruption_type_target"].long()
    corruption_per = F.cross_entropy(outputs["corruption_type_logits"], corruption_target, reduction="none")
    corruption_sum = corruption_per.sum()
    losses["corruption_type"] = _weighted_mean(corruption_sum, corruption_target.numel(), synchronize=synchronize)
    metrics["corruption_type_nll_sum"] = corruption_sum.detach()
    metrics["corruption_type_count"] = torch.tensor(corruption_target.numel(), device=count_sum.device, dtype=torch.float32)
    metrics["corruption_type_correct"] = (outputs["corruption_type_logits"].argmax(-1) == corruption_target).sum().float()

    positive = count_target > 0
    if positive.any():
        summary_mean = outputs["removed_summary_mean"][positive]
        summary_scale = outputs["removed_summary_log_scale"][positive]
        summary_target = batch["removed_summary_target"][positive].float()
        summary_sum = _gaussian_nll_sum(summary_mean, summary_scale, summary_target)
        summary_count = torch.tensor(summary_target.numel(), device=count_sum.device, dtype=torch.float32)
    else:
        summary_sum = zero
        summary_count = torch.zeros((), device=count_sum.device)
    losses["removed_summary"] = _weighted_mean(summary_sum, summary_count, synchronize=synchronize)
    metrics["removed_summary_nll_sum"] = summary_sum.detach()
    metrics["removed_summary_count"] = summary_count.detach()

    query_for_target, existence_target = match_missing_queries(
        outputs, batch,
        include_missing_pid=(
            weights.get("missing_pid_natural", 0.0) > 0
            or weights.get("missing_pid_balanced", 0.0) > 0
        ),
        include_topology=weights.get("missing_topology", 0.0) > 0,
    )
    existence_logits = outputs["missing_query_existence_logits"]
    existence_sum = F.binary_cross_entropy_with_logits(
        existence_logits, existence_target, reduction="sum",
    )
    existence_count = torch.tensor(existence_target.numel(), device=count_sum.device, dtype=torch.float32)
    metrics["missing_existence_nll_sum"] = existence_sum.detach()
    metrics["missing_existence_count"] = existence_count
    metrics["missing_existence_correct"] = (
        (existence_logits >= 0) == existence_target.bool()
    ).sum().float()

    matched_mean, matched_valid = _gather_matched(outputs["missing_query_features_mean"], query_for_target)
    matched_scale, _ = _gather_matched(outputs["missing_query_features_log_scale"], query_for_target)
    target_features = batch["missing_particle_target"][matched_valid].float()
    target_feature_valid = _missing_feature_validity(batch)[matched_valid]
    if len(target_features):
        feature_delta = target_features - matched_mean
        matched_period = batch["missing_delta_phi_period"].float()[:, None].expand_as(
            matched_valid
        )[matched_valid]
        feature_delta[:, 3] = _wrap_standardized_phi(
            feature_delta[:, 3], matched_period,
        )
        feature_delta = feature_delta * torch.exp(-matched_scale)
        feature_element = (
            0.5 * feature_delta.square() + matched_scale
            + 0.5 * math.log(2.0 * math.pi)
        )
        feature_sum = feature_element[target_feature_valid].sum()
        feature_count = target_feature_valid.sum().float()
    else:
        feature_sum, feature_count = zero, torch.zeros((), device=count_sum.device)
    metrics["missing_feature_nll_sum"] = feature_sum.detach()
    metrics["missing_feature_count"] = feature_count

    matched_charge, _ = _gather_matched(outputs["missing_charge_logits"], query_for_target)
    target_charge = batch["missing_charge_target"][matched_valid].long()
    if len(target_charge):
        charge_sum = F.cross_entropy(matched_charge, target_charge, reduction="sum")
        charge_count = torch.tensor(target_charge.numel(), device=count_sum.device, dtype=torch.float32)
        metrics["missing_charge_correct"] = (matched_charge.argmax(-1) == target_charge).sum().float()
    else:
        charge_sum, charge_count = zero, torch.zeros((), device=count_sum.device)
    metrics["missing_charge_nll_sum"] = charge_sum.detach()
    metrics["missing_charge_count"] = charge_count

    set_terms = (
        _weighted_mean(existence_sum, existence_count, synchronize=synchronize),
        _weighted_mean(feature_sum, feature_count, synchronize=synchronize),
        _weighted_mean(charge_sum, charge_count, synchronize=synchronize),
    )
    losses["missing_set"] = torch.stack(set_terms).mean()

    matched_topology, _ = _gather_matched(outputs["missing_topology_prediction"], query_for_target)
    topology_target = batch["missing_topology_target"][matched_valid].float()
    if len(topology_target):
        topology_sum = F.binary_cross_entropy_with_logits(matched_topology, topology_target, reduction="sum")
        topology_count = torch.tensor(topology_target.numel(), device=count_sum.device, dtype=torch.float32)
    else:
        topology_sum, topology_count = zero, torch.zeros((), device=count_sum.device)
    losses["missing_topology"] = _weighted_mean(topology_sum, topology_count, synchronize=synchronize)
    metrics["missing_topology_nll_sum"] = topology_sum.detach()
    metrics["missing_topology_count"] = topology_count

    pid_target = batch["missing_pid_target"][matched_valid].long()
    if len(pid_target) and (pid_target < 0).any():
        raise ValueError("Every selected missing query must have a signed PID target")
    for name, key, weighted in (
        ("missing_pid_natural", "missing_pid_logits", False),
        ("missing_pid_balanced", "missing_pid_balanced_logits", True),
    ):
        if weights.get(name, 0) <= 0:
            continue
        matched_logits, _ = _gather_matched(outputs[key], query_for_target)
        if len(pid_target):
            per = F.cross_entropy(matched_logits, pid_target, reduction="none")
            event_weight = (
                missing_pid_species_weights.to(per.device)[pid_target]
                if weighted else torch.ones_like(per)
            )
            local_sum = (per * event_weight).sum()
            local_count = event_weight.sum()
            metrics[f"{name}_correct"] = (matched_logits.argmax(-1) == pid_target).sum().float()
        else:
            local_sum, local_count = zero, torch.zeros((), device=count_sum.device)
        losses[name] = _weighted_mean(local_sum, local_count, synchronize=synchronize)
        metrics[f"{name}_nll_sum"] = local_sum.detach()
        metrics[f"{name}_count"] = local_count.detach()

    total = sum(
        (weights.get(name, 0.0) * value for name, value in losses.items()),
        zero,
    )
    metrics.update({f"loss_{name}": value.detach() for name, value in losses.items()})
    metrics["missing_track_loss"] = total.detach()
    return total, metrics


def particle_pid_components(
    outputs: Mapping[str, Any],
    batch: Mapping[str, torch.Tensor],
    normalizers: Mapping[str, Any] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return species NLL/validity and summed response NLL/count."""
    shape = batch["particle_mask"].shape
    species_score = torch.zeros(shape, device=batch["particle_mask"].device)
    species_valid = torch.zeros(shape, dtype=torch.bool, device=batch["particle_mask"].device)
    response_score = torch.zeros(shape, device=batch["particle_mask"].device)
    response_count = torch.zeros_like(response_score)
    species_normalizer = max(float((normalizers or {}).get("species", 1.0)), 1e-6)
    response_normalizers = list((normalizers or {}).get("responses", []))
    if "pid_logits" in outputs and "pid_class" in batch:
        target = batch["pid_class"].long(); valid = target >= 0
        value = F.cross_entropy(outputs["pid_logits"].transpose(1, 2), target.clamp_min(0), reduction="none")
        species_score += value * valid / species_normalizer; species_valid |= valid
    if "pid_bin_logits" in outputs and "pid_bins" in batch:
        valid_channels = batch.get("pid_target_valid", batch["pid_bins"] >= 0)
        for channel, logits in enumerate(outputs["pid_bin_logits"]):
            target = batch["pid_bins"][..., channel].long(); valid = valid_channels[..., channel] & (target >= 0)
            value = F.cross_entropy(logits.transpose(1, 2), target.clamp_min(0), reduction="none")
            scale = max(float(response_normalizers[channel]), 1e-6) if channel < len(response_normalizers) else 1.0
            response_score += value * valid / scale; response_count += valid
    if "pid_mean" in outputs and "pid_values" in batch:
        valid_channels = batch.get("pid_target_valid", torch.isfinite(batch["pid_values"]))
        delta = (batch["pid_values"] - outputs["pid_mean"]) * torch.exp(-outputs["pid_log_scale"])
        values = 0.5 * delta.square() + outputs["pid_log_scale"] + 0.5 * torch.log(torch.tensor(2 * torch.pi, device=response_score.device))
        response_score += (values * valid_channels).sum(-1); response_count += valid_channels.sum(-1)
    return species_score, species_valid, response_score, response_count

def masked_pid_score(
    outputs: Mapping[str, Any],
    batch: Mapping[str, torch.Tensor],
    *,
    species_weight: float = 0.5,
    response_weight: float = 0.5,
    normalizers: Mapping[str, Any] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-particle balanced PID score and a valid-target indicator."""
    species, species_valid, response_sum, response_count = particle_pid_components(outputs, batch, normalizers)
    response_valid = response_count > 0
    response = response_sum / response_count.clamp_min(1)
    numerator = species_weight * species * species_valid + response_weight * response * response_valid
    denominator = species_weight * species_valid + response_weight * response_valid
    return numerator / denominator.clamp_min(1e-6), denominator > 0
