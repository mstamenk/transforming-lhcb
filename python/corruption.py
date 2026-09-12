#!/usr/bin/env python3
"""Leakage-safe, deterministic missing-track corruptions for reconstructed jets.

The production signed-PID shards already contain a compact float32 kinematic
bank in ``pairwise_inputs``.  This module uses that bank to construct paired
clean/residual views on the fly, avoiding a second materialized copy of the ML
dataset.  Removed particles are compacted away completely: there is no mask
placeholder, empty slot, original multiplicity, or intervention flag in the
residual model input.

Only reconstructed quantities are targets.  Generator flavour and source
sample labels are never consulted when choosing or describing a corruption.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import tempfile
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import yaml
from torch.utils.data import Dataset

from dataset import JetShardDataset, collate_jets


MASK64 = (1 << 64) - 1
FORMAT_VERSION = 2

PARTICLE_TARGET_NAMES = (
    "log_pt_fraction",
    "log_p_fraction",
    "delta_eta",
    "delta_phi",
    "signed_log_ip",
    "log1p_ipchi2",
)
TOPOLOGY_TARGET_NAMES = (
    "in_displaced_vertex",
    "in_cascade",
    "track_available",
)
REMOVED_SUMMARY_NAMES = (
    "log1p_removed_energy_over_residual_energy",
    "removed_pt_parallel_over_residual_pt",
    "removed_pt_perp_over_residual_pt",
    "removed_pz_over_residual_p",
    "log1p_removed_massless_mass_gev",
    "removed_charge_over_4",
    "removed_charged_count_over_4",
    "removed_neutral_count_over_4",
    "removed_displaced_count_over_4",
    "log1p_dropped_stored_vertex_tokens",
    "log1p_dropped_stored_cascade_tokens",
)
CORRUPTION_TYPE_NAMES = (
    "complete",
    "random_charged",
    "random_neutral",
    "opposite_sign_pair",
    "angular_local_group",
    "tracks_from_displaced_vertex",
    "tracks_from_cascade",
)

# Particle feature indices in the signed_pid_v1 contract.
PF_LOG_PT = 0
PF_LOG_P = 1
PF_LOG_PT_FRACTION = 2
PF_LOG_P_FRACTION = 3
PF_DELTA_ETA = 4
PF_DELTA_PHI = 5
PF_SIGNED_LOG_IP = 7
PF_LOG1P_IPCHI2 = 9

# pairwise_inputs is an intentionally unstandardized float32 particle bank.
PW_DELTA_ETA = 0
PW_DELTA_PHI = 1
PW_LOG_P = 2
PW_CHARGE = 3
PW_STATE_Z = 6
PW_DIRECTION_X = 7
PW_DIRECTION_Y = 8
PW_DIRECTION_Z = 9
PW_TRACK_AVAILABLE = 10

GLOBAL_LOG_JET_PT = 0
GLOBAL_JET_ETA = 1
GLOBAL_LOG_MASSLESS_MASS = 2
GLOBAL_LOG_N_PARTICLES = 3
GLOBAL_CHARGED_FRACTION = 4
GLOBAL_JET_WIDTH = 5
GLOBAL_N_PVS = 6
GLOBAL_LOG_N_DISPLACED = 7
GLOBAL_LOG_N_PAIR_VERTICES = 8
GLOBAL_LOG_N_TRIPLET_VERTICES = 9
GLOBAL_LOG_N_QUAD_VERTICES = 10
GLOBAL_LOG_N_CHAINS = 11
GLOBAL_VERTICES_TRUNCATED = 12
GLOBAL_CHAINS_TRUNCATED = 13

PARTICLE_AXIS_KEYS = {
    "particle_features",
    "particle_feature_valid",
    "pairwise_inputs",
    "pid_features",
    "particle_mask",
    "pid_class",
    "pid_bins",
    "pid_values",
    "particle_origin",
    "particle_targets",
    "pid_target_valid",
    "pid_eligible",
    "particle_charge",
    "particle_original_index",
    "target_reco_id",
    "pid_available",
}


def _splitmix64(value: int) -> int:
    value = (int(value) + 0x9E3779B97F4A7C15) & MASK64
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & MASK64
    return (value ^ (value >> 31)) & MASK64


def _uniform(value: int) -> float:
    return (_splitmix64(value) >> 11) / float(1 << 53)


def _wrap_phi(value: torch.Tensor) -> torch.Tensor:
    return torch.remainder(value + math.pi, 2.0 * math.pi) - math.pi


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(descriptor, "w") as handle:
            json.dump(payload, handle, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


@dataclass(frozen=True)
class MissingTrackCorruptionConfig:
    """Configuration for deterministic reconstructed-track deletion."""

    seed: int = 20260814
    views_per_jet: int = 1
    max_missing: int = 4
    min_survivors: int = 1
    # K=0,1,2,3,4 is sampled independently of the corruption mechanism.
    count_probabilities: tuple[float, ...] = (0.25, 0.35, 0.25, 0.10, 0.05)
    # ``complete`` is selected only by K=0 and therefore has zero mechanism
    # probability.  The remaining six probabilities sum to one.
    type_probabilities: tuple[float, ...] = (0.0, 0.25, 0.10, 0.20, 0.15, 0.18, 0.12)
    min_displaced_track_pt_mev: float = 250.0
    min_displaced_track_ipchi2: float = 4.0
    max_abs_state_z_mm: float = 500.0
    topology_mode: str = "compact_unaffected"
    sanitize_jet_width: bool = True
    resample_each_epoch: bool = True

    def __post_init__(self) -> None:
        if self.views_per_jet <= 0:
            raise ValueError("views_per_jet must be positive")
        if not 1 <= self.max_missing <= 4:
            raise ValueError("max_missing must be between one and four")
        if self.min_survivors < 1:
            raise ValueError("min_survivors must be at least one")
        if len(self.count_probabilities) != self.max_missing + 1:
            raise ValueError(
                "count_probabilities must contain K=0 through max_missing"
            )
        if any(value < 0 for value in self.count_probabilities):
            raise ValueError("count probabilities cannot be negative")
        if not math.isclose(sum(self.count_probabilities), 1.0, abs_tol=1e-9):
            raise ValueError("count probabilities must sum to one")
        if len(self.type_probabilities) != len(CORRUPTION_TYPE_NAMES):
            raise ValueError(
                f"type_probabilities needs {len(CORRUPTION_TYPE_NAMES)} entries"
            )
        if any(value < 0 for value in self.type_probabilities):
            raise ValueError("type probabilities cannot be negative")
        if self.type_probabilities[0] != 0:
            raise ValueError("complete must have zero mechanism probability")
        if not math.isclose(sum(self.type_probabilities[1:]), 1.0, abs_tol=1e-9):
            raise ValueError("non-complete type probabilities must sum to one")
        if self.topology_mode not in {"compact_unaffected", "drop_all"}:
            raise ValueError("topology_mode must be compact_unaffected or drop_all")
        if not self.sanitize_jet_width:
            raise ValueError(
                "Jet width cannot be reproduced safely; sanitize_jet_width must remain true"
            )

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "MissingTrackCorruptionConfig":
        # The production transformer YAML also records human-readable target
        # orders and rebuild provenance alongside the executable corruption
        # settings.  Ignore those metadata-only keys here; the trainer validates
        # them separately against :func:`corruption_contract`.
        values = dict(payload)
        if "max_removed" in values and "max_missing" not in values:
            values["max_missing"] = values.pop("max_removed")
        if values.get("topology_mode") == "drop_affected":
            values["topology_mode"] = "compact_unaffected"
        allowed = {item.name for item in fields(cls)}
        values = {key: value for key, value in values.items() if key in allowed}
        if "count_probabilities" in values:
            probabilities = values["count_probabilities"]
            if isinstance(probabilities, Mapping):
                maximum = int(values.get("max_missing", cls.max_missing))
                values["count_probabilities"] = tuple(
                    float(probabilities[str(index)] if str(index) in probabilities else probabilities[index])
                    for index in range(maximum + 1)
                )
            else:
                values["count_probabilities"] = tuple(float(x) for x in probabilities)
        if "type_probabilities" in values:
            probabilities = values["type_probabilities"]
            if isinstance(probabilities, Mapping):
                values["type_probabilities"] = tuple(
                    float(probabilities[name]) for name in CORRUPTION_TYPE_NAMES
                )
            else:
                values["type_probabilities"] = tuple(float(x) for x in probabilities)
        return cls(**values)


def corruption_contract(config: MissingTrackCorruptionConfig) -> dict[str, Any]:
    """Machine-readable interface consumed by the trainer and evaluator."""

    return {
        "format_version": FORMAT_VERSION,
        "objective": "reconstructed missing-track denoising",
        "reconstructed_only": True,
        "generator_labels_used": False,
        "genuine_token_deletion": True,
        "residual_has_mask_placeholder": False,
        "config": asdict(config),
        "corruption_type_names": list(CORRUPTION_TYPE_NAMES),
        "type_sampling": {
            "definition": (
                "configured type probabilities are preference weights, "
                "renormalized over mechanisms feasible for the selected jet and K"
            ),
            "feasibility_recorded_as": "corruption_feasible_types",
            "no_silent_fallback": True,
            "topology_mechanisms": (
                "select K deterministic members from a stored reconstructed "
                "vertex/cascade with at least K supported constituents"
            ),
        },
        "removed_summary_names": list(REMOVED_SUMMARY_NAMES),
        "missing_particle_target_names": list(PARTICLE_TARGET_NAMES),
        "missing_particle_validity": {
            "tensor": "missing_particle_valid",
            "kinematics": "defined for every selected physical constituent",
            "signed_log_ip_and_log1p_ipchi2": (
                "exact source particle_feature_valid columns 7 and 9"
            ),
        },
        "periodic_targets": {
            "delta_phi": (
                "wrapped raw angle standardized without robust clipping so one "
                "complete 2*pi period is retained; encoder inputs remain clipped"
            ),
        },
        "displaced_track_definition": (
            "track available, valid measured log1p_ipchi2, configured pT/IPchi2/z cuts"
        ),
        "missing_topology_target_names": list(TOPOLOGY_TARGET_NAMES),
        "missing_charge_encoding": {"0": "negative", "1": "neutral", "2": "positive"},
        "missing_pid_encoding": "signed reconstructed-ID class from source manifest",
        "missing_pid_validity": (
            "all selected missing constituents require pid_class >= 0; padded queries use -100"
        ),
        "topology_policy": (
            "retain and compact only stored vertex/cascade tokens with zero incidence "
            "to deleted tracks; discard cascades incident on discarded vertices"
            if config.topology_mode == "compact_unaffected"
            else "clear all vertex/cascade tokens in both clean and residual views"
        ),
        "global_policy": {
            "recomputed": [
                "log_jet_pt",
                "jet_eta",
                "log_massless_jet_mass",
                "log1p_n_particles",
                "charged_fraction",
                "log1p_n_displaced_tracks",
                "log1p_n_pair_vertices",
                "log1p_n_triplet_vertices",
                "log1p_n_quad_vertices",
                "log1p_n_chains",
            ],
            "preserved_event_context": ["n_pvs", "polarity"],
            "sanitized_to_training_center": [
                "jet_width", "vertices_truncated", "chains_truncated"
            ],
            "global_feature_valid_false": [
                "jet_width", "vertices_truncated", "chains_truncated"
            ],
        },
        "kinematic_source": (
            "float32 pairwise_inputs: log1p(p), track directions, and relative eta/phi; "
            "neutral directions use the reconstructed jet axis inferred from track directions"
        ),
        "energy_definition": "massless E = |p|; stored Daughters_E is not used",
        "limitations": [
            "Jet clustering cannot be rerun because particles outside the stored jet are absent.",
            "The source jet-width definition is undocumented and is removed from both views.",
            "Only stored, capped topology candidates can be retained; hidden pre-cap candidates cannot be recovered online.",
            "Track state z is relative to the PV in the compact bank, so displaced-track eligibility uses the documented relative-z proxy.",
        ],
    }


class MissingTrackCorruptionDataset(Dataset):
    """Paired clean/residual views generated deterministically from a jet dataset."""

    def __init__(
        self,
        source: str | Path | Sequence[str | Path] | JetShardDataset,
        *,
        config: MissingTrackCorruptionConfig | None = None,
        cache_size: int = 1,
        include_clean: bool = True,
    ) -> None:
        self.base = (
            source
            if isinstance(source, JetShardDataset)
            else JetShardDataset(source, cache_size=cache_size)
        )
        self.config = config or MissingTrackCorruptionConfig()
        self.include_clean = bool(include_clean)
        self.epoch = 0
        self.metadata = copy.deepcopy(self.base.metadata)
        self.metadata["missing_track_corruption"] = corruption_contract(self.config)
        self._validate_contract()

        particle_scaler = self.metadata["robust_scalers"]["particle_features"]
        global_scaler = self.metadata["robust_scalers"]["global_features"]
        self.particle_center = torch.tensor(particle_scaler["center"], dtype=torch.float32)
        self.particle_scale = torch.tensor(particle_scaler["scale"], dtype=torch.float32)
        self.particle_clip = float(particle_scaler["clip_standardized"])
        self.delta_phi_period = float(
            2.0 * math.pi / self.particle_scale[PF_DELTA_PHI].item()
        )
        self.global_center = torch.tensor(global_scaler["center"], dtype=torch.float32)
        self.global_scale = torch.tensor(global_scaler["scale"], dtype=torch.float32)
        self.global_clip = float(global_scaler["clip_standardized"])

    def _validate_contract(self) -> None:
        feature_names = self.metadata.get("feature_names", {})
        expected_particle = self.metadata["robust_scalers"]["particle_features"]["names"]
        expected_global = self.metadata["robust_scalers"]["global_features"]["names"]
        if list(feature_names.get("particle", ())) != list(expected_particle):
            raise ValueError("Particle feature names and scaler order differ")
        if list(feature_names.get("global", ())) != list(expected_global):
            raise ValueError("Global feature names and scaler order differ")
        required_particle = {
            "log_pt_fraction": PF_LOG_PT_FRACTION,
            "log_p_fraction": PF_LOG_P_FRACTION,
            "delta_eta": PF_DELTA_ETA,
            "delta_phi": PF_DELTA_PHI,
            "signed_log_ip": PF_SIGNED_LOG_IP,
            "log1p_ipchi2": PF_LOG1P_IPCHI2,
        }
        for name, index in required_particle.items():
            if expected_particle[index] != name:
                raise ValueError(
                    f"Particle contract drift at index {index}: expected {name}, "
                    f"found {expected_particle[index]}"
                )
        if len(expected_global) != 14 or expected_global[GLOBAL_JET_WIDTH] != "jet_width":
            raise ValueError("Global feature contract is incompatible with corruption rebuild")
        pairwise = list(feature_names.get("pairwise_inputs", ()))
        expected_pairwise = [
            "delta_eta", "delta_phi", "log_p", "charge", "state_x", "state_y",
            "state_z", "direction_x", "direction_y", "direction_z", "track_available",
        ]
        if pairwise != expected_pairwise:
            raise ValueError(f"pairwise_inputs contract drift: {pairwise}")

    def __len__(self) -> int:
        return len(self.base) * self.config.views_per_jet

    @property
    def lengths(self) -> list[int]:
        return [length * self.config.views_per_jet for length in self.base.lengths]

    @property
    def offsets(self) -> list[int]:
        return np.cumsum([0, *self.lengths], dtype=np.int64).tolist()

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _base_location(self, index: int) -> tuple[int, int]:
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        return index // self.config.views_per_jet, index % self.config.views_per_jet

    def _scaler_raw(self, values: torch.Tensor, center: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        return values.float() * scale + center

    def _standardize_particle_column(self, raw: torch.Tensor, index: int) -> torch.Tensor:
        return torch.clamp(
            (raw - self.particle_center[index]) / self.particle_scale[index],
            -self.particle_clip,
            self.particle_clip,
        )

    def _standardize_global(self, raw: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        result = torch.zeros_like(raw, dtype=torch.float32)
        result[valid] = torch.clamp(
            (raw[valid] - self.global_center[valid]) / self.global_scale[valid],
            -self.global_clip,
            self.global_clip,
        )
        return result

    def _particle_vectors(self, sample: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """Recover massless momenta from the float32 compact kinematic bank."""

        pairwise = sample["pairwise_inputs"].float()
        valid = sample["particle_mask"].bool()
        momentum = torch.expm1(pairwise[:, PW_LOG_P]).clamp_min(0.0)
        direction = pairwise[:, PW_DIRECTION_X:PW_DIRECTION_Z + 1]
        norm = torch.linalg.vector_norm(direction, dim=-1)
        track = valid & (pairwise[:, PW_TRACK_AVAILABLE] > 0.5) & (norm > 1e-6)

        # Track directions are absolute.  Infer the original jet direction from
        # direction minus the stored relative coordinates, then use it for
        # neutral constituents whose track direction is intentionally absent.
        stored_eta = self._scaler_raw(
            sample["global_features"][GLOBAL_JET_ETA],
            self.global_center[GLOBAL_JET_ETA],
            self.global_scale[GLOBAL_JET_ETA],
        )
        jet_eta = stored_eta
        jet_phi = torch.tensor(0.0, dtype=torch.float32)
        if track.any():
            transverse_direction = torch.linalg.vector_norm(direction[:, :2], dim=-1).clamp_min(1e-8)
            track_eta = torch.asinh(direction[:, 2] / transverse_direction)
            track_phi = torch.atan2(direction[:, 1], direction[:, 0])
            inferred_eta = track_eta - pairwise[:, PW_DELTA_ETA]
            inferred_phi = _wrap_phi(track_phi - pairwise[:, PW_DELTA_PHI])
            weight = torch.where(track, momentum * transverse_direction, 0.0)
            weight_sum = weight.sum().clamp_min(1e-8)
            jet_eta = (weight * inferred_eta).sum() / weight_sum
            jet_phi = torch.atan2(
                (weight * torch.sin(inferred_phi)).sum(),
                (weight * torch.cos(inferred_phi)).sum(),
            )

        absolute_eta = jet_eta + pairwise[:, PW_DELTA_ETA]
        absolute_phi = _wrap_phi(jet_phi + pairwise[:, PW_DELTA_PHI])
        pt = momentum / torch.cosh(absolute_eta).clamp_min(1e-8)
        reconstructed = torch.stack(
            (pt * torch.cos(absolute_phi), pt * torch.sin(absolute_phi), pt * torch.sinh(absolute_eta)),
            dim=-1,
        )
        exact_tracks = momentum[:, None] * direction / norm.clamp_min(1e-8)[:, None]
        reconstructed[track] = exact_tracks[track]
        reconstructed[~valid] = 0.0
        if not torch.isfinite(reconstructed).all():
            raise ValueError("Non-finite momentum reconstructed from pairwise_inputs")
        return reconstructed

    def _displaced_track_mask(
        self, sample: Mapping[str, torch.Tensor], momenta: torch.Tensor
    ) -> torch.Tensor:
        pairwise = sample["pairwise_inputs"].float()
        valid = sample["particle_mask"].bool()
        if "particle_feature_valid" not in sample:
            raise KeyError(
                "Displaced-track reconstruction requires particle_feature_valid"
            )
        ipchi2_valid = sample["particle_feature_valid"][
            :, PF_LOG1P_IPCHI2
        ].bool()
        pt = torch.linalg.vector_norm(momenta[:, :2], dim=-1)
        raw_ipchi2 = self._scaler_raw(
            sample["particle_features"][:, PF_LOG1P_IPCHI2],
            self.particle_center[PF_LOG1P_IPCHI2],
            self.particle_scale[PF_LOG1P_IPCHI2],
        )
        ipchi2 = torch.expm1(raw_ipchi2).clamp_min(0.0)
        return (
            valid
            & (pairwise[:, PW_TRACK_AVAILABLE] > 0.5)
            & ipchi2_valid
            & (pt >= self.config.min_displaced_track_pt_mev)
            & (ipchi2 >= self.config.min_displaced_track_ipchi2)
            & (pairwise[:, PW_STATE_Z].abs() <= self.config.max_abs_state_z_mm)
        )

    def _seed(self, sample: Mapping[str, torch.Tensor], view: int) -> int:
        identity = int(sample.get("jet_uid", sample["sample_index"]).item()) & MASK64
        epoch = self.epoch if self.config.resample_each_epoch else 0
        value = identity ^ (int(self.config.seed) & MASK64)
        value ^= ((int(view) + 1) * 0xD6E8FEB86659FD93) & MASK64
        value ^= ((int(epoch) + 1) * 0xA5A3564E27F8862B) & MASK64
        return _splitmix64(value)

    @staticmethod
    def _choose_from_probabilities(probabilities: Sequence[float], seed: int) -> int:
        value = _uniform(seed)
        cumulative = 0.0
        for index, probability in enumerate(probabilities):
            cumulative += probability
            if value < cumulative:
                return index
        return len(probabilities) - 1

    def _choose_count(self, seed: int) -> int:
        return self._choose_from_probabilities(
            self.config.count_probabilities, seed ^ 0x4CF5AD432745937F
        )

    def _choose_feasible_count(self, seed: int, maximum: int) -> int:
        """Draw K from the configured prior restricted to this physical jet.

        Very low-multiplicity jets cannot support every configured K while
        retaining ``min_survivors``.  Renormalizing the allowed classes avoids
        the old many-to-one ``min(requested, maximum)`` distortion.  The
        realized marginal is measured by the audit/validator and is not
        claimed to be exactly the design prior.
        """

        probabilities = self.config.count_probabilities[: maximum + 1]
        total = float(sum(probabilities))
        if total <= 0:
            raise ValueError("No positive missing-count probability is feasible")
        return self._choose_from_probabilities(
            [float(value) / total for value in probabilities],
            seed ^ 0x4CF5AD432745937F,
        )

    def _choose_feasible_type(
        self, feasible: Mapping[int, list[int]], seed: int,
    ) -> tuple[int, list[int]]:
        """Choose among mechanisms that genuinely exist for this jet and K."""

        mechanisms = sorted(feasible)
        weights = [float(self.config.type_probabilities[index]) for index in mechanisms]
        total = sum(weights)
        if total <= 0:
            raise ValueError(f"Feasible corruption mechanisms have zero weight: {mechanisms}")
        choice = self._choose_from_probabilities(
            [value / total for value in weights], seed ^ 0xDB4F0B9175AE2165,
        )
        mechanism = mechanisms[choice]
        return mechanism, feasible[mechanism]

    def _ranked_positions(
        self, sample: Mapping[str, torch.Tensor], positions: Sequence[int], seed: int
    ) -> list[int]:
        original = sample.get("particle_original_index")
        return sorted(
            (int(position) for position in positions),
            key=lambda position: _splitmix64(
                seed
                ^ (((int(original[position].item()) if original is not None else position) + 1)
                   * 0x9E3779B185EBCA87)
            ),
        )

    def _topology_groups(
        self,
        sample: Mapping[str, torch.Tensor],
        kind: str,
        eligible: torch.Tensor,
        count: int,
    ) -> list[tuple[int, ...]]:
        key = "vertex_track_mask" if kind == "vertex" else "candidate_track_mask"
        valid_key = "vertex_mask" if kind == "vertex" else "candidate_mask"
        if key not in sample or valid_key not in sample:
            return []
        incidence = sample[key].bool()
        object_valid = sample[valid_key].bool()
        physical_count = int(sample["particle_mask"].sum().item())
        groups: set[tuple[int, ...]] = set()
        for row in torch.nonzero(object_valid, as_tuple=False).flatten().tolist():
            members = tuple(
                int(value)
                for value in torch.nonzero(incidence[row] & eligible, as_tuple=False).flatten().tolist()
            )
            all_members = tuple(
                int(value)
                for value in torch.nonzero(incidence[row], as_tuple=False).flatten().tolist()
            )
            if (
                members == all_members
                and len(members) >= count
                and physical_count - count >= self.config.min_survivors
            ):
                groups.add(tuple(sorted(members)))
        return sorted(groups)

    def _angular_group(
        self,
        sample: Mapping[str, torch.Tensor],
        momenta: torch.Tensor,
        positions: Sequence[int],
        count: int,
        seed: int,
    ) -> list[int]:
        if len(positions) < count:
            return []
        ranked = self._ranked_positions(sample, positions, seed ^ 0x94D049BB133111EB)
        anchor = ranked[0]
        pt = torch.linalg.vector_norm(momenta[:, :2], dim=-1).clamp_min(1e-8)
        eta = torch.asinh(momenta[:, 2] / pt)
        phi = torch.atan2(momenta[:, 1], momenta[:, 0])
        original = sample.get("particle_original_index")

        def distance(position: int) -> tuple[float, int]:
            delta_eta = float(eta[position] - eta[anchor])
            delta_phi = float(_wrap_phi(phi[position] - phi[anchor]))
            identity = int(original[position].item()) if original is not None else position
            tie = _splitmix64(seed ^ ((identity + 1) * 0xBF58476D1CE4E5B9))
            return delta_eta * delta_eta + delta_phi * delta_phi, tie

        return sorted((int(value) for value in positions), key=distance)[:count]

    def _select_mechanism(
        self,
        sample: Mapping[str, torch.Tensor],
        momenta: torch.Tensor,
        mechanism: int,
        count: int,
        supported: torch.Tensor,
        seed: int,
    ) -> list[int]:
        valid = sample["particle_mask"].bool()
        pairwise = sample["pairwise_inputs"].float()
        charge = sample["particle_charge"].long()
        track = pairwise[:, PW_TRACK_AVAILABLE] > 0.5
        charged = supported & (charge != 0) & track
        neutral = supported & (charge == 0)
        if mechanism == 1:
            positions = torch.nonzero(charged, as_tuple=False).flatten().tolist()
            if len(positions) >= count:
                return self._ranked_positions(sample, positions, seed)[:count]
        elif mechanism == 2:
            positions = torch.nonzero(neutral, as_tuple=False).flatten().tolist()
            if len(positions) >= count:
                return self._ranked_positions(sample, positions, seed)[:count]
        elif mechanism == 3 and count == 2:
            positive = torch.nonzero(charged & (charge > 0), as_tuple=False).flatten().tolist()
            negative = torch.nonzero(charged & (charge < 0), as_tuple=False).flatten().tolist()
            pairs = [(plus, minus) for plus in positive for minus in negative]
            if pairs:
                pairs.sort(
                    key=lambda pair: _splitmix64(
                        seed
                        ^ ((pair[0] + 1) * 0x9E3779B185EBCA87)
                        ^ ((pair[1] + 1) * 0xC2B2AE3D27D4EB4F)
                    )
                )
                return sorted(pairs[0])
        elif mechanism == 4:
            positions = torch.nonzero(supported, as_tuple=False).flatten().tolist()
            return self._angular_group(sample, momenta, positions, count, seed)
        elif mechanism in {5, 6}:
            kind = "vertex" if mechanism == 5 else "cascade"
            groups = self._topology_groups(sample, kind, supported, count)
            if groups:
                chosen = groups[_splitmix64(seed ^ 0xC2B2AE3D27D4EB4F) % len(groups)]
                return self._ranked_positions(
                    sample, chosen, seed ^ 0x27D4EB2F165667C5,
                )[:count]
        return []

    def _select_removed(
        self, sample: Mapping[str, torch.Tensor], momenta: torch.Tensor, seed: int
    ) -> tuple[int, list[int], int, torch.Tensor]:
        valid = sample["particle_mask"].bool()
        # The first production contract requires a defined signed-PID target
        # for every removed constituent.  This is reconstructed-ID metadata,
        # never generator information.
        supported = valid & (sample["pid_class"].long() >= 0)
        maximum = min(
            self.config.max_missing,
            max(0, int(valid.sum().item()) - self.config.min_survivors),
            int(supported.sum().item()),
        )
        if maximum <= 0:
            feasible_mask = torch.zeros(len(CORRUPTION_TYPE_NAMES), dtype=torch.bool)
            feasible_mask[0] = True
            return 0, [], 0, feasible_mask
        count = self._choose_feasible_count(seed, maximum)
        if count == 0:
            feasible_mask = torch.zeros(len(CORRUPTION_TYPE_NAMES), dtype=torch.bool)
            feasible_mask[0] = True
            return 0, [], maximum, feasible_mask
        feasible: dict[int, list[int]] = {}
        for mechanism in range(1, len(CORRUPTION_TYPE_NAMES)):
            selected = self._select_mechanism(
                sample,
                momenta,
                mechanism,
                count,
                supported,
                seed ^ (mechanism * 0x85EBCA77C2B2AE63),
            )
            if len(selected) == count:
                feasible[mechanism] = selected
        # Angular grouping is feasible whenever ``maximum >= count``; reaching
        # this branch therefore signals a broken reconstructed-particle
        # contract rather than a reason to relabel the example as complete.
        if not feasible:
            raise RuntimeError("No corruption mechanism is feasible for a positive K")
        mechanism, selected = self._choose_feasible_type(feasible, seed)
        feasible_mask = torch.zeros(len(CORRUPTION_TYPE_NAMES), dtype=torch.bool)
        feasible_mask[list(feasible)] = True
        return mechanism, selected, maximum, feasible_mask

    def _topology_selection(
        self, sample: Mapping[str, torch.Tensor], removed: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        n_particles = len(removed)
        vertex_valid = sample.get("vertex_mask", torch.zeros(0, dtype=torch.bool)).bool()
        candidate_valid = sample.get("candidate_mask", torch.zeros(0, dtype=torch.bool)).bool()
        vertex_affected = torch.zeros_like(vertex_valid)
        candidate_affected = torch.zeros_like(candidate_valid)
        if "vertex_track_mask" in sample and len(vertex_valid):
            vertex_affected = sample["vertex_track_mask"].bool()[:, :n_particles][:, removed].any(1)
        if "candidate_track_mask" in sample and len(candidate_valid):
            candidate_affected |= sample["candidate_track_mask"].bool()[:, :n_particles][:, removed].any(1)
        if "candidate_vertex_mask" in sample and len(candidate_valid) and len(vertex_valid):
            candidate_affected |= (
                sample["candidate_vertex_mask"].bool()[:, :len(vertex_valid)]
                & vertex_affected[None, :]
            ).any(1)
        if self.config.topology_mode == "drop_all":
            # Encoder policy and target definition are deliberately separate:
            # all aggregate tokens are hidden in both views, while the summary
            # counts only aggregates actually incident on a removed particle.
            return (
                torch.zeros_like(vertex_valid),
                torch.zeros_like(candidate_valid),
                vertex_valid & vertex_affected,
                candidate_valid & candidate_affected,
            )
        keep_vertex = vertex_valid & ~vertex_affected
        if "candidate_vertex_mask" in sample and len(candidate_valid) and len(vertex_valid):
            candidate_affected |= (
                sample["candidate_vertex_mask"].bool()[:, :len(vertex_valid)]
                & ~keep_vertex[None, :]
            ).any(1)
        keep_candidate = candidate_valid & ~candidate_affected
        return keep_vertex, keep_candidate, vertex_valid & ~keep_vertex, candidate_valid & ~keep_candidate

    @staticmethod
    def _empty_like_first(value: torch.Tensor, size: int, fill: float | int | bool = 0) -> torch.Tensor:
        shape = (size, *value.shape[1:])
        return torch.full(shape, fill, dtype=value.dtype)

    @staticmethod
    def _particle_fill(key: str, value: torch.Tensor) -> float | int | bool:
        if value.dtype == torch.bool:
            return False
        if key in {"particle_original_index", "pid_class", "pid_bins"}:
            return -1
        return 0

    def _recenter_particle_features(
        self,
        base: torch.Tensor,
        momenta: torch.Tensor,
        jet_vector: torch.Tensor,
        jet_energy: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        result = base.float().clone()
        momentum = torch.linalg.vector_norm(momenta, dim=-1)
        pt = torch.linalg.vector_norm(momenta[:, :2], dim=-1)
        eta = torch.asinh(momenta[:, 2] / pt.clamp_min(1e-8))
        phi = torch.atan2(momenta[:, 1], momenta[:, 0])
        jet_pt = torch.linalg.vector_norm(jet_vector[:2]).clamp_min(1e-8)
        jet_p = torch.linalg.vector_norm(jet_vector).clamp_min(1e-8)
        jet_eta = torch.asinh(jet_vector[2] / jet_pt)
        jet_phi = torch.atan2(jet_vector[1], jet_vector[0])
        raw_columns = {
            PF_LOG_PT: torch.log1p(pt.clamp_min(0.0)),
            PF_LOG_P: torch.log1p(momentum.clamp_min(0.0)),
            PF_LOG_PT_FRACTION: torch.log((pt / jet_pt).clamp_min(1e-8)),
            PF_LOG_P_FRACTION: torch.log((momentum / jet_p).clamp_min(1e-8)),
            PF_DELTA_ETA: eta - jet_eta,
            PF_DELTA_PHI: _wrap_phi(phi - jet_phi),
        }
        for index, raw in raw_columns.items():
            result[:, index] = self._standardize_particle_column(raw, index)
        pairwise = torch.zeros((len(momenta), 11), dtype=torch.float32)
        pairwise[:, PW_DELTA_ETA] = raw_columns[PF_DELTA_ETA]
        pairwise[:, PW_DELTA_PHI] = raw_columns[PF_DELTA_PHI]
        pairwise[:, PW_LOG_P] = raw_columns[PF_LOG_P]
        return result, pairwise

    def _build_view(
        self,
        sample: Mapping[str, torch.Tensor],
        all_momenta: torch.Tensor,
        keep_particle: torch.Tensor,
        keep_vertex: torch.Tensor,
        keep_candidate: torch.Tensor,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        output = {key: value.clone() for key, value in sample.items()}
        n_particles = len(sample["particle_mask"])
        particle_positions = torch.nonzero(keep_particle, as_tuple=False).flatten()
        compact_momenta = all_momenta[particle_positions]
        jet_vector = compact_momenta.sum(0)
        jet_energy = torch.linalg.vector_norm(compact_momenta, dim=-1).sum()
        if len(compact_momenta) == 0:
            raise ValueError("A residual view cannot contain zero particles")

        base_features = sample["particle_features"][particle_positions]
        recomputed_features, recomputed_pairwise = self._recenter_particle_features(
            base_features, compact_momenta, jet_vector, jet_energy
        )
        original_pairwise = sample["pairwise_inputs"][particle_positions].float()
        recomputed_pairwise[:, PW_CHARGE:] = original_pairwise[:, PW_CHARGE:]

        for key in PARTICLE_AXIS_KEYS:
            if key not in sample:
                continue
            value = sample[key]
            packed = self._empty_like_first(
                value, n_particles, self._particle_fill(key, value)
            )
            selected = value[particle_positions]
            if key == "particle_features":
                selected = recomputed_features.to(value.dtype)
            elif key == "pairwise_inputs":
                selected = recomputed_pairwise.to(value.dtype)
            packed[:len(selected)] = selected
            output[key] = packed
        output["particle_mask"] = torch.arange(n_particles) < len(particle_positions)

        vertex_positions = torch.nonzero(keep_vertex, as_tuple=False).flatten()
        candidate_positions = torch.nonzero(keep_candidate, as_tuple=False).flatten()
        n_vertices = len(sample.get("vertex_mask", torch.zeros(0)))
        n_candidates = len(sample.get("candidate_mask", torch.zeros(0)))

        if "vertex_features" in sample:
            packed = self._empty_like_first(sample["vertex_features"], n_vertices)
            packed[:len(vertex_positions)] = sample["vertex_features"][vertex_positions]
            output["vertex_features"] = packed
            output["vertex_mask"] = torch.arange(n_vertices) < len(vertex_positions)
        if "candidate_features" in sample:
            packed = self._empty_like_first(sample["candidate_features"], n_candidates)
            packed[:len(candidate_positions)] = sample["candidate_features"][candidate_positions]
            output["candidate_features"] = packed
            output["candidate_mask"] = torch.arange(n_candidates) < len(candidate_positions)

        if "vertex_track_mask" in sample:
            incidence = torch.zeros((n_vertices, n_particles), dtype=torch.bool)
            incidence[:len(vertex_positions), :len(particle_positions)] = (
                sample["vertex_track_mask"][vertex_positions][:, particle_positions]
            )
            output["vertex_track_mask"] = incidence
        if "candidate_track_mask" in sample:
            incidence = torch.zeros((n_candidates, n_particles), dtype=torch.bool)
            incidence[:len(candidate_positions), :len(particle_positions)] = (
                sample["candidate_track_mask"][candidate_positions][:, particle_positions]
            )
            output["candidate_track_mask"] = incidence
        if "candidate_vertex_mask" in sample:
            incidence = torch.zeros((n_candidates, n_vertices), dtype=torch.bool)
            incidence[:len(candidate_positions), :len(vertex_positions)] = (
                sample["candidate_vertex_mask"][candidate_positions][:, vertex_positions]
            )
            output["candidate_vertex_mask"] = incidence
        if "pairwise_features" in sample:
            packed = torch.zeros(
                (n_particles, n_particles, *sample["pairwise_features"].shape[2:]),
                dtype=sample["pairwise_features"].dtype,
            )
            selected = sample["pairwise_features"][particle_positions][:, particle_positions]
            packed[:len(particle_positions), :len(particle_positions)] = selected
            output["pairwise_features"] = packed

        displaced = self._displaced_track_mask(sample, all_momenta) & keep_particle
        vertex_membership = (
            output.get("vertex_track_mask", torch.zeros((0, n_particles), dtype=torch.bool))
            [output.get("vertex_mask", torch.zeros(0, dtype=torch.bool))]
        )
        vertex_sizes = vertex_membership.sum(1) if len(vertex_membership) else torch.zeros(0)
        pair_count = int((vertex_sizes == 2).sum().item())
        triplet_count = int((vertex_sizes == 3).sum().item())
        quad_count = int((vertex_sizes == 4).sum().item())
        chain_count = int(output.get("candidate_mask", torch.zeros(0)).sum().item())

        jet_pt = torch.linalg.vector_norm(jet_vector[:2])
        jet_p = torch.linalg.vector_norm(jet_vector)
        jet_eta = torch.asinh(jet_vector[2] / jet_pt.clamp_min(1e-8))
        mass2 = jet_energy.square() - jet_p.square()
        mass = torch.sqrt(mass2.clamp_min(0.0))
        charge = sample["particle_charge"][particle_positions]
        n_physical = len(particle_positions)
        raw_global = torch.zeros(14, dtype=torch.float32)
        raw_global[GLOBAL_LOG_JET_PT] = torch.log1p(jet_pt)
        raw_global[GLOBAL_JET_ETA] = jet_eta
        raw_global[GLOBAL_LOG_MASSLESS_MASS] = torch.log1p(mass)
        raw_global[GLOBAL_LOG_N_PARTICLES] = math.log1p(n_physical)
        raw_global[GLOBAL_CHARGED_FRACTION] = float((charge != 0).sum()) / n_physical
        raw_global[GLOBAL_N_PVS] = torch.round(self._scaler_raw(
            sample["global_features"][GLOBAL_N_PVS],
            self.global_center[GLOBAL_N_PVS],
            self.global_scale[GLOBAL_N_PVS],
        ))
        raw_global[GLOBAL_LOG_N_DISPLACED] = math.log1p(int(displaced.sum().item()))
        raw_global[GLOBAL_LOG_N_PAIR_VERTICES] = math.log1p(pair_count)
        raw_global[GLOBAL_LOG_N_TRIPLET_VERTICES] = math.log1p(triplet_count)
        raw_global[GLOBAL_LOG_N_QUAD_VERTICES] = math.log1p(quad_count)
        raw_global[GLOBAL_LOG_N_CHAINS] = math.log1p(chain_count)
        raw_global[GLOBAL_VERTICES_TRUNCATED] = 0.0
        raw_global[GLOBAL_CHAINS_TRUNCATED] = 0.0
        global_valid = torch.ones(14, dtype=torch.bool)
        global_valid[GLOBAL_JET_WIDTH] = False
        global_valid[GLOBAL_VERTICES_TRUNCATED] = False
        global_valid[GLOBAL_CHAINS_TRUNCATED] = False
        output["global_features"] = self._standardize_global(raw_global, global_valid)
        output["global_feature_valid"] = global_valid
        output["n_particles_input"] = torch.tensor(n_physical, dtype=torch.float32)
        output["n_particles_stored"] = torch.tensor(n_physical, dtype=torch.float32)
        output["n_track_lines"] = torch.tensor(int(displaced.sum()), dtype=torch.float32)
        output["n_pair_total"] = torch.tensor(pair_count, dtype=torch.float32)
        output["n_triplet_total"] = torch.tensor(triplet_count, dtype=torch.float32)
        output["n_quad_total"] = torch.tensor(quad_count, dtype=torch.float32)
        output["n_chain_total"] = torch.tensor(chain_count, dtype=torch.float32)
        output["particles_truncated"] = torch.tensor(0.0)
        output["vertices_truncated"] = torch.tensor(0.0)
        output["chains_truncated"] = torch.tensor(0.0)

        context = {
            "jet_vector": jet_vector,
            "jet_energy": jet_energy,
            "particle_positions": particle_positions,
            "displaced": displaced,
            "pair_count": torch.tensor(pair_count),
            "triplet_count": torch.tensor(triplet_count),
            "quad_count": torch.tensor(quad_count),
            "chain_count": torch.tensor(chain_count),
        }
        return output, context

    def build_exact_views(
        self,
        sample: Mapping[str, torch.Tensor],
        removed_positions: Sequence[int],
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """Build aligned complete/residual views for an externally frozen deletion.

        Candidate-removal studies choose tracks with reconstructed selections
        fixed before inference.  This public adapter deliberately reuses the
        production corruption rebuild instead of duplicating its compact
        momentum, coordinate, topology, and global-feature policy.  Positions
        refer to the compact particle axis; ``particle_original_index`` remains
        the stable link to the source ROOT daughter index.
        """
        physical = sample["particle_mask"].bool()
        positions = [int(value) for value in removed_positions]
        if not positions or len(positions) > self.config.max_missing:
            raise ValueError(
                "Exact removal needs between one and max_missing particle positions"
            )
        if len(set(positions)) != len(positions):
            raise ValueError("Exact removal positions must be unique")
        if any(value < 0 or value >= len(physical) or not physical[value] for value in positions):
            raise ValueError("Exact removal selected a padded or out-of-range particle")
        if int(physical.sum().item()) - len(positions) < self.config.min_survivors:
            raise ValueError("Exact removal leaves fewer than min_survivors particles")

        removed = torch.zeros_like(physical)
        removed[positions] = True
        momenta = self._particle_vectors(sample)
        keep_vertex, keep_candidate, _, _ = self._topology_selection(sample, removed)
        residual, _ = self._build_view(
            sample, momenta, physical & ~removed, keep_vertex, keep_candidate,
        )

        clean_removed = torch.zeros_like(removed)
        clean_vertex, clean_candidate, _, _ = self._topology_selection(
            sample, clean_removed,
        )
        clean, _ = self._build_view(
            sample, momenta, physical, clean_vertex, clean_candidate,
        )
        return clean, residual

    def _targets(
        self,
        sample: Mapping[str, torch.Tensor],
        momenta: torch.Tensor,
        removed_positions: list[int],
        corruption_type: int,
        feasible_max: int,
        feasible_types: torch.Tensor,
        residual_context: Mapping[str, torch.Tensor],
        dropped_vertices: torch.Tensor,
        dropped_candidates: torch.Tensor,
        seed: int,
        view: int,
    ) -> dict[str, torch.Tensor]:
        maximum = self.config.max_missing
        missing_set_valid = torch.zeros(maximum, dtype=torch.bool)
        missing_particle_target = torch.zeros((maximum, len(PARTICLE_TARGET_NAMES)), dtype=torch.float32)
        missing_particle_valid = torch.zeros((maximum, len(PARTICLE_TARGET_NAMES)), dtype=torch.bool)
        missing_pid_target = torch.full((maximum,), -100, dtype=torch.long)
        missing_charge_target = torch.full((maximum,), -100, dtype=torch.long)
        missing_topology_target = torch.zeros((maximum, len(TOPOLOGY_TARGET_NAMES)), dtype=torch.float32)
        missing_original_index = torch.full((maximum,), -1, dtype=torch.long)

        removed = torch.tensor(removed_positions, dtype=torch.long)
        k = len(removed_positions)
        if k:
            missing_set_valid[:k] = True
            residual_vector = residual_context["jet_vector"]
            residual_energy = residual_context["jet_energy"].clamp_min(1e-8)
            residual_pt = torch.linalg.vector_norm(residual_vector[:2]).clamp_min(1e-8)
            residual_p = torch.linalg.vector_norm(residual_vector).clamp_min(1e-8)
            residual_phi = torch.atan2(residual_vector[1], residual_vector[0])

            removed_vectors = momenta[removed]
            removed_energy_each = torch.linalg.vector_norm(removed_vectors, dim=-1)
            removed_vector = removed_vectors.sum(0)
            removed_energy = removed_energy_each.sum()
            parallel = removed_vector[0] * torch.cos(residual_phi) + removed_vector[1] * torch.sin(residual_phi)
            perpendicular = -removed_vector[0] * torch.sin(residual_phi) + removed_vector[1] * torch.cos(residual_phi)
            removed_mass2 = removed_energy.square() - removed_vector.square().sum()
            removed_mass_gev = torch.sqrt(removed_mass2.clamp_min(0.0)) / 1000.0

            removed_charge = sample["particle_charge"][removed].long()
            displaced = self._displaced_track_mask(sample, momenta)
            in_vertex = (
                sample.get("vertex_track_mask", torch.zeros((0, len(momenta)), dtype=torch.bool))
                [sample.get("vertex_mask", torch.zeros(0, dtype=torch.bool))]
            )
            in_candidate = (
                sample.get("candidate_track_mask", torch.zeros((0, len(momenta)), dtype=torch.bool))
                [sample.get("candidate_mask", torch.zeros(0, dtype=torch.bool))]
            )
            per_track_vertex = in_vertex[:, removed].any(0) if len(in_vertex) else torch.zeros(k, dtype=torch.bool)
            per_track_candidate = in_candidate[:, removed].any(0) if len(in_candidate) else torch.zeros(k, dtype=torch.bool)
            track_available = sample["pairwise_inputs"][removed, PW_TRACK_AVAILABLE] > 0.5

            # Recompute the missing queries in the residual coordinate system.
            target_features, _ = self._recenter_particle_features(
                sample["particle_features"][removed],
                removed_vectors,
                residual_vector,
                residual_energy,
            )
            feature_indices = (
                PF_LOG_PT_FRACTION,
                PF_LOG_P_FRACTION,
                PF_DELTA_ETA,
                PF_DELTA_PHI,
                PF_SIGNED_LOG_IP,
                PF_LOG1P_IPCHI2,
            )
            missing_particle_target[:k] = target_features[:, feature_indices]
            # The encoder's robustly standardized delta-phi is clipped, but a
            # circular likelihood target must retain one complete 2*pi branch.
            # Otherwise values close to +pi and -pi collapse to +/-clip and the
            # shortest-period residual can no longer reconnect the boundary.
            removed_phi = torch.atan2(removed_vectors[:, 1], removed_vectors[:, 0])
            raw_delta_phi = _wrap_phi(removed_phi - residual_phi)
            missing_particle_target[:k, 3] = (
                raw_delta_phi - self.particle_center[PF_DELTA_PHI]
            ) / self.particle_scale[PF_DELTA_PHI]
            # Kinematic targets are defined for every physical constituent.
            # The two track-geometry targets retain their exact source feature
            # validity rather than treating standardized zero as a measurement.
            missing_particle_valid[:k, :4] = True
            source_feature_valid = sample["particle_feature_valid"][removed].bool()
            missing_particle_valid[:k, 4] = source_feature_valid[:, PF_SIGNED_LOG_IP]
            missing_particle_valid[:k, 5] = source_feature_valid[:, PF_LOG1P_IPCHI2]
            missing_pid_target[:k] = sample["pid_class"][removed].long()
            missing_charge_target[:k] = removed_charge.clamp(-1, 1) + 1
            missing_topology_target[:k] = torch.stack(
                (per_track_vertex.float(), per_track_candidate.float(), track_available.float()),
                dim=-1,
            )
            if "particle_original_index" in sample:
                missing_original_index[:k] = sample["particle_original_index"][removed].long()

            charged_count = int((removed_charge != 0).sum().item())
            neutral_count = k - charged_count
            summary = torch.stack(
                (
                    torch.log1p(removed_energy / residual_energy),
                    parallel / residual_pt,
                    perpendicular / residual_pt,
                    removed_vector[2] / residual_p,
                    torch.log1p(removed_mass_gev),
                    removed_charge.sum().float() / 4.0,
                    torch.tensor(charged_count / 4.0),
                    torch.tensor(neutral_count / 4.0),
                    displaced[removed].sum().float() / 4.0,
                    torch.log1p(dropped_vertices.sum().float()),
                    torch.log1p(dropped_candidates.sum().float()),
                )
            ).float()
        else:
            summary = torch.zeros(len(REMOVED_SUMMARY_NAMES), dtype=torch.float32)

        targets = {
            "missing_count_target": torch.tensor(k, dtype=torch.long),
            "corruption_type_target": torch.tensor(corruption_type, dtype=torch.long),
            "corruption_feasible_max": torch.tensor(feasible_max, dtype=torch.long),
            "corruption_feasible_types": feasible_types.bool().clone(),
            "removed_summary_target": summary,
            "missing_set_valid": missing_set_valid,
            "missing_particle_target": missing_particle_target,
            "missing_particle_valid": missing_particle_valid,
            "missing_delta_phi_period": torch.tensor(
                self.delta_phi_period, dtype=torch.float32,
            ),
            "missing_pid_target": missing_pid_target,
            "missing_charge_target": missing_charge_target,
            "missing_topology_target": missing_topology_target,
            "missing_original_index": missing_original_index,
            "corruption_seed": torch.tensor(seed & 0x7FFFFFFFFFFFFFFF, dtype=torch.long),
            "corruption_view": torch.tensor(view, dtype=torch.long),
            "corruption_epoch": torch.tensor(
                self.epoch if self.config.resample_each_epoch else 0, dtype=torch.long
            ),
        }
        for key, value in targets.items():
            if value.is_floating_point() and not torch.isfinite(value).all():
                raise ValueError(f"Non-finite corruption target {key}")
        return targets

    def __getitem__(self, index: int) -> dict[str, Any]:
        base_index, view = self._base_location(index)
        sample = self.base[base_index]
        momenta = self._particle_vectors(sample)
        seed = self._seed(sample, view)
        corruption_type, removed_positions, feasible_max, feasible_types = (
            self._select_removed(sample, momenta, seed)
        )
        removed_mask = torch.zeros_like(sample["particle_mask"], dtype=torch.bool)
        if removed_positions:
            removed_mask[removed_positions] = True
        keep_particle = sample["particle_mask"].bool() & ~removed_mask

        keep_vertex, keep_candidate, dropped_vertices, dropped_candidates = (
            self._topology_selection(sample, removed_mask)
        )
        residual, residual_context = self._build_view(
            sample, momenta, keep_particle, keep_vertex, keep_candidate
        )
        targets = self._targets(
            sample,
            momenta,
            removed_positions,
            corruption_type,
            feasible_max,
            feasible_types,
            residual_context,
            dropped_vertices,
            dropped_candidates,
            seed,
            view,
        )
        result = {"residual": residual, **targets}
        if self.include_clean:
            clean_removed = torch.zeros_like(removed_mask)
            clean_vertex, clean_candidate, _, _ = self._topology_selection(
                sample, clean_removed,
            )
            clean, _ = self._build_view(
                sample,
                momenta,
                sample["particle_mask"].bool(),
                clean_vertex,
                clean_candidate,
            )
            result["clean"] = clean
        return result

    def selection_targets(self, index: int) -> dict[str, torch.Tensor]:
        """Return only deterministic corruption labels for a fast train census.

        This deliberately skips view rebuilding.  With the production frozen
        corruption schedule, the resulting class counts are the exact
        train-only priors used for missing-PID balancing and score correction.
        """

        base_index, view = self._base_location(index)
        sample = self.base[base_index]
        momenta = self._particle_vectors(sample)
        seed = self._seed(sample, view)
        corruption_type, positions, feasible_max, feasible_types = self._select_removed(
            sample, momenta, seed,
        )
        pid_counts = torch.zeros(
            len(self.metadata.get("pid_class_order", ())), dtype=torch.long,
        )
        if positions:
            target = sample["pid_class"][torch.tensor(positions, dtype=torch.long)].long()
            if (target < 0).any() or (target >= len(pid_counts)).any():
                raise ValueError("Selection census encountered an invalid signed PID class")
            pid_counts += torch.bincount(target, minlength=len(pid_counts))
        return {
            "missing_count": torch.tensor(len(positions), dtype=torch.long),
            "corruption_type": torch.tensor(corruption_type, dtype=torch.long),
            "feasible_max": torch.tensor(feasible_max, dtype=torch.long),
            "feasible_types": feasible_types,
            "missing_pid_counts": pid_counts,
        }


class MissingTrackSelectionDataset(Dataset):
    """Cheap target-only adapter used once before optimization."""

    def __init__(self, source: MissingTrackCorruptionDataset) -> None:
        self.source = source

    def __len__(self) -> int:
        return len(self.source)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return self.source.selection_targets(index)


def collate_missing_track_pairs(samples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Collate nested paired views and fixed-shape corruption targets."""

    if not samples:
        raise ValueError("Cannot collate an empty missing-track batch")
    has_clean = "clean" in samples[0]
    if any(("clean" in sample) != has_clean for sample in samples):
        raise KeyError("A missing-track batch mixes paired and residual-only rows")
    result: dict[str, Any] = {
        "residual": collate_jets([sample["residual"] for sample in samples]),
    }
    if has_clean:
        result["clean"] = collate_jets([sample["clean"] for sample in samples])
    target_keys = set(samples[0]) - {"clean", "residual"}
    for key in sorted(target_keys):
        if any(key not in sample for sample in samples):
            raise KeyError(f"Paired batch lacks target key {key}")
        result[key] = torch.stack([sample[key] for sample in samples])
    return result

