#!/usr/bin/env python3
"""Sharded datasets and batching for the heterogeneous masked-PID model.

The GPU-facing format is a manifest of ``.npz`` or ``.pt`` shards. Every
array's first axis is the jet index. Arrays may be fixed-size or per-jet object
arrays; :func:`collate_jets` pads particle, vertex, and candidate axes.
"""

from __future__ import annotations

import bisect
import glob
import json
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from pid_targets import (
    SIGNED_RECO_PID_TARGETS,
    signed_reco_pid_classes,
    validate_signed_reco_pid_metadata,
)


ALIASES = {
    "particles": "particle_features", "particle_valid": "particle_mask",
    "particle_valid_mask": "particle_mask", "valid_particle_mask": "particle_mask",
    "pid": "pid_features", "vertices": "vertex_features",
    "vertex_valid": "vertex_mask", "candidates": "candidate_features",
    "cascade_features": "candidate_features", "cascade_valid": "candidate_mask",
    "globals": "global_features", "jet_features": "global_features",
    "flavour": "jet_flavour", "flavor": "jet_flavour",
    "jet_uid": "sample_index",
}
MASK_KEYS = {"particle_mask", "particle_feature_valid", "vertex_mask", "candidate_mask", "pid_eligible", "pid_available", "pid_target_valid"}
INDEX_KEYS = {'raw_entry', 'jet_index', 'candidate_class', 'particle_charge', 'pid_class', 'run_number', 'target_reco_id', 'vertex_class', 'split', 'polarity', 'sample_index', 'source_entry', 'source_id', 'particle_original_index', 'pid_bins', 'sample_id', 'particle_origin', 'event_uid', 'jet_flavour', 'event_id', 'event_number'}
INCIDENCE_KEYS = {"vertex_track_mask", "candidate_track_mask", "candidate_vertex_mask"}
POLARITY_VALUES = (-1, 1)


def _canonical(key: str) -> str:
    return ALIASES.get(key, key)


def validate_feature_contract(metadata: Mapping[str, Any]) -> None:
    """Fail closed when manifest metadata advertises target leakage.

    Vertex/cascade token inputs are intentionally geometry-only. Masses and PID
    hypotheses remain analysis outputs and must never become encoder features.
    """
    groups = metadata.get("feature_names", metadata.get("features", {}))
    if not isinstance(groups, Mapping): return
    particle = [str(v).lower() for v in groups.get("particle", groups.get("particle_features", []))]
    topology = [str(v).lower() for key in ("vertex", "vertex_features", "candidate", "candidate_features") for v in groups.get(key, [])]
    globals_ = [str(v).lower() for v in groups.get("global", groups.get("global_features", []))]
    pid_markers = ("target", "reco_id", "pdg", "particle_id", "pid", "nne", "nnk", "nnp", "nnpi", "nnmu")
    local_pid_or_mass = {"energy", "log_e", "e_fraction", "mass", "calo_ecal", "calo_hcal2ecal", "calo_e49", "calo_prs", "has_pid", "has_muon_pid", "has_calo"}
    bad_particle = [name for name in particle if name in local_pid_or_mass or any(marker in name for marker in pid_markers)]
    # Topology tokens are allowed PID-independent observables such as common-
    # pion mass, summed charge, and momentum. Species-named mass hypotheses are
    # not. The postprocessor additionally records its construction contract.
    named_hypotheses = ("kaon", "proton", "electron", "muon", "d0", "bplus", "exclusive")
    bad_topology = [
        name for name in topology
        if any(marker in name for marker in pid_markers)
        or ("mass" in name and "common_pion" not in name and "all_pion" not in name)
        or any(marker in name for marker in named_hypotheses)
    ]
    safe_global_masses = ("massless", "common_pion", "all_pion")
    bad_global = [
        name for name in globals_
        if any(marker in name for marker in ("truth", "sample_id", "flavour", "flavor", "label", "energy"))
        or ("mass" in name and not any(marker in name for marker in safe_global_masses))
    ]
    if bad_particle or bad_topology or bad_global:
        raise ValueError(
            "Encoder feature contract contains leakage or non-geometric topology features: "
            f"particle={bad_particle}, topology={bad_topology}, global={bad_global}"
        )


def _read_manifest(path: Path) -> tuple[list[Path], dict[str, Any]]:
    payload = json.loads(path.read_text())
    records = payload if isinstance(payload, list) else payload.get("shards", payload.get("files", []))
    metadata = {} if isinstance(payload, list) else payload.get("metadata", {})
    paths = []
    lengths: dict[str, int] = {}
    for record in records:
        raw = record if isinstance(record, str) else record.get("path", record.get("file"))
        if raw is None:
            raise ValueError(f"Manifest record has no path: {record}")
        item = Path(raw)
        resolved = (item if item.is_absolute() else path.parent / item).resolve()
        paths.append(resolved)
        if isinstance(record, Mapping) and record.get("jets") is not None:
            lengths[str(resolved)] = int(record["jets"])
    if lengths:
        metadata = dict(metadata); metadata["_shard_lengths"] = lengths
    return paths, metadata


def resolve_shards(source: str | Path | Sequence[str | Path]) -> tuple[list[Path], dict[str, Any]]:
    """Resolve a manifest, directory, glob, shard, or sequence of those."""
    sources = [source] if isinstance(source, (str, Path)) else source
    result: list[Path] = []
    metadata: dict[str, Any] = {}
    for raw in sources:
        path = Path(raw)
        if path.suffix == ".json" and path.is_file():
            found, meta = _read_manifest(path.resolve()); result.extend(found); metadata.update(meta)
        elif path.is_dir():
            result.extend(sorted(path.glob("*.npz"))); result.extend(sorted(path.glob("*.pt")))
        elif any(char in str(raw) for char in "*?["):
            result.extend(Path(item).resolve() for item in sorted(glob.glob(str(raw))))
        else:
            result.append(path.resolve())
    unique = list(dict.fromkeys(result))
    if not unique:
        raise FileNotFoundError(f"No ML shards resolved from {source}")
    missing = [str(path) for path in unique if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing ML shards: {missing[:5]}")
    return unique, metadata


def _load_shard(
    path: Path,
    *,
    include_analysis: bool = False,
    analysis_keys: set[str] | None = None,
) -> Mapping[str, Any]:
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as payload:
            return {
                key: payload[key]
                for key in payload.files
                if (
                    include_analysis
                    or key in (analysis_keys or set())
                    or not (key.startswith("analysis_") or key.startswith("legacy_sv_") or key == "target_reco_id")
                )
            }
    if path.suffix in {".pt", ".pth"}:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(payload, Mapping):
            raise TypeError(f"Expected a mapping in {path}")
        return {
            key: value for key, value in payload.items()
            if (
                include_analysis
                or key in (analysis_keys or set())
                or not (key.startswith("analysis_") or key.startswith("legacy_sv_") or key == "target_reco_id")
            )
        }
    raise ValueError(f"Unsupported shard format: {path}")


def _shard_length(payload: Mapping[str, Any]) -> int:
    for key in ("particle_features", "particles", "event_uid", "event_id", "global_features"):
        if key in payload:
            return len(payload[key])
    raise ValueError("Cannot determine shard length")


def _shard_keys(path: Path) -> set[str]:
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as payload:
            return set(payload.files)
    return set(_load_shard(path, include_analysis=False))


def _validate_shard_polarity(path: Path, expected_length: int) -> None:
    """Validate the required scalar detector condition without loading a full shard."""
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as payload:
            if "polarity" not in payload.files:
                raise KeyError(f"{path} is missing required event-level polarity")
            values = np.asarray(payload["polarity"])
    else:
        payload = _load_shard(path, include_analysis=False)
        if "polarity" not in payload:
            raise KeyError(f"{path} is missing required event-level polarity")
        values = _as_array(payload["polarity"])
    if values.shape != (expected_length,):
        raise ValueError(
            f"{path}: polarity must be one scalar per jet with shape "
            f"({expected_length},), got {values.shape}"
        )
    if values.dtype.kind not in "iu":
        raise TypeError(f"{path}: polarity must have an integer dtype, got {values.dtype}")
    valid = (values == POLARITY_VALUES[0]) | (values == POLARITY_VALUES[1])
    if not np.all(valid):
        observed = np.unique(values).tolist()
        raise ValueError(
            f"{path}: polarity must contain only -1 (down) and +1 (up); "
            f"observed {observed}"
        )


def _as_array(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    if isinstance(value, np.ndarray) and value.dtype == object and value.ndim == 0:
        value = value.item()
    return np.asarray(value)


class JetShardDataset(Dataset):
    """Random access over shards with a small per-worker LRU shard cache."""

    def __init__(
        self,
        source: str | Path | Sequence[str | Path],
        *,
        cache_size: int = 1,
        include_analysis: bool = False,
        analysis_keys: Sequence[str] | None = None,
    ) -> None:
        self.paths, self.metadata = resolve_shards(source)
        validate_feature_contract(self.metadata)
        validate_signed_reco_pid_metadata(self.metadata)
        target_contract = self.metadata.get("pid_target_contract", {})
        self.pid_target_contract = (
            target_contract.get("name") if isinstance(target_contract, Mapping) else None
        )
        self.cache_size = max(1, int(cache_size))
        self.include_analysis = bool(include_analysis)
        self.analysis_keys = set(analysis_keys or ())
        self._cache: OrderedDict[int, Mapping[str, Any]] = OrderedDict()
        self.lengths = []
        manifest_lengths = self.metadata.pop("_shard_lengths", {})
        for path in self.paths:
            canonical = {_canonical(key) for key in _shard_keys(path)}
            missing = {"particle_features", "pid_features", "polarity"} - canonical
            if self.pid_target_contract in SIGNED_RECO_PID_TARGETS:
                missing |= {"target_reco_id"} - canonical
            if missing:
                raise KeyError(f"{path} is missing {sorted(missing)}")
            if str(path) in manifest_lengths:
                length = int(manifest_lengths[str(path)])
            else:
                length = _shard_length(_load_shard(path, include_analysis=False))
            _validate_shard_polarity(path, length)
            self.lengths.append(length)
        self.offsets = np.cumsum([0, *self.lengths], dtype=np.int64).tolist()

    def __len__(self) -> int:
        return self.offsets[-1]

    def _payload(self, shard_id: int) -> Mapping[str, Any]:
        if shard_id in self._cache:
            payload = self._cache.pop(shard_id); self._cache[shard_id] = payload; return payload
        payload = _load_shard(
            self.paths[shard_id],
            include_analysis=self.include_analysis,
            analysis_keys=(
                self.analysis_keys | {"target_reco_id"}
                if self.pid_target_contract in SIGNED_RECO_PID_TARGETS
                else self.analysis_keys
            ),
        )
        if self.pid_target_contract in SIGNED_RECO_PID_TARGETS:
            payload = dict(payload)
            reco_id = _as_array(payload.pop("target_reco_id"))
            pid_class = signed_reco_pid_classes(reco_id, self.pid_target_contract)
            payload["pid_class"] = pid_class
            # Analysis/counterfactual builders need the exact reconstructed
            # identity to freeze candidate track selections and to propagate
            # the signed-target contract into compact residual views.  It is
            # still absent from ordinary training samples, and every model
            # caller uses an explicit encoder-key allowlist.
            if self.include_analysis or "target_reco_id" in self.analysis_keys:
                payload["target_reco_id"] = reco_id
            physical = _as_array(payload.get("particle_mask", np.ones_like(pid_class)))
            # The signed-ID species target is defined for every supported
            # reconstructed daughter.  Per-response availability remains in
            # pid_target_valid and is intentionally independent of this mask.
            payload["pid_eligible"] = physical.astype(bool) & (pid_class >= 0)
        self._cache[shard_id] = payload
        while len(self._cache) > self.cache_size:
            _, old = self._cache.popitem(last=False)
            if hasattr(old, "close"): old.close()
        return payload

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        if index < 0: index += len(self)
        if not 0 <= index < len(self): raise IndexError(index)
        shard_id = bisect.bisect_right(self.offsets, index) - 1
        payload, local = self._payload(shard_id), index - self.offsets[shard_id]
        sample: dict[str, torch.Tensor] = {}
        for raw_key, values in payload.items():
            if raw_key in {"num_jets", "metadata"}: continue
            key = _canonical(raw_key)
            try: array = _as_array(values[local])
            except (IndexError, TypeError): continue
            if array.dtype.kind in "OUS": continue
            if key in MASK_KEYS or key in INCIDENCE_KEYS: dtype = torch.bool
            elif key in INDEX_KEYS or key.endswith("_class") or key.endswith("_labels"): dtype = torch.long
            else: dtype = torch.float32
            tensor = torch.as_tensor(array, dtype=dtype)
            sample[key] = tensor
        sample.setdefault("sample_index", torch.tensor(index, dtype=torch.long))
        particles = sample.get("particle_features")
        if particles is None or particles.ndim != 2:
            raise ValueError(f"Jet {index}: particle_features must have shape [N,F]")
        sample.setdefault("particle_mask", torch.ones(particles.shape[0], dtype=torch.bool))
        if sample["pid_features"].shape[0] != particles.shape[0]:
            raise ValueError(f"Jet {index}: PID and particle lengths differ")
        return sample


class ShardLocalDistributedSampler(Sampler[int]):
    """Shuffle deterministically while preserving compressed-shard locality.

    A global random index permutation makes each DataLoader worker repeatedly
    decompress unrelated NPZ shards and defeats its small LRU cache. Shard
    order and rows within each shard are both randomized here, then every DDP
    rank receives an equally sized contiguous part of that shuffled stream.
    """

    def __init__(
        self,
        dataset: JetShardDataset,
        *,
        num_replicas: int = 1,
        rank: int = 0,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
    ) -> None:
        self.dataset = dataset
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        self.epoch = 0
        if self.num_replicas <= 0 or not 0 <= self.rank < self.num_replicas:
            raise ValueError(
                f"Invalid distributed sampler rank={rank}, replicas={num_replicas}"
            )
        size = len(dataset)
        if self.drop_last and size % self.num_replicas:
            self.num_samples = size // self.num_replicas
        else:
            self.num_samples = (size + self.num_replicas - 1) // self.num_replicas
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        n_shards = len(self.dataset.lengths)
        shard_order = (
            torch.randperm(n_shards, generator=generator).tolist()
            if self.shuffle else list(range(n_shards))
        )
        pieces = []
        for shard_id in shard_order:
            length = int(self.dataset.lengths[shard_id])
            offset = int(self.dataset.offsets[shard_id])
            local = (
                torch.randperm(length, generator=generator)
                if self.shuffle else torch.arange(length)
            )
            pieces.append(local + offset)
        indices = torch.cat(pieces) if pieces else torch.empty(0, dtype=torch.long)
        if self.drop_last:
            indices = indices[:self.total_size]
        elif len(indices) < self.total_size:
            padding = self.total_size - len(indices)
            repeats = (padding + len(indices) - 1) // max(len(indices), 1)
            indices = torch.cat((indices, indices.repeat(repeats)[:padding]))
        start = self.rank * self.num_samples
        return iter(indices[start:start + self.num_samples].tolist())

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)


def _pad_first(values: list[torch.Tensor], fill: float | int | bool = 0) -> torch.Tensor:
    shape = (len(values), max(v.shape[0] for v in values), *values[0].shape[1:])
    output = torch.full(shape, fill, dtype=values[0].dtype)
    for row, value in enumerate(values): output[row, :value.shape[0]] = value
    return output


def _pad_matrix(values: list[torch.Tensor], first: int, second: int) -> torch.Tensor:
    output = torch.zeros((len(values), first, second, *values[0].shape[2:]), dtype=values[0].dtype)
    for row, value in enumerate(values): output[row, :value.shape[0], :value.shape[1]] = value
    return output


def collate_jets(samples: Sequence[Mapping[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Pad heterogeneous jet records and return a dense training batch."""
    if not samples: raise ValueError("Cannot collate an empty batch")
    missing_polarity = [index for index, sample in enumerate(samples) if "polarity" not in sample]
    if missing_polarity:
        raise KeyError(
            "Every jet must carry event-level polarity; missing from batch rows "
            f"{missing_polarity[:8]}"
        )
    integer_dtypes = {
        torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64,
    }
    for index, sample in enumerate(samples):
        value = sample["polarity"]
        if (
            value.ndim != 0
            or value.dtype not in integer_dtypes
            or int(value.item()) not in POLARITY_VALUES
        ):
            observed = value.detach().cpu().tolist()
            raise ValueError(
                f"Batch row {index}: polarity must be scalar -1 (down) or +1 (up), "
                f"got {observed}"
            )
    batch: dict[str, torch.Tensor] = {}
    keys = set.intersection(*(set(sample) for sample in samples))
    particle_keys = {"particle_features", "particle_feature_valid", "pairwise_inputs", "pid_features", "particle_mask", "pid_class", "pid_bins", "pid_values", "particle_origin", "particle_targets", "pid_target_valid", "pid_eligible", "particle_charge", "particle_original_index", "target_reco_id", "pid_available"}
    vertex_keys = {"vertex_features", "vertex_mask", "vertex_class", "vertex_targets"}
    candidate_keys = {"candidate_features", "candidate_mask", "candidate_class", "candidate_targets", "candidate_mass_targets"}
    npart = max(s["particle_features"].shape[0] for s in samples)
    nvert = max((s.get("vertex_features", torch.empty(0, 0)).shape[0] for s in samples), default=0)
    ncand = max((s.get("candidate_features", torch.empty(0, 0)).shape[0] for s in samples), default=0)
    for key in sorted(keys):
        values = [sample[key] for sample in samples]
        if key in particle_keys | vertex_keys | candidate_keys:
            fill = False if key.endswith("mask") or key.endswith("valid") else (-100 if values[0].dtype == torch.long else 0)
            batch[key] = _pad_first(values, fill)
        elif key == "pairwise_features": batch[key] = _pad_matrix(values, npart, npart)
        elif key == "vertex_track_mask": batch[key] = _pad_matrix(values, nvert, npart)
        elif key == "candidate_track_mask": batch[key] = _pad_matrix(values, ncand, npart)
        elif key == "candidate_vertex_mask": batch[key] = _pad_matrix(values, ncand, nvert)
        elif values[0].ndim == 0: batch[key] = torch.stack(values)
        elif all(v.shape == values[0].shape for v in values): batch[key] = torch.stack(values)
    batch.setdefault("particle_mask", torch.ones(batch["particle_features"].shape[:2], dtype=torch.bool))
    if "vertex_features" in batch: batch.setdefault("vertex_mask", torch.ones(batch["vertex_features"].shape[:2], dtype=torch.bool))
    if "candidate_features" in batch: batch.setdefault("candidate_mask", torch.ones(batch["candidate_features"].shape[:2], dtype=torch.bool))
    if batch["polarity"].shape != (len(samples),):
        raise ValueError(f"Collated polarity must have shape [batch], got {tuple(batch['polarity'].shape)}")
    return batch


def move_to_device(batch: Mapping[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def stable_sample_ids(batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
    if "event_uid" in batch: ids = batch["event_uid"].long()
    elif "event_id" in batch: ids = batch["event_id"].long()
    else: return batch["sample_index"].long() & 0x7FFFFFFFFFFFFFFF
    if "jet_index" in batch: ids = ids * 2 + batch["jet_index"].long()
    return ids & 0x7FFFFFFFFFFFFFFF


def pid_eligibility(batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
    """Physical particles with at least one usable species/response target."""
    # Clone because bool() aliases an already-boolean input; the fallback
    # refinements below must never rewrite the physical padding mask.
    eligible = batch["particle_mask"].bool().clone()
    if "pid_eligible" in batch:
        return eligible & batch["pid_eligible"].bool()
    if "pid_target_valid" in batch:
        target_valid = batch["pid_target_valid"].bool()
        eligible &= target_valid.any(-1) if target_valid.ndim == 3 else target_valid
    elif "pid_class" in batch:
        eligible &= batch["pid_class"] >= 0
    elif "pid_bins" in batch:
        eligible &= (batch["pid_bins"] >= 0).any(-1)
    if "particle_charge" in batch:
        eligible &= batch["particle_charge"] != 0
    return eligible


def deterministic_pid_mask(valid_mask: torch.Tensor, sample_ids: torch.Tensor, *, fraction: float, seed: int, epoch: int = 0, ensure_one: bool = True) -> torch.Tensor:
    """Stateless masking invariant to DataLoader ordering and DDP world size."""
    if not 0.0 <= fraction <= 1.0: raise ValueError(f"Invalid mask fraction {fraction}")
    n_particles = valid_mask.shape[1]
    particle = torch.arange(n_particles, device=valid_mask.device, dtype=torch.int64)[None]
    value = sample_ids[:, None].to(torch.int64) ^ (particle * 0x1E35A7BD) ^ int(seed) ^ (int(epoch) * 0x6C8E9CF5)
    value ^= value >> 30; value *= -4658895280553007687
    value ^= value >> 27; value *= -7723592293110705685
    value ^= value >> 31
    uniform = (value & 0x7FFFFFFFFFFFFFFF).to(torch.float64) / float(0x7FFFFFFFFFFFFFFF)
    selected = valid_mask & (uniform < fraction)
    if ensure_one and fraction > 0:
        missing = valid_mask.any(1) & ~selected.any(1)
        if missing.any():
            chosen = uniform.masked_fill(~valid_mask, 2.0).argmin(1)
            selected[missing, chosen[missing]] = True
    return selected


@dataclass(frozen=True)
class BatchDimensions:
    particle_features: int
    pid_features: int
    global_features: int = 0
    vertex_features: int = 0
    candidate_features: int = 0
    pairwise_features: int = 0

    @classmethod
    def from_batch(cls, batch: Mapping[str, torch.Tensor]) -> "BatchDimensions":
        width = lambda key: int(batch[key].shape[-1]) if key in batch else 0
        pairwise = width("pairwise_features")
        if not pairwise and "pairwise_inputs" in batch:
            pairwise = 5
        return cls(width("particle_features"), width("pid_features"), width("global_features"), width("vertex_features"), width("candidate_features"), pairwise)
