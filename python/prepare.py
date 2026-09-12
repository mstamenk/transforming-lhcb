#!/usr/bin/env python3
"""Build leakage-safe, sharded transformer inputs with PID-blind topology.

The expensive event work is a compiled RDataFrame graph.  ROOT is used as a
stable intermediate; uproot only repacks its already-derived fixed arrays into
GPU-friendly NPZ shards and never reconstructs physics objects.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import platform
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import awkward as ak
import numpy as np
import uproot
import yaml

from features import (PARTICLE_NAMES, PID_NAMES, PID_CLASS_ORDER, GLOBAL_NAMES, EVENT_CONTEXT_NAMES, VERTEX_NAMES, CANDIDATE_NAMES, PAIRWISE_INPUT_NAMES, LEGACY_SV, dense, padded, signed_log, validated_polarity, feature_payload, quantile_edges, robust_scalers, apply_scaler, fiducial_jet_mask, selected_jet_mask, _splitmix64_array, jet_selection_hash)
from ml_rdf_helpers import declare_ml_helpers
from pid_targets import (
    SIGNED_RECO_PID_TARGETS,
    pid_target_sha256,
    signed_reco_pid_classes,
    signed_reco_pid_class_order,
    signed_reco_pid_contract,
)


RECORD_RVEC_I = [
    "particle_valid", "original_index", "charge", "has_track", "has_pid", "has_muon_pid", "has_calo",
    "target_species", "target_reco_id", "target_valid_e", "target_valid_k", "target_valid_p", "target_valid_pi", "target_valid_mu",
    "vertex_valid", "vertex_n_tracks", "vertex_track0", "vertex_track1", "vertex_track2", "vertex_track3", "vertex_charge",
    "chain_valid", "chain_child_n_tracks", "chain_track0", "chain_track1", "chain_track2", "chain_track3", "chain_bachelor", "chain_child_vertex_index", "chain_charge",
]
RECORD_RVEC_F = [
    "log_pt", "log_p", "log_e", "pt_fraction", "e_fraction", "delta_eta", "delta_phi", "px", "py", "pz", "energy",
    "ip", "ip_raw", "log1p_ipchi2", "track_chi2", "qoverp", "state_dx", "state_dy", "state_dz", "dir_x", "dir_y", "dir_z",
    "calo_ecal", "calo_hcal2ecal", "calo_e49", "calo_prs", "target_nne", "target_nnk", "target_nnp", "target_nnpi", "target_nnmu",
    "vertex_x", "vertex_y", "vertex_z", "vertex_rms", "vertex_max_doca", "vertex_flight_pv", "vertex_ip_pv", "vertex_pointing",
    "vertex_px", "vertex_py", "vertex_pz", "vertex_pt", "vertex_mass_pi", "vertex_corrected_mass_pi", "vertex_min_ipchi2", "vertex_sum_ipchi2", "vertex_fit_proxy",
    "chain_parent_x", "chain_parent_y", "chain_parent_z", "chain_parent_rms", "chain_parent_max_doca", "chain_parent_flight_pv", "chain_parent_ip_pv", "chain_parent_pointing",
    "chain_child_flight", "chain_child_pointing", "chain_child_ip_pv", "chain_child_mass_pi", "chain_child_corrected_mass_pi", "chain_parent_mass_pi", "chain_parent_corrected_mass_pi", "chain_parent_pt", "chain_fit_proxy",
]
RECORD_SCALARS = ["n_particles_input", "n_particles_stored", "n_track_lines", "n_pair_total", "n_triplet_total", "n_quad_total", "n_chain_total", "vertices_truncated", "chains_truncated", "particles_truncated"]
RAW_FORMAT_VERSION = 3


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as handle:
        temp = Path(handle.name)
        json.dump(payload, handle, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def atomic_npz(path: Path, payload: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".npz")
    os.close(descriptor)
    temp = Path(name)
    try:
        np.savez_compressed(temp, **payload)
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def processing_fingerprint(
    cfg: dict[str, Any],
    dataset: dict[str, Any] | None = None,
) -> str:
    contract = {
        "raw_format_version": RAW_FORMAT_VERSION,
        "data": cfg.get("data", {}),
        "split": cfg.get("split", {}),
        "seed": cfg.get("seed", 2026),
        "particle_names": PARTICLE_NAMES,
        "pid_names": PID_NAMES,
        "global_names": GLOBAL_NAMES,
        "vertex_names": VERTEX_NAMES,
        "candidate_names": CANDIDATE_NAMES,
        "pairwise_input_names": PAIRWISE_INPUT_NAMES,
    }
    encoded = json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def explicit_source_id(path: Path) -> int:
    """Stable non-manifest source namespace for an explicitly supplied file."""
    digest = hashlib.sha256(str(path.resolve()).encode()).digest()
    return 0x80000000 | (int.from_bytes(digest[:4], "big") & 0x7FFFFFFF)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-config", type=Path, default=Path("config/dataset.yaml"))
    p.add_argument("--transformer-config", type=Path, default=Path("config/features.yaml"))
    p.add_argument("--output", type=Path, default=Path("outputs/ml"))
    p.add_argument("--sample", choices=["all", "bb", "cc", "light", "z_bb"], default="all")
    p.add_argument("--polarity", choices=["all", "up", "down"], default="all")
    p.add_argument("--file-index", type=int, help="process exactly this zero-based manifest file")
    p.add_argument("--files", type=Path, nargs="+", help="input override; source ids are stable hashes of absolute paths")
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--max-events", type=int)
    p.add_argument("--shard-size", type=int)
    p.add_argument("--max-particles", type=int)
    p.add_argument("--max-vertices", type=int)
    p.add_argument("--max-candidates", type=int)
    p.add_argument("--seed", type=int)
    p.add_argument("--write-root", action="store_true", help="retain the event-level ROOT intermediate")
    p.add_argument("--finalize-only", action="store_true", help="derive train-only PID quantiles and manifests from raw_*.npz below output")
    p.add_argument(
        "--raw-input", type=Path,
        help=(
            "read compatible raw_source*.npz shards from this directory while "
            "writing finalized artifacts below --output; valid only with --finalize-only"
        ),
    )
    p.add_argument(
        "--reference-preprocessing", type=Path,
        help=(
            "transform raw shards with an existing preprocessing.json instead "
            "of refitting quantiles/scalers; required for inference-only samples"
        ),
    )
    p.add_argument(
        "--finalize-workers", type=int, default=1,
        help="parallel shard writers when using --reference-preprocessing",
    )
    p.add_argument(
        "--reuse-finalized-shards", action="store_true",
        help=(
            "validate and reuse split NPZ files left by a reference-based "
            "finalization that failed before writing manifests"
        ),
    )
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def git_hash() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def selected_sources(dataset: dict[str, Any], args: argparse.Namespace) -> list[tuple[int, Path, dict[str, Any]]]:
    base = args.dataset_config.resolve().parent.parent / dataset.get("data_dir", "data")
    if args.files:
        records = [(explicit_source_id(p), p.resolve(), {"name": p.name, "sample": "unknown", "polarity": "unknown", "pt_bin_gev": None}) for p in args.files]
        ids = [record[0] for record in records]
        if len(ids) != len(set(ids)):
            raise RuntimeError("Explicit input paths produced a source-id collision")
    else:
        records = []
        for i, item in enumerate(dataset["files"]):
            if args.sample != "all" and item["sample"] != args.sample: continue
            if args.polarity != "all" and item["polarity"] != args.polarity: continue
            records.append((i, (base / item["name"]).resolve(), item))
    if args.file_index is not None:
        matches = [r for r in records if r[0] == args.file_index]
        if not matches: raise IndexError(f"Manifest file index {args.file_index} is not selected")
        records = matches
    missing = [str(p) for _, p, _ in records if not p.is_file()]
    if missing: raise FileNotFoundError(f"Missing input files: {missing}")
    if not records: raise RuntimeError("No input ROOT files selected")
    return records


def record_expression(j: int, pv: str, cfg: dict[str, Any]) -> str:
    p = f"Jet{j}_Daughters_"
    d = cfg["data"]
    maxv = int(d["max_vertices"])
    pair, triple = maxv // 2, maxv // 4
    quad = maxv - pair - triple
    return (
        "lhcb_ml::build_jet_record("
        + ",".join(p + x for x in ["E", "pT", "ID", "pX", "pY", "pZ", "Eta", "Phi", "Q", "IP", "IPraw", "IPCHI2", "NNe", "NNk", "NNp", "NNpi", "NNmu", "Chi2", "QoverP", "trackX", "trackY", "trackZ", "trackVX", "trackVY", "trackVZ", "CaloNeutralEcal", "CaloNeutralHcal2Ecal", "CaloNeutralE49", "CaloNeutralPrs"])
        + f",Jet{j}_PT,Jet{j}_ETA,Jet{j}_PHI,Jet{j}_PE,{pv}_X,{pv}_Y,{pv}_Z"
        + f",{d['max_particles']},{pair},{triple},{quad},{d['max_candidates']}"
        + f",{d.get('max_vertex_doca_mm', 0.25)},{d.get('min_vertex_flight_mm', 0.0)},{d.get('min_chain_pointing', 0.95)},{d.get('max_abs_state_z_mm', 500.0)}"
        + f",{d.get('min_vertex_track_pt_mev', 250.0)},{d.get('min_vertex_track_ipchi2', 4.0)})"
    )


def snapshot_source(
    ROOT: Any,
    source_id: int,
    path: Path,
    tree_name: str,
    output: Path,
    cfg: dict[str, Any],
    dataset: dict[str, Any],
    source_meta: dict[str, Any],
    max_events: int | None,
) -> int:
    df = ROOT.RDataFrame(tree_name, str(path))
    entry_limits = [
        int(value)
        for value in (source_meta.get("entry_stop"), max_events)
        if value is not None
    ]
    if entry_limits:
        entry_stop = min(entry_limits)
        if entry_stop < 1:
            raise ValueError(f"entry_stop must be positive, got {entry_stop}")
        if dataset.get("ordered_entry_selection", False):
            if ROOT.IsImplicitMTEnabled():
                raise RuntimeError(
                    "ordered_entry_selection requires implicit multithreading "
                    "to be disabled"
                )
            df = df.Range(entry_stop)
        else:
            df = df.Filter(f"rdfentry_ < {entry_stop}", "configured input entry stop")
    row_filter = dataset.get("row_filter")
    if row_filter:
        if not isinstance(row_filter, str):
            raise TypeError("dataset row_filter must be a C++ expression string")
        df = df.Filter(row_filter, "dataset row filter")
    if dataset.get("selected_rows_are_limit", False):
        selected_rows = source_meta.get("selected_rows")
        if selected_rows is None:
            raise ValueError("selected_rows_are_limit requires selected_rows for every source")
        selected_rows = int(selected_rows)
        if selected_rows < 1:
            raise ValueError(f"selected_rows must be positive, got {selected_rows}")
        if ROOT.IsImplicitMTEnabled():
            raise RuntimeError(
                "selected_rows_are_limit requires ordered single-thread execution"
            )
        # Range after the event-level row filter gives an exact deterministic
        # retained-row cap without pre-scanning legacy compressed TTrees.
        df = df.Range(selected_rows)
    df = df.Define("source_entry", "rdfentry_")
    df = df.Define("raw_entry", "static_cast<ULong64_t>(source_entry)")
    df = df.Define("source_id", str(source_id))
    # The jet-specific PV is the only PV association available for each jet.
    for j in (0, 1):
        df = df.Define(f"rec{j}", record_expression(j, f"Jet{j}_OWNPV", cfg))
        for name in RECORD_RVEC_I + RECORD_RVEC_F + RECORD_SCALARS:
            df = df.Define(f"j{j}_{name}", f"rec{j}.{name}")
        for name in LEGACY_SV:
            df = df.Define(f"j{j}_legacy_sv_{name}", f"lhcb_ml::first_or(Jet{j}_BDTTag_{name})")
    columns = ["runNumber", "eventNumber", "Polarity", "nPVs", "nTracks", "source_entry", "raw_entry", "source_id"]
    for j in (0, 1):
        columns += [f"Jet{j}_{x}" for x in ["PT", "ETA", "PHI", "M", "PE", "width", "nDaughters", "OWNPV_X", "OWNPV_Y", "OWNPV_Z", "ENDVERTEX_X", "ENDVERTEX_Y", "ENDVERTEX_Z", "ORIVX_X", "ORIVX_Y", "ORIVX_Z", "vtx_x", "vtx_y", "vtx_z", "BDTTag_Tag", "BDTTag_NbTag", "mc_flavour", "mc_deltaR", "mc_PT"]]
        columns += [f"j{j}_{x}" for x in RECORD_RVEC_I + RECORD_RVEC_F + RECORD_SCALARS]
        columns += [f"j{j}_legacy_sv_{x}" for x in LEGACY_SV]
    opts = ROOT.RDF.RSnapshotOptions(); opts.fMode = "RECREATE"; opts.fCompressionAlgorithm = 4; opts.fCompressionLevel = 4; opts.fLazy = True
    snap_df = df.Snapshot("EventML", str(output), columns, opts)
    return int(snap_df.Count().GetValue())


def concatenate_rows(a: dict[str,np.ndarray], b: dict[str,np.ndarray]) -> dict[str,np.ndarray]:
    return {k: np.concatenate([a[k], b[k]], axis=0) for k in a}


def write_raw_shards(root_path: Path, output: Path, source_id: int, source_meta: dict[str, Any], cfg: dict[str, Any], shard_size: int, seed: int, fingerprint: str) -> list[Path]:
    paths=[]; sequence=0
    for chunk in uproot.iterate(f"{root_path}:EventML", step_size=max(1,shard_size//2), library="ak"):
        jet0=feature_payload(chunk,0,source_meta,cfg["split"],seed)
        jet1=feature_payload(chunk,1,source_meta,cfg["split"],seed)
        payload=concatenate_rows(jet0,jet1)
        order=np.lexsort((payload["jet_index"],payload["source_entry"]));payload={k:v[order] for k,v in payload.items()}
        for start in range(0,len(order),shard_size):
            part={k:v[start:start+shard_size] for k,v in payload.items()};path=output/f"raw_source{source_id:03d}_{sequence:05d}.npz"
            size=len(part["event_uid"])
            part["raw_format_version"]=np.full(size,RAW_FORMAT_VERSION,dtype=np.int16)
            part["raw_processing_sha256"]=np.full(size,fingerprint.encode(),dtype="S64")
            atomic_npz(path,part);paths.append(path);sequence+=1
    return paths


def discover_raw_shards(
    output: Path, expected_fingerprint: str | None,
) -> list[Path]:
    """Return compatible top-level raw shards, rejecting mixed productions."""
    raw=sorted(output.glob("raw_source*.npz"))
    if not raw:
        raise FileNotFoundError(f"No top-level raw_source*.npz shards in {output}")
    identities:set[tuple[int,int]]=set()
    expected_particle=len(PARTICLE_NAMES)
    observed_fingerprint: str | None = None
    for path in raw:
        pieces=path.stem.split("_")
        if len(pieces)!=3 or not pieces[1].startswith("source"):
            raise ValueError(f"Malformed raw shard name: {path.name}")
        identity=(int(pieces[1].removeprefix("source")),int(pieces[2]))
        if identity in identities:
            raise ValueError(f"Duplicate raw source/chunk identity: {identity}")
        identities.add(identity)
        with np.load(path,allow_pickle=False) as z:
            required={"source_id","event_uid","split","particle_features","pid_values","pid_target_valid","vertex_features","candidate_features","global_features","polarity","raw_format_version","raw_processing_sha256"}
            missing=required-set(z.files)
            if missing:
                raise ValueError(f"{path.name} lacks required raw fields: {sorted(missing)}")
            if z["particle_features"].ndim!=3 or z["particle_features"].shape[-1]!=expected_particle:
                raise ValueError(f"{path.name} has incompatible particle feature shape {z['particle_features'].shape}")
            if z["global_features"].shape[-1]!=len(GLOBAL_NAMES) or z["pid_values"].shape[-1]!=len(PID_NAMES):
                raise ValueError(f"{path.name} has incompatible global/PID feature dimensions")
            polarity=np.asarray(z["polarity"])
            if polarity.shape!=(len(z["event_uid"]),) or polarity.dtype.kind not in "iu":
                raise ValueError(f"{path.name} polarity must be an integer scalar per jet")
            if not np.all((polarity==-1)|(polarity==1)):
                raise ValueError(f"{path.name} has invalid polarity values {np.unique(polarity).tolist()}")
            if not np.all(z["source_id"]==identity[0]):
                raise ValueError(f"{path.name} source_id content does not match its filename")
            if not np.all(z["raw_format_version"]==RAW_FORMAT_VERSION):
                raise ValueError(f"{path.name} uses an incompatible raw format")
            fingerprints=np.unique(z["raw_processing_sha256"].astype("U64"))
            if len(fingerprints) != 1:
                raise ValueError(f"{path.name} contains mixed raw processing fingerprints")
            fingerprint = str(fingerprints[0])
            if expected_fingerprint is not None and fingerprint != expected_fingerprint:
                raise ValueError(f"{path.name} processing fingerprint does not match this configuration")
            if observed_fingerprint is None:
                observed_fingerprint = fingerprint
            elif fingerprint != observed_fingerprint:
                raise ValueError("Explicit raw input contains shards from mixed productions")
    return raw


def raw_fingerprint(raw_path: Path) -> str:
    with np.load(raw_path, allow_pickle=False) as payload:
        values = np.unique(payload["raw_processing_sha256"].astype("U64"))
    if len(values) != 1:
        raise ValueError(f"{raw_path} contains mixed raw processing fingerprints")
    return str(values[0])


def resolve_jet_selection(
    raw_paths: list[Path], cfg: dict[str, Any], names: dict[int, str],
) -> dict[str, Any] | None:
    configured = cfg.get("data", {}).get("jet_selection")
    if configured is None:
        return None
    selection = dict(configured)
    for key in ("pt_min_mev", "eta_min", "eta_max"):
        if key not in selection:
            raise ValueError(f"data.jet_selection requires {key}")
    if not float(selection["eta_min"]) < float(selection["eta_max"]):
        raise ValueError("data.jet_selection eta_min must be smaller than eta_max")
    caps = selection.get("max_jets_by_split", {})
    if not isinstance(caps, dict):
        raise TypeError("data.jet_selection.max_jets_by_split must be a mapping")
    unknown = set(caps) - set(names.values())
    if unknown:
        raise ValueError(f"Unknown capped split names: {sorted(unknown)}")
    hash_parts: dict[int, list[np.ndarray]] = {split: [] for split in names}
    available = {split: 0 for split in names}
    for path in raw_paths:
        with np.load(path, allow_pickle=False) as payload:
            base = fiducial_jet_mask(payload, selection)
            hashes = jet_selection_hash(payload)
            split_values = np.asarray(payload["split"], dtype=np.int8)
            for split in names:
                mask = base & (split_values == split)
                available[split] += int(mask.sum())
                if names[split] in caps and mask.any():
                    hash_parts[split].append(hashes[mask])
    thresholds: dict[int, int] = {}
    selected_counts: dict[int, int] = {}
    for split, name in names.items():
        if name not in caps:
            selected_counts[split] = available[split]
            continue
        target = int(caps[name])
        if target < 1:
            raise ValueError(f"Jet cap for {name} must be positive")
        if available[split] < target:
            raise ValueError(
                f"Fiducial {name} split has {available[split]} jets, below requested {target}"
            )
        values = np.concatenate(hash_parts[split])
        threshold = int(np.partition(values, target - 1)[target - 1])
        count = int((values <= np.uint64(threshold)).sum())
        if count != target:
            raise RuntimeError(
                f"Stable-hash collision prevents exact {name} cap: selected {count}, target {target}"
            )
        thresholds[split] = threshold
        selected_counts[split] = target
    return {
        "config": selection,
        "available_by_split": {names[key]: value for key, value in available.items()},
        "selected_by_split": {names[key]: value for key, value in selected_counts.items()},
        "threshold_by_split": {names[key]: value for key, value in thresholds.items()},
        "thresholds": thresholds,
    }


def entropy_from_counts(counts: np.ndarray) -> float:
    total = float(counts.sum())
    if total <= 0:
        return 1.0
    probabilities = counts[counts > 0].astype(np.float64) / total
    value = float(-(probabilities * np.log(probabilities)).sum())
    return max(value, 1e-6)


def finalize_reference_shard(
    raw_path: Path, output: Path, names: dict[int, str], edges: list[Any],
    scalers: dict[str, Any], storage_dtype: Any, signed_species: bool,
    species_target: str, bins: int, pid_continuous_dim: int,
    selection: dict[str, Any] | None,
    reuse_finalized_shards: bool = False,
) -> list[tuple[int, str, int]]:
    """Apply frozen preprocessing to one raw shard and write its split parts."""
    with np.load(raw_path, allow_pickle=False) as payload:
        base = {key: payload[key] for key in payload.files}
    result: list[tuple[int, str, int]] = []
    physical = selected_jet_mask(base, selection)
    for split, name in names.items():
        select = (base["split"] == split) & physical
        if not select.any():
            continue
        destination = output / f"{name}_{raw_path.stem.removeprefix('raw_')}.npz"
        if reuse_finalized_shards:
            if not destination.is_file() or destination.stat().st_size == 0:
                raise FileNotFoundError(
                    f"Cannot reuse missing or empty finalized shard: {destination}"
                )
            with np.load(destination, allow_pickle=False) as existing:
                observed = len(existing["event_uid"])
                required = {"pid_bins", "particle_features", "global_features"}
                missing = required.difference(existing.files)
                if missing:
                    raise ValueError(
                        f"Reusable shard {destination} lacks keys {sorted(missing)}"
                    )
                if pid_continuous_dim == 0 and "pid_values" in existing.files:
                    raise ValueError(
                        f"Reusable shard {destination} retained forbidden pid_values"
                    )
            expected = int(select.sum())
            if observed != expected:
                raise ValueError(
                    f"Reusable shard {destination} has {observed} jets, expected {expected}"
                )
            result.append((split, destination.name, expected))
            continue
        part = {key: value[select] for key, value in base.items()}
        pid_bins = np.full(part["pid_values"].shape, -1, np.int16)
        for channel, edge in enumerate(edges):
            values = part["pid_values"][..., channel]
            valid = part["pid_target_valid"][..., channel]
            pid_bins[..., channel][valid] = np.clip(
                np.searchsorted(np.asarray(edge[1:-1]), values[valid], side="right"),
                0, bins - 1,
            )
        part["pid_bins"] = pid_bins
        if signed_species:
            part["pid_class"] = signed_reco_pid_classes(
                part["target_reco_id"], species_target,
            )
            part["pid_eligible"] = part["particle_mask"] & (part["pid_class"] >= 0)
        for key, stats in scalers.items():
            apply_scaler(part, key, stats)
        for key in (
            "particle_features", "pid_features", "global_features",
            "vertex_features", "candidate_features",
        ):
            part[key] = part[key].astype(storage_dtype)
        if pid_continuous_dim == 0:
            part.pop("pid_values", None)
        atomic_npz(destination, part)
        result.append((split, destination.name, int(select.sum())))
    return result


def finalize(
    output: Path, cfg: dict[str, Any],
    dataset: dict[str, Any], force: bool,
    reference_preprocessing: Path | None = None,
    finalize_workers: int = 1,
    raw_input: Path | None = None,
    reuse_finalized_shards: bool = False,
) -> dict[str, Any]:
    fingerprint=processing_fingerprint(cfg,dataset)
    raw_root = (raw_input or output).resolve()
    raw=discover_raw_shards(
        raw_root, None if raw_input is not None else fingerprint,
    )
    source_raw_fingerprint = raw_fingerprint(raw[0])
    raw_jets = 0
    empty_jet_rows = 0
    for raw_path in raw:
        with np.load(raw_path, allow_pickle=False) as arrays:
            raw_jets += len(arrays["event_uid"])
            empty_jet_rows += int(
                (~np.asarray(arrays["particle_mask"], dtype=bool).any(axis=1)).sum()
            )
    expected_jets = dataset.get("selected_jet_slots")
    if expected_jets is not None:
        if raw_jets != int(expected_jets):
            raise RuntimeError(
                f"Prepared jet count {raw_jets} != selected_jet_slots "
                f"{int(expected_jets)}"
            )
    names={0:"train",1:"validation",2:"test",3:"analysis"}
    jet_selection = resolve_jet_selection(raw, cfg, names)
    finalized=[output/f"{name}.json" for name in names.values()]+[output/"preprocessing.json",output/"metadata.json"]
    finalized += [path for name in names.values() for path in output.glob(f"{name}_source*.npz")]
    existing=[path for path in finalized if path.exists()]
    if reuse_finalized_shards and reference_preprocessing is None:
        raise ValueError("--reuse-finalized-shards requires --reference-preprocessing")
    if existing and not force and not reuse_finalized_shards:
        raise FileExistsError(f"Finalized artifacts already exist; pass --force to replace them: {existing[:5]}")
    bins=int(cfg.get("objectives",{}).get("masked_pid",{}).get("n_bins",32))
    reference_meta: dict[str, Any] | None = None
    if reference_preprocessing is None:
        edges=quantile_edges(raw,bins,jet_selection)
        seed=int(cfg.get("seed",2026));scalers=robust_scalers(raw,seed,selection=jet_selection)
    else:
        reference_path = reference_preprocessing.resolve()
        reference_payload = json.loads(reference_path.read_text())
        if not isinstance(reference_payload, dict):
            raise TypeError("Reference preprocessing payload must be a mapping")
        # Accept either preprocessing.json itself or a finalized split manifest.
        # The latter is useful for immutable legacy productions that retained the
        # complete preprocessing contract in every manifest but no standalone file.
        reference_meta = reference_payload.get("metadata", reference_payload)
        if not isinstance(reference_meta, dict):
            raise TypeError("Reference manifest metadata must be a mapping")
        expected_features = {
            "particle": PARTICLE_NAMES, "pid": PID_NAMES, "global": GLOBAL_NAMES,
            "vertex": VERTEX_NAMES, "candidate": CANDIDATE_NAMES,
            "pairwise_inputs": PAIRWISE_INPUT_NAMES,
        }
        observed_features = reference_meta.get("feature_names", {})
        for key, expected in expected_features.items():
            if list(observed_features.get(key, [])) != list(expected):
                raise ValueError(f"Reference preprocessing feature order differs for {key}")
        if "event_context" in observed_features and list(
            observed_features["event_context"]
        ) != EVENT_CONTEXT_NAMES:
            raise ValueError("Reference preprocessing event-context feature order differs")
        edges = reference_meta.get("pid_quantile_edges")
        scalers = reference_meta.get("robust_scalers")
        if not isinstance(edges, list) or len(edges) != len(PID_NAMES):
            raise ValueError("Reference preprocessing has incompatible PID quantile edges")
        if any(len(edge) != bins + 1 for edge in edges):
            raise ValueError("Reference PID quantile bin count differs from this config")
        if not isinstance(scalers, dict):
            raise ValueError("Reference preprocessing lacks robust_scalers")
    storage_dtype=np.float16 if cfg.get("data",{}).get("storage_dtype","float16")=="float16" else np.float32
    records={0:[],1:[],2:[],3:[]}; split_counts={0:0,1:0,2:0,3:0}
    species_target = cfg.get("objectives",{}).get("masked_pid",{}).get("species_target")
    signed_species = species_target in SIGNED_RECO_PID_TARGETS
    pid_class_order = (
        list(signed_reco_pid_class_order(species_target))
        if signed_species else list(PID_CLASS_ORDER)
    )
    species_counts=np.zeros(int(cfg.get("model",{}).get("pid_num_classes",6)),dtype=np.int64)
    if len(pid_class_order) != len(species_counts):
        raise ValueError(
            "Configured PID output width and target class order differ: "
            f"{len(species_counts)} != {len(pid_class_order)}"
        )
    response_counts=np.zeros((len(PID_NAMES),bins),dtype=np.int64)
    source_sample_counts=np.zeros(4,dtype=np.int64)
    flavour_probe_counts=np.zeros(3,dtype=np.int64)
    raw_to_process = raw
    if reference_meta is not None and (finalize_workers > 1 or reuse_finalized_shards):
        pid_continuous_dim = int(cfg.get("model",{}).get("pid_continuous_dim",0))
        def record_results(results: list[tuple[int, str, int]]) -> None:
            for split, path_name, jets in results:
                records[split].append({"path": path_name, "jets": jets})
                split_counts[split] += jets

        if finalize_workers > 1:
            with concurrent.futures.ThreadPoolExecutor(max_workers=finalize_workers) as executor:
                futures = [
                    executor.submit(
                        finalize_reference_shard, raw_path, output, names, edges,
                        scalers, storage_dtype, signed_species, species_target,
                        bins, pid_continuous_dim, jet_selection,
                        reuse_finalized_shards,
                    )
                    for raw_path in raw
                ]
                for future in concurrent.futures.as_completed(futures):
                    record_results(future.result())
        else:
            for raw_path in raw:
                record_results(finalize_reference_shard(
                    raw_path, output, names, edges, scalers, storage_dtype,
                    signed_species, species_target, bins, pid_continuous_dim,
                    jet_selection, reuse_finalized_shards,
                ))
        for split in records:
            records[split].sort(key=lambda record: record["path"])
        raw_to_process = []
    for raw_path in raw_to_process:
        with np.load(raw_path) as z: base={k:z[k] for k in z.files}
        physical=selected_jet_mask(base,jet_selection)
        for split,name in names.items():
            select=(base["split"]==split)&physical
            if not select.any(): continue
            part={k:v[select] for k,v in base.items()};pb=np.full(part["pid_values"].shape,-1,np.int16)
            for c,edge in enumerate(edges):
                v=part["pid_values"][...,c];valid=part["pid_target_valid"][...,c];pb[...,c][valid]=np.clip(np.searchsorted(np.asarray(edge[1:-1]),v[valid],side="right"),0,bins-1)
            part["pid_bins"]=pb
            if signed_species:
                part["pid_class"] = signed_reco_pid_classes(
                    part["target_reco_id"], species_target,
                )
                # Species supervision covers every physical daughter mapped by
                # the signed reconstructed-ID contract. Response heads remain
                # gated independently by pid_target_valid.
                part["pid_eligible"] = part["particle_mask"] & (part["pid_class"] >= 0)
            if split==0:
                species=part["pid_class"];eligible=part["pid_eligible"]&(species>=0)&(species<len(species_counts))
                species_counts += np.bincount(species[eligible],minlength=len(species_counts))
                sample=part["sample_id"];known=(sample>=0)&(sample<4)
                source_sample_counts += np.bincount(sample[known],minlength=4)
                flavour=part["jet_flavour"];known_flavour=(flavour>=0)&(flavour<3)
                flavour_probe_counts += np.bincount(flavour[known_flavour],minlength=3)
                for channel in range(len(PID_NAMES)):
                    channel_valid=part["pid_target_valid"][...,channel]&(pb[...,channel]>=0)
                    response_counts[channel] += np.bincount(pb[...,channel][channel_valid],minlength=bins)
            for key,stats in scalers.items(): apply_scaler(part,key,stats)
            for key in ("particle_features","pid_features","global_features","vertex_features","candidate_features"):
                part[key]=part[key].astype(storage_dtype)
            if int(cfg.get("model",{}).get("pid_continuous_dim",0)) == 0:
                part.pop("pid_values",None)
            dest=output/f"{name}_{raw_path.stem.removeprefix('raw_')}.npz"
            atomic_npz(dest,part);records[split].append({"path":dest.name,"jets":int(select.sum())});split_counts[split]+=int(select.sum())
    expected_outputs={output/record["path"] for group in records.values() for record in group}
    for name in names.values():
        for stale in output.glob(f"{name}_source*.npz"):
            if stale not in expected_outputs:
                stale.unlink()
    if reference_meta is None:
        marginal={"species_counts":species_counts.astype(int).tolist(),"species_entropy":entropy_from_counts(species_counts),"response_counts":response_counts.astype(int).tolist(),"response_entropy":[entropy_from_counts(row) for row in response_counts]}
        if signed_species:
            marginal["species_counts_scope"] = (
                "training split, physical particles with a supported signed reconstructed ID"
            )
        sample_counts={"source_classes":{"bb":int(source_sample_counts[0]),"cc":int(source_sample_counts[1]),"light":int(source_sample_counts[2]),"z_bb":int(source_sample_counts[3])},"flavour_probe":{"beauty":int(flavour_probe_counts[0]),"charm":int(flavour_probe_counts[1]),"light":int(flavour_probe_counts[2])}}
    else:
        marginal = reference_meta.get("marginal_target_statistics")
        sample_counts = reference_meta.get("sample_counts")
        if not isinstance(marginal, dict) or not isinstance(sample_counts, dict):
            raise ValueError("Reference preprocessing lacks frozen training statistics")
    feature_names={"particle":PARTICLE_NAMES,"pid":PID_NAMES,"global":GLOBAL_NAMES,"event_context":EVENT_CONTEXT_NAMES,"vertex":VERTEX_NAMES,"candidate":CANDIDATE_NAMES,"pairwise_inputs":PAIRWISE_INPUT_NAMES}
    # An external reference owns the immutable preprocessing fingerprint.  Some
    # legacy manifests stored polarity in a separate event_context_contract but
    # did not include it in feature_names.  Preserve that exact payload rather
    # than manufacturing a different hash for numerically identical inputs.
    if reference_meta is not None:
        feature_names = dict(reference_meta["feature_names"])
    meta={"format_version":RAW_FORMAT_VERSION,"created_utc":datetime.now(timezone.utc).isoformat(),"raw_processing_sha256":source_raw_fingerprint,"storage_dtype":np.dtype(storage_dtype).name,"pairwise_input_dtype":"float32","feature_names":feature_names,"feature_units":{"particle":{"log_pt":"log1p(MeV)","log_p":"log1p(MeV)","log_pt_fraction":"dimensionless","log_p_fraction":"dimensionless","delta_eta":"dimensionless","delta_phi":"rad","charge":"e","signed_log_ip":"signed log1p(mm)","signed_log_ip_raw":"signed log1p(source units; semantics unverified)","log1p_ipchi2":"dimensionless","log1p_track_chi2":"dimensionless","track_x_minus_pv":"mm","track_y_minus_pv":"mm","track_z_minus_pv":"mm","direction_x":"dimensionless","direction_y":"dimensionless","direction_z":"dimensionless","availability_flags":"boolean"},"global":{"log_jet_pt":"log1p(MeV)","jet_eta":"dimensionless","log_massless_jet_mass":"log1p(MeV)","log1p_n_particles":"log1p(count)","charged_fraction":"dimensionless","jet_width":"source units; observed MeV-like, semantics unverified","n_pvs":"count","topology_counts":"log1p(count)","topology_truncation_flags":"boolean"},"event_context":{"polarity":"categorical: -1=magnet down, +1=magnet up"},"vertex":{"position_and_distances":"mm","pointing":"dimensionless","log_pt_and_common_pion_masses":"log1p(MeV)","charge":"e"},"candidate":{"position_and_distances":"mm","pointing":"dimensionless","log_pt_and_common_pion_masses":"log1p(MeV)","charge":"e"},"analysis":{"qoverp":"source units and semantics unverified; excluded from encoder"}},"event_context_contract":{"polarity":{"required":True,"source_branch":"Polarity","stored_key":"polarity","allowed_values":{"-1":"down","1":"up"},"encoding":"categorical integer for a model-level learned embedding","standardized":False}},"pid_class_order":pid_class_order,"pid_quantile_edges":edges,"pid_bin_counts":[bins]*len(PID_NAMES),"pid_bins":bins,"robust_scalers":scalers,"marginal_target_statistics":marginal,"sample_counts":sample_counts,"fit_proxy":True,"missing_track_covariance":True,"topology_is_pid_blind":True,"legacy_sv_is_model_input":False,"analysis_only_prefix":"analysis_","dense_caps":{"particles":int(cfg.get("data",{}).get("max_particles",64)),"vertices":int(cfg.get("data",{}).get("max_vertices",64)),"candidates":int(cfg.get("data",{}).get("max_candidates",64))},"pairwise_features":["log_delta_r","log_common_pion_mass2","charge_product","log_doca_proxy","doca_available"]}
    meta["jet_selection"] = {
        "requires_at_least_one_stored_constituent": True,
        "raw_jets": raw_jets,
        "empty_jet_rows_excluded": empty_jet_rows,
        "fiducial_and_cap": None if jet_selection is None else {
            key: value for key, value in jet_selection.items() if key != "thresholds"
        },
    }
    if signed_species:
        contract = signed_reco_pid_contract(species_target)
        meta["pid_target_contract"] = contract
        meta["pid_target_sha256"] = pid_target_sha256(contract)
    fingerprint_payload=json.dumps({k:meta[k] for k in ("feature_names","pid_quantile_edges","robust_scalers")},sort_keys=True,separators=(",",":"))
    meta["preprocessing_sha256"]=hashlib.sha256(fingerprint_payload.encode()).hexdigest()
    if reference_meta is not None:
        expected_sha = reference_meta.get("preprocessing_sha256")
        if meta["preprocessing_sha256"] != expected_sha:
            raise ValueError(
                "Reconstructed reference preprocessing fingerprint differs: "
                f"{meta['preprocessing_sha256']} != {expected_sha}"
            )
        meta["preprocessing_fit"] = "external_training_reference"
        meta["preprocessing_reference"] = str(reference_preprocessing.resolve())
        meta["reused_finalized_shards"] = bool(reuse_finalized_shards)
    for split,name in names.items(): atomic_json(output/f"{name}.json",{"metadata":meta,"shards":records[split]})
    atomic_json(output/"preprocessing.json",meta)
    split_summary={names[k]:v for k,v in split_counts.items()}
    source_summaries=[]
    raw_source_ids={int(path.stem.split("_")[1].removeprefix("source")) for path in raw}
    for path in sorted(raw_root.glob("metadata_source*.json")):
        try:
            payload=json.loads(path.read_text())
            if payload.get("source_id") in raw_source_ids and payload.get("raw_processing_sha256")==source_raw_fingerprint:
                source_summaries.append(payload)
        except (OSError, json.JSONDecodeError):
            continue
    finalized_metadata={
        "created_utc":datetime.now(timezone.utc).isoformat(),
        "config":cfg,
        "input_files":[item for source in source_summaries for item in source.get("input_files",[])],
        "events":sum(int(source.get("events",0)) for source in source_summaries),
        "jets":sum(split_counts.values()),
        "raw_jets":raw_jets,
        "empty_jet_rows_excluded":empty_jet_rows,
        "source_jobs":len(source_summaries),
        "raw_shards":len(raw),
        "raw_input":str(raw_root),
        "raw_processing_sha256":source_raw_fingerprint,
        "split_counts":split_summary,
        "preprocessing":str((output/"preprocessing.json").resolve()),
        "preprocessing_sha256":meta["preprocessing_sha256"],
        "limitations":[
            "Track covariance matrices are absent; all local vertices are equal-weight straight-line geometric fit proxies.",
            "Legacy BDTTag summaries have no track membership or fitted position and are evaluation-only.",
        ],
    }
    atomic_json(output/"metadata.json",finalized_metadata)
    return {"raw_shards":len(raw),"split_counts":split_summary,"empty_jet_rows_excluded":empty_jet_rows,"preprocessing":str(output/"preprocessing.json"),"metadata":str(output/"metadata.json")}


def main() -> None:
    args=parse_args();dataset=yaml.safe_load(args.dataset_config.read_text());cfg=yaml.safe_load(args.transformer_config.read_text())
    if args.raw_input is not None and not args.finalize_only:
        raise ValueError("--raw-input is valid only together with --finalize-only")
    # Raw shards already contain their deterministic split assignment.  A
    # finalize-only inference pass therefore does not need split fractions in
    # the model config (the frozen v3 training config intentionally only names
    # its manifests).  Fraction validation remains mandatory while producing
    # raw shards, where those values actually determine the assignment.
    if not args.finalize_only:
        split_cfg=cfg.get("split",{});fractions=[float(split_cfg.get(name,0)) for name in ("train","validation","test","analysis")]
        if any(value<0 for value in fractions) or not math.isclose(sum(fractions),1.0,rel_tol=0,abs_tol=1e-9):
            raise ValueError(f"train/validation/test/analysis fractions must be non-negative and sum to one, got {fractions}")
    cfg.setdefault("data",{});cfg["data"]["max_particles"]=args.max_particles or cfg["data"].get("max_particles",64);cfg["data"]["max_vertices"]=args.max_vertices or cfg["data"].get("max_vertices",32);cfg["data"]["max_candidates"]=args.max_candidates or cfg["data"].get("max_candidates",32)
    shard_size=args.shard_size or cfg["data"].get("shard_size",16384);seed=args.seed if args.seed is not None else cfg.get("seed",2026)
    cfg["seed"]=int(seed)
    cfg["data"]["max_events_per_source"]=args.max_events
    fingerprint=processing_fingerprint(cfg,dataset)
    args.output.mkdir(parents=True,exist_ok=True)
    if args.finalize_only:
        print(json.dumps(finalize(
            args.output, cfg, dataset, args.force,
            args.reference_preprocessing, args.finalize_workers,
            args.raw_input, args.reuse_finalized_shards,
        ),indent=2));return
    import ROOT
    sources=selected_sources(dataset,args)
    ordered_entry_selection = bool(
        dataset.get("ordered_entry_selection", False)
        and (
            args.max_events is not None
            or dataset.get("selected_rows_are_limit", False)
            or any(meta.get("entry_stop") is not None for _, _, meta in sources)
        )
    )
    effective_threads = 1 if ordered_entry_selection else args.threads
    if not ordered_entry_selection:
        ROOT.EnableImplicitMT(args.threads)
    declare_ml_helpers(ROOT)
    started=time.time();produced=[];event_count=0
    for source_id,path,meta in sources:
        stale_raw=list(args.output.glob(f"raw_source{source_id:03d}_*.npz"))
        if stale_raw and not args.force: raise FileExistsError(f"Raw shards already exist for source {source_id}; pass --force to replace them")
        for stale in stale_raw: stale.unlink()
        retained=args.output/f"source{source_id:03d}_eventml.root"
        if retained.exists() and not args.force: raise FileExistsError(retained)
        if args.write_root:
            tmp=retained
        else:
            descriptor,tmp_name=tempfile.mkstemp(prefix=f"lhcb_ml_{source_id:03d}_",suffix=".root",dir="/tmp")
            os.close(descriptor);tmp=Path(tmp_name)
        try:
            events=snapshot_source(
                ROOT, source_id, path, dataset["tree_name"], tmp, cfg,
                dataset, meta, args.max_events,
            )
            expected_rows = meta.get("selected_rows")
            if expected_rows is not None and events != int(expected_rows):
                raise RuntimeError(
                    f"Source {source_id} retained {events} rows, expected "
                    f"{int(expected_rows)}"
                )
            event_count+=events
            produced += write_raw_shards(tmp,args.output,source_id,meta,cfg,shard_size,seed,fingerprint)
            atomic_json(args.output/f"metadata_source{source_id:03d}.json",{
                "created_utc":datetime.now(timezone.utc).isoformat(),"source_id":source_id,
                "raw_processing_sha256":fingerprint,"input_files":[str(path)],
                "events":events,"jets":2*events,"config":cfg,
                "input_schema":dataset.get("input_schema","canonical"),
                "row_filter":dataset.get("row_filter"),
                "entry_stop":meta.get("entry_stop"),
            })
        finally:
            if not args.write_root: tmp.unlink(missing_ok=True)
    # An array task writes only source-local raw shards. Shared train/validation/
    # test manifests and train-only statistics are created once with
    # --finalize-only after every task has completed, avoiding races.
    summary={"raw_shards_produced":len(produced)}
    if args.file_index is None:
        summary.update(finalize(
            args.output, cfg, dataset, args.force,
            args.reference_preprocessing, args.finalize_workers,
            reuse_finalized_shards=args.reuse_finalized_shards,
        ))
    metadata={"created_utc":datetime.now(timezone.utc).isoformat(),"raw_processing_sha256":fingerprint,"input_files":[str(x[1]) for x in sources],"events":event_count,"jets":2*event_count,"threads":effective_threads,"requested_threads":args.threads,"ordered_entry_selection":ordered_entry_selection,"wall_seconds":time.time()-started,"root_version":str(ROOT.gROOT.GetVersion()),"python_version":platform.python_version(),"git_commit":git_hash(),"config":cfg,"input_schema":dataset.get("input_schema","canonical"),"row_filter":dataset.get("row_filter"),"entry_stops":{str(x[0]):x[2].get("entry_stop") for x in sources},"limitations":["Track covariance matrices are absent; all local vertices are equal-weight straight-line geometric fit proxies.","Legacy BDTTag summaries have no track membership or fitted position and are evaluation-only."],**summary}
    metadata_name=f"metadata_source{sources[0][0]:03d}.json" if args.file_index is not None else "metadata.json"
    if args.file_index is not None:
        metadata["source_id"]=sources[0][0]
    atomic_json(args.output/metadata_name,metadata);print(json.dumps(metadata,indent=2))


if __name__ == "__main__": main()
