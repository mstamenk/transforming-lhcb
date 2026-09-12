"""Input feature definitions, ROOT-array transforms and train-only scaling."""
from __future__ import annotations
import math
from pathlib import Path
from typing import Any
import awkward as ak
import numpy as np
from pid_targets import signed_reco_pid_classes

PARTICLE_NAMES = [
    "log_pt", "log_p", "log_pt_fraction", "log_p_fraction", "delta_eta", "delta_phi", "charge",
    "signed_log_ip", "signed_log_ip_raw", "log1p_ipchi2", "log1p_track_chi2",
    "track_x_minus_pv", "track_y_minus_pv",
    "track_z_minus_pv", "direction_x", "direction_y", "direction_z",
    "track_available", "ip_available", "ip_raw_available", "ipchi2_available",
    "track_chi2_available",
]

PID_NAMES = ["NNe", "NNk", "NNp", "NNpi", "NNmu", "calo_ecal", "calo_hcal_over_ecal", "calo_e49", "calo_prs"]

PID_CLASS_ORDER = ["unknown", "electron", "muon", "photon", "charged_pion", "charged_kaon", "proton", "pi0", "neutral_kaon", "lambda"]

GLOBAL_NAMES = [
    "log_jet_pt", "jet_eta", "log_massless_jet_mass", "log1p_n_particles",
    "charged_fraction", "jet_width", "n_pvs", "log1p_n_displaced_tracks",
    "log1p_n_pair_vertices", "log1p_n_triplet_vertices",
    "log1p_n_quad_vertices", "log1p_n_chains", "vertices_truncated",
    "chains_truncated",
]

EVENT_CONTEXT_NAMES = ["polarity"]

VERTEX_NAMES = ["x_minus_pv", "y_minus_pv", "z_minus_pv", "rms", "max_track_distance", "signed_flight_pv", "ip_pv", "pointing", "log1p_min_ipchi2", "log1p_sum_ipchi2", "fit_proxy", "n_tracks", "log_pt", "log_common_pion_mass", "log_common_pion_corrected_mass", "charge"]

CANDIDATE_NAMES = ["parent_x_minus_pv", "parent_y_minus_pv", "parent_z_minus_pv", "parent_rms", "parent_max_track_distance", "parent_signed_flight_pv", "parent_ip_pv", "parent_pointing", "child_signed_flight", "child_pointing", "child_ip_pv", "fit_proxy", "child_n_tracks", "log_parent_pt", "log_child_common_pion_mass", "log_child_common_pion_corrected_mass", "log_parent_common_pion_mass", "log_parent_common_pion_corrected_mass", "charge"]

PAIRWISE_INPUT_NAMES = ["delta_eta", "delta_phi", "log_p", "charge", "state_x", "state_y", "state_z", "direction_x", "direction_y", "direction_z", "track_available"]

LEGACY_SV = ["fdrMin", "ptSvrJet", "nTrk", "nTrkJet", "drSvrJet", "absQSum", "m", "mCor", "fdChi2", "ipChi2Sum", "bdt0", "bdt1", "pass", "tau", "z", "pt", "backwards"]

def dense(chunk: ak.Array, key: str, dtype: Any) -> np.ndarray:
    return np.asarray(ak.to_numpy(chunk[key]), dtype=dtype)

def padded(chunk: ak.Array, key: str, width: int, dtype: Any, fill: float | int = 0) -> np.ndarray:
    values = ak.pad_none(chunk[key], width, axis=1, clip=True)
    return np.asarray(ak.to_numpy(ak.fill_none(values, fill)), dtype=dtype)

def signed_log(x: np.ndarray) -> np.ndarray:
    return np.sign(x) * np.log1p(np.abs(x))

def validated_polarity(chunk: ak.Array, source_meta: dict[str, Any]) -> np.ndarray:
    """Return the categorical magnet setting and reject missing/ambiguous values."""
    values = np.asarray(ak.to_numpy(chunk["Polarity"]))
    if values.ndim != 1:
        raise ValueError(f"Polarity must be one scalar per event, got shape {values.shape}")
    if values.dtype.kind not in "iu":
        raise TypeError(f"Polarity must have an integer dtype, got {values.dtype}")
    valid = (values == -1) | (values == 1)
    if not np.all(valid):
        raise ValueError(
            "Polarity must contain only -1 (magnet down) and +1 (magnet up); "
            f"observed {np.unique(values).tolist()}"
        )
    manifest_value = source_meta.get("polarity", "unknown")
    expected = {"down": -1, "up": 1}.get(manifest_value)
    if expected is not None and not np.all(values == expected):
        raise ValueError(
            f"ROOT Polarity disagrees with manifest polarity={manifest_value!r}: "
            f"observed {np.unique(values).tolist()}"
        )
    return values.astype(np.int8, copy=False)

def feature_payload(
    chunk: ak.Array,
    jet: int,
    source_meta: dict[str, Any],
    split_cfg: dict[str, float],
    seed: int,
    *,
    record_prefix: str | None = None,
    jet_prefix: str | None = None,
) -> dict[str, np.ndarray]:
    """Pack one jet record into the stable GPU-facing feature contract.

    The default prefixes preserve the production path exactly. Alternate
    prefixes are used by paired counterfactual datasets after the same compiled
    ``build_jet_record`` graph has been rerun on modified constituents.
    """
    p = record_prefix if record_prefix is not None else f"j{jet}_"
    jp = jet_prefix if jet_prefix is not None else f"Jet{jet}_"
    n = len(chunk)
    getf = lambda name: dense(chunk, p + name, np.float32)
    geti = lambda name: dense(chunk, p + name, np.int16)
    valid = geti("particle_valid").astype(bool); charge = geti("charge"); has_track = geti("has_track").astype(bool)
    ip, ipraw, chi2 = getf("ip"), getf("ip_raw"), getf("track_chi2")
    ip_valid = has_track & np.isfinite(ip) & (ip > -99)
    ipraw_valid = has_track & np.isfinite(ipraw) & (ipraw > -99)
    ipchi2_valid = has_track & np.isfinite(getf("log1p_ipchi2")) & (getf("log1p_ipchi2") >= 0)
    chi2_valid = has_track & np.isfinite(chi2) & (chi2 > -999)
    def usable(x: np.ndarray, available: np.ndarray) -> np.ndarray:
        return np.where(available, x, 0).astype(np.float32)
    jet_pt = dense(chunk, jp + "PT", np.float32)
    jet_eta = dense(chunk, jp + "ETA", np.float32)
    pt_fraction = getf("pt_fraction")
    momentum = np.expm1(getf("log_p"))
    jet_momentum = np.maximum(jet_pt * np.cosh(jet_eta), 1e-6)[:, None]
    log_pt_fraction = np.log(np.maximum(pt_fraction, 1e-8))
    log_p_fraction = np.log(np.maximum(momentum / jet_momentum, 1e-8))
    particle = np.stack([
        getf("log_pt"), getf("log_p"), log_pt_fraction, log_p_fraction, getf("delta_eta"), getf("delta_phi"), charge.astype(np.float32),
        signed_log(usable(ip, ip_valid)), signed_log(usable(ipraw, ipraw_valid)), usable(getf("log1p_ipchi2"), ipchi2_valid), np.log1p(np.maximum(usable(chi2, chi2_valid), 0)),
        usable(getf("state_dx"), has_track), usable(getf("state_dy"), has_track), usable(getf("state_dz"), has_track),
        usable(getf("dir_x"), has_track), usable(getf("dir_y"), has_track), usable(getf("dir_z"), has_track),
        has_track.astype(np.float32), ip_valid.astype(np.float32), ipraw_valid.astype(np.float32), ipchi2_valid.astype(np.float32),
        chi2_valid.astype(np.float32),
    ], axis=-1).astype(np.float32)
    base_valid = np.repeat(valid[..., None], 7, axis=-1)
    particle_feature_valid = np.concatenate([
        base_valid,
        ip_valid[..., None], ipraw_valid[..., None], ipchi2_valid[..., None], chi2_valid[..., None],
        np.repeat(has_track[..., None], 6, axis=-1),
        np.repeat(valid[..., None], 5, axis=-1),
    ], axis=-1)
    particle = np.where(particle_feature_valid, particle, 0).astype(np.float32)
    pid_values = np.stack([getf(x) for x in ["target_nne", "target_nnk", "target_nnp", "target_nnpi", "target_nnmu", "calo_ecal", "calo_hcal2ecal", "calo_e49", "calo_prs"]], axis=-1)
    calo_valid = [np.isfinite(getf(name)) & (getf(name) > -900) for name in ("calo_ecal", "calo_hcal2ecal", "calo_e49", "calo_prs")]
    pid_valid = np.stack([geti(x).astype(bool) for x in ["target_valid_e", "target_valid_k", "target_valid_p", "target_valid_pi", "target_valid_mu"]]
                         + calo_valid, axis=-1) & valid[..., None]
    pid_values = np.where(pid_valid, pid_values, 0).astype(np.float32)
    nv = geti("vertex_valid").shape[1]; nc = geti("chain_valid").shape[1]
    pvx = dense(chunk, jp + "OWNPV_X", np.float32)[:, None]; pvy = dense(chunk, jp + "OWNPV_Y", np.float32)[:, None]; pvz = dense(chunk, jp + "OWNPV_Z", np.float32)[:, None]
    vertex = np.stack([getf("vertex_x")-pvx, getf("vertex_y")-pvy, getf("vertex_z")-pvz, getf("vertex_rms"), getf("vertex_max_doca"), getf("vertex_flight_pv"), getf("vertex_ip_pv"), getf("vertex_pointing"), np.log1p(np.maximum(getf("vertex_min_ipchi2"), 0)), np.log1p(np.maximum(getf("vertex_sum_ipchi2"), 0)), getf("vertex_fit_proxy"), geti("vertex_n_tracks").astype(np.float32), np.log1p(np.maximum(getf("vertex_pt"), 0)), np.log1p(np.maximum(getf("vertex_mass_pi"), 0)), np.log1p(np.maximum(getf("vertex_corrected_mass_pi"), 0)), geti("vertex_charge").astype(np.float32)], axis=-1).astype(np.float32)
    candidate = np.stack([getf("chain_parent_x")-pvx, getf("chain_parent_y")-pvy, getf("chain_parent_z")-pvz, getf("chain_parent_rms"), getf("chain_parent_max_doca"), getf("chain_parent_flight_pv"), getf("chain_parent_ip_pv"), getf("chain_parent_pointing"), getf("chain_child_flight"), getf("chain_child_pointing"), getf("chain_child_ip_pv"), getf("chain_fit_proxy"), geti("chain_child_n_tracks").astype(np.float32), np.log1p(np.maximum(getf("chain_parent_pt"), 0)), np.log1p(np.maximum(getf("chain_child_mass_pi"), 0)), np.log1p(np.maximum(getf("chain_child_corrected_mass_pi"), 0)), np.log1p(np.maximum(getf("chain_parent_mass_pi"), 0)), np.log1p(np.maximum(getf("chain_parent_corrected_mass_pi"), 0)), geti("chain_charge").astype(np.float32)], axis=-1).astype(np.float32)
    vtracks = np.stack([geti(f"vertex_track{i}") for i in range(4)], axis=-1); ctracks = np.stack([geti(f"chain_track{i}") for i in range(4)] + [geti("chain_bachelor")], axis=-1)
    particle_axis = np.arange(valid.shape[1], dtype=np.int16)[None, None, :]
    vertex_track_mask = ((vtracks[..., None] == particle_axis) & (vtracks[..., None] >= 0)).any(2)
    candidate_track_mask = ((ctracks[..., None] == particle_axis) & (ctracks[..., None] >= 0)).any(2)
    child_index = geti("chain_child_vertex_index"); vertex_axis = np.arange(nv, dtype=np.int16)[None, None, :]
    candidate_vertex_mask = (child_index[..., None] == vertex_axis) & (child_index[..., None] >= 0)
    # Stored jet mass inherits reconstructed Daughters_E mass hypotheses. Build
    # the model-visible mass only from three-momenta instead.
    px = np.where(valid, getf("px"), 0.0)
    py = np.where(valid, getf("py"), 0.0)
    pz = np.where(valid, getf("pz"), 0.0)
    massless_energy = np.sqrt(px * px + py * py + pz * pz).sum(1)
    massless_mass2 = massless_energy * massless_energy - px.sum(1) ** 2 - py.sum(1) ** 2 - pz.sum(1) ** 2
    massless_jet_mass = np.sqrt(np.maximum(massless_mass2, 0.0))
    n_track_lines = dense(chunk, p+"n_track_lines", np.float32)
    n_pair_total = dense(chunk, p+"n_pair_total", np.float32)
    n_triplet_total = dense(chunk, p+"n_triplet_total", np.float32)
    n_quad_total = dense(chunk, p+"n_quad_total", np.float32)
    n_chain_total = dense(chunk, p+"n_chain_total", np.float32)
    globals_ = np.stack([
        np.log1p(np.maximum(jet_pt, 0)), dense(chunk, jp + "ETA", np.float32),
        np.log1p(massless_jet_mass), np.log1p(dense(chunk, p+"n_particles_input", np.float32)),
        (valid & (charge != 0)).sum(1)/np.maximum(valid.sum(1), 1),
        dense(chunk, jp + "width", np.float32), dense(chunk, "nPVs", np.float32),
        np.log1p(n_track_lines), np.log1p(n_pair_total), np.log1p(n_triplet_total),
        np.log1p(n_quad_total), np.log1p(n_chain_total),
        dense(chunk, p+"vertices_truncated", np.float32),
        dense(chunk, p+"chains_truncated", np.float32),
    ], axis=-1).astype(np.float32)
    run = dense(chunk, "runNumber", np.uint32); event = dense(chunk, "eventNumber", np.uint64); source = dense(chunk, "source_id", np.uint32)
    # Match the C++ SplitMix64 split exactly enough by invoking stable integer arithmetic in NumPy.
    mask64 = np.uint64(0xFFFFFFFFFFFFFFFF)
    def mix(x: np.ndarray) -> np.ndarray:
        with np.errstate(over="ignore"):
            x=(x+np.uint64(0x9e3779b97f4a7c15))&mask64;x=((x^(x>>np.uint64(30)))*np.uint64(0xbf58476d1ce4e5b9))&mask64;x=((x^(x>>np.uint64(27)))*np.uint64(0x94d049bb133111eb))&mask64;return x^(x>>np.uint64(31))
    # Mix each identifier at full width; explicit source ids are 32-bit hashes.
    event_uid = mix(mix(source.astype(np.uint64)) ^ mix(run.astype(np.uint64)) ^ mix(event))
    u=(mix(event_uid ^ np.uint64(seed))>>np.uint64(11)).astype(np.float64)/9007199254740992.0
    train_edge=float(split_cfg["train"]);validation_edge=train_edge+float(split_cfg["validation"]);test_edge=validation_edge+float(split_cfg["test"])
    split=np.full(n,3,np.int8);split[u<train_edge]=0;split[(u>=train_edge)&(u<validation_edge)]=1;split[(u>=validation_edge)&(u<test_edge)]=2
    sample_map={"bb":0,"z_bb":3,"cc":1,"light":2,"data":-1,"unknown":-1}; ptbin=source_meta.get("pt_bin_gev") or [math.nan,math.nan]
    mc_flavour = dense(chunk, jp + "mc_flavour", np.int32)
    abs_flavour = np.abs(mc_flavour)
    jet_flavour = np.full(n, -1, dtype=np.int8)
    jet_flavour[abs_flavour == 5] = 0
    jet_flavour[abs_flavour == 4] = 1
    jet_flavour[np.isin(abs_flavour, (1, 2, 3, 21))] = 2
    payload = {
        "particle_features": particle, "particle_feature_valid": particle_feature_valid, "particle_mask": valid, "particle_charge": charge.astype(np.int8), "particle_original_index": geti("original_index"),
        "pid_features": pid_values, "pid_values": pid_values.copy(), "pid_available": pid_valid, "pid_target_valid": pid_valid.copy(),
        "pid_eligible": valid & pid_valid.any(-1), "pid_class": geti("target_species").astype(np.int8), "target_reco_id": geti("target_reco_id").astype(np.int32),
        "global_features": globals_, "vertex_features": np.where(geti("vertex_valid")[...,None].astype(bool), vertex, 0), "vertex_mask": geti("vertex_valid").astype(bool), "vertex_track_mask": vertex_track_mask,
        "candidate_features": np.where(geti("chain_valid")[...,None].astype(bool), candidate, 0), "candidate_mask": geti("chain_valid").astype(bool), "candidate_track_mask": candidate_track_mask, "candidate_vertex_mask": candidate_vertex_mask,
        "event_uid": event_uid, "jet_uid": (event_uid*2+np.uint64(jet)), "run_number": run, "event_number": event, "source_id": source, "source_entry": dense(chunk,"source_entry",np.uint64), "jet_index": np.full(n,jet,np.int8), "split": split,
        "sample_id": np.full(n,sample_map.get(source_meta.get("sample","unknown"),-1),np.int8), "jet_flavour": jet_flavour, "polarity": validated_polarity(chunk, source_meta), "pt_bin_low": np.full(n,ptbin[0] if ptbin[0] is not None else np.nan,np.float32), "pt_bin_high": np.full(n,ptbin[1] if len(ptbin)>1 and ptbin[1] is not None else np.nan,np.float32),
    }
    if "raw_entry" in chunk.fields:
        payload["raw_entry"] = dense(chunk, "raw_entry", np.uint64)
    pairwise_inputs = np.stack([
        getf("delta_eta"), getf("delta_phi"), getf("log_p"), charge.astype(np.float32),
        usable(getf("state_dx"), has_track), usable(getf("state_dy"), has_track), usable(getf("state_dz"), has_track),
        usable(getf("dir_x"), has_track), usable(getf("dir_y"), has_track), usable(getf("dir_z"), has_track), has_track.astype(np.float32),
    ], axis=-1).astype(np.float32)
    payload["pairwise_inputs"] = np.where(valid[..., None], pairwise_inputs, 0).astype(np.float32)
    for name in ["n_particles_input","n_particles_stored","n_track_lines","n_pair_total","n_triplet_total","n_quad_total","n_chain_total","vertices_truncated","chains_truncated","particles_truncated"]: payload[name]=dense(chunk,p+name,np.int32)
    # Analysis-only quantities. They are deliberately absent from encoder feature matrices.
    for name in ["log_e","e_fraction","energy","qoverp","vertex_charge","vertex_px","vertex_py","vertex_pz","vertex_pt","vertex_mass_pi","vertex_corrected_mass_pi","chain_charge","chain_child_mass_pi","chain_child_corrected_mass_pi","chain_parent_mass_pi","chain_parent_corrected_mass_pi","chain_parent_pt"]: payload["analysis_"+name]=getf(name) if name not in {"vertex_charge","chain_charge"} else geti(name)
    for name in LEGACY_SV: payload["legacy_sv_"+name]=dense(chunk,p+"legacy_sv_"+name,np.float32)
    for name in ["PT","ETA","PHI","M","PE","nDaughters","OWNPV_X","OWNPV_Y","OWNPV_Z","ENDVERTEX_X","ENDVERTEX_Y","ENDVERTEX_Z","ORIVX_X","ORIVX_Y","ORIVX_Z","vtx_x","vtx_y","vtx_z","BDTTag_Tag","BDTTag_NbTag","mc_flavour","mc_deltaR","mc_PT"]:
        payload["analysis_jet_" + name.lower()] = dense(chunk, jp + name, np.float32)
    return payload

def quantile_edges(
    raw_paths: list[Path], bins: int, selection: dict[str, Any] | None = None,
) -> list[list[float]]:
    values=[[] for _ in PID_NAMES]
    for path in raw_paths:
        with np.load(path) as z:
            train=(z["split"]==0)&selected_jet_mask(z, selection); x=z["pid_values"][train]; valid=z["pid_target_valid"][train]
            for c in range(len(PID_NAMES)):
                if valid[...,c].any(): values[c].append(x[...,c][valid[...,c]])
    result=[]
    for chunks in values:
        x=np.concatenate(chunks) if chunks else np.asarray([0.,1.]); edge=np.quantile(x,np.linspace(0,1,bins+1));edge=np.maximum.accumulate(edge)
        result.append(edge.astype(float).tolist())
    return result

def robust_scalers(
    raw_paths: list[Path], seed: int, sample_per_shard: int = 20000,
    selection: dict[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    """Fit approximate median/IQR scaling using training objects only."""
    specifications = {
        "particle_features": (PARTICLE_NAMES, "particle_mask", "particle_feature_valid"),
        "pid_features": (PID_NAMES, "particle_mask", "pid_target_valid"),
        "global_features": (GLOBAL_NAMES, None, None),
        "vertex_features": (VERTEX_NAMES, "vertex_mask", None),
        "candidate_features": (CANDIDATE_NAMES, "candidate_mask", None),
    }
    result: dict[str, dict[str, Any]] = {}
    for group_index, (key, (names, mask_key, feature_valid_key)) in enumerate(specifications.items()):
        samples: list[list[np.ndarray]] = [[] for _ in names]
        for shard_index, path in enumerate(raw_paths):
            with np.load(path) as z:
                train = (z["split"] == 0) & selected_jet_mask(z, selection)
                values = np.asarray(z[key][train], dtype=np.float32)
                if mask_key is None:
                    rows = values.reshape(-1, values.shape[-1])
                    valid = np.isfinite(rows)
                else:
                    entity_valid = np.asarray(z[mask_key][train], dtype=bool)
                    rows = values[entity_valid]
                    if feature_valid_key is not None:
                        valid = np.asarray(z[feature_valid_key][train], dtype=bool)[entity_valid]
                    else:
                        valid = np.ones(rows.shape, dtype=bool)
                    valid &= np.isfinite(rows)
                rng = np.random.default_rng(seed + group_index * 1000003 + shard_index)
                for feature in range(len(names)):
                    candidates = rows[valid[:, feature], feature]
                    if len(candidates) > sample_per_shard:
                        candidates = candidates[rng.choice(len(candidates), sample_per_shard, replace=False)]
                    if len(candidates):
                        samples[feature].append(candidates.astype(np.float32, copy=False))
        center = np.zeros(len(names), dtype=np.float32)
        scale = np.ones(len(names), dtype=np.float32)
        low = np.zeros(len(names), dtype=np.float32)
        high = np.zeros(len(names), dtype=np.float32)
        counts = np.zeros(len(names), dtype=np.int64)
        passthrough = np.zeros(len(names), dtype=bool)
        for feature, name in enumerate(names):
            values = np.concatenate(samples[feature]) if samples[feature] else np.asarray([0.0], dtype=np.float32)
            counts[feature] = len(values)
            q25, median, q75 = np.quantile(values, [0.25, 0.5, 0.75])
            robust_scale = (q75 - q25) / 1.349
            if not np.isfinite(robust_scale) or robust_scale < 1e-6:
                robust_scale = float(np.std(values))
            if not np.isfinite(robust_scale) or robust_scale < 1e-6:
                robust_scale = 1.0
            center[feature], scale[feature] = median, robust_scale
            low[feature], high[feature] = np.quantile(values, [0.001, 0.999])
            if name == "charge" or name.endswith("_available"):
                center[feature], scale[feature], passthrough[feature] = 0.0, 1.0, True
        result[key] = {
            "names": names,
            "center": center.astype(float).tolist(),
            "scale": scale.astype(float).tolist(),
            "raw_q001": low.astype(float).tolist(),
            "raw_q999": high.astype(float).tolist(),
            "sample_counts": counts.astype(int).tolist(),
            "passthrough": passthrough.astype(bool).tolist(),
            "clip_standardized": 8.0,
            "fit_split": "train",
        }
    return result

def apply_scaler(part: dict[str, np.ndarray], key: str, stats: dict[str, Any]) -> None:
    values = np.asarray(part[key], dtype=np.float32)
    center = np.asarray(stats["center"], dtype=np.float32)
    scale = np.asarray(stats["scale"], dtype=np.float32)
    transformed = np.clip((values - center) / scale, -float(stats["clip_standardized"]), float(stats["clip_standardized"]))
    if key == "particle_features":
        valid = np.asarray(part["particle_feature_valid"], dtype=bool) & np.asarray(part["particle_mask"], dtype=bool)[..., None]
    elif key == "pid_features":
        valid = np.asarray(part["pid_target_valid"], dtype=bool) & np.asarray(part["particle_mask"], dtype=bool)[..., None]
    elif key == "vertex_features":
        valid = np.asarray(part["vertex_mask"], dtype=bool)[..., None]
    elif key == "candidate_features":
        valid = np.asarray(part["candidate_mask"], dtype=bool)[..., None]
    else:
        valid = np.isfinite(values)
    part[key] = np.where(valid, transformed, 0).astype(np.float32)

def fiducial_jet_mask(payload: Any, selection: dict[str, Any] | None) -> np.ndarray:
    physical = np.asarray(payload["particle_mask"], dtype=bool).any(axis=1)
    if not selection:
        return physical
    required = {"analysis_jet_pt", "analysis_jet_eta"}
    missing = required - set(payload.files if hasattr(payload, "files") else payload)
    if missing:
        raise KeyError(f"Raw shard lacks fiducial-selection fields: {sorted(missing)}")
    pt = np.asarray(payload["analysis_jet_pt"], dtype=np.float64)
    eta = np.asarray(payload["analysis_jet_eta"], dtype=np.float64)
    return (
        physical
        & (pt > float(selection["pt_min_mev"]))
        & (eta > float(selection["eta_min"]))
        & (eta < float(selection["eta_max"]))
    )

def selected_jet_mask(payload: Any, selection: dict[str, Any] | None) -> np.ndarray:
    base = fiducial_jet_mask(payload, None if selection is None else selection["config"])
    if selection is None:
        return base
    split_values = np.asarray(payload["split"], dtype=np.int8)
    hashes = jet_selection_hash(payload)
    result = np.zeros(len(base), dtype=bool)
    for split in np.unique(split_values[base]):
        split = int(split)
        mask = base & (split_values == split)
        threshold = selection["thresholds"].get(split)
        result |= mask if threshold is None else mask & (hashes <= np.uint64(threshold))
    return result

def _splitmix64_array(values: np.ndarray) -> np.ndarray:
    result = np.asarray(values, dtype=np.uint64).copy()
    result += np.uint64(0x9E3779B97F4A7C15)
    result = (result ^ (result >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    result = (result ^ (result >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return result ^ (result >> np.uint64(31))

def jet_selection_hash(payload: Any) -> np.ndarray:
    identity = np.asarray(payload["event_uid"], dtype=np.uint64).copy()
    identity ^= (np.asarray(payload["jet_index"], dtype=np.uint64) + np.uint64(1)) * np.uint64(0xD6E8FEB86659FD93)
    identity ^= (np.asarray(payload["source_id"], dtype=np.uint64) + np.uint64(1)) * np.uint64(0xA5A3564E27F8862B)
    return _splitmix64_array(identity)
