#!/usr/bin/env python3
"""Canonical reconstructed-PID target mappings used by dataset preparation.

The signed target is deliberately derived only from the reconstructed
``Daughters_ID`` hypothesis.  It is target metadata and is never an encoder
input.  Reconstructed charge remains visible to the encoder.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping

import numpy as np


SIGNED_RECO_PID_TARGET = "signed_reconstructed_id_v1"
SIGNED_RECO_PID_TARGET_WITH_V0 = "signed_reconstructed_id_with_v0_v2"
SIGNED_RECO_PID_TARGETS = (
    SIGNED_RECO_PID_TARGET,
    SIGNED_RECO_PID_TARGET_WITH_V0,
)
SIGNED_RECO_PID_CLASS_ORDER = (
    "electron_minus",
    "electron_plus",
    "muon_minus",
    "muon_plus",
    "photon",
    "pi0",
    "pion_plus",
    "pion_minus",
    "kaon_plus",
    "kaon_minus",
    "proton",
    "antiproton",
)
SIGNED_RECO_PID_CLASS_ORDER_WITH_V0 = (
    *SIGNED_RECO_PID_CLASS_ORDER,
    "k0s",
    "lambda",
    "antilambda",
)

# PDG sign conventions are used for the reconstructed hypothesis.  Photons
# are the sole deliberate sign merge: both +22 and the tuple's -22 coding map
# to one physical photon class.
_SIGNED_ID_TO_CLASS = {
    11: 0,
    -11: 1,
    13: 2,
    -13: 3,
    111: 5,
    211: 6,
    -211: 7,
    321: 8,
    -321: 9,
    2212: 10,
    -2212: 11,
}
_SIGNED_ID_TO_CLASS_WITH_V0 = {
    **_SIGNED_ID_TO_CLASS,
    310: 12,
    3122: 13,
    -3122: 14,
}
_EXPECTED_CHARGE = {
    11: -1,
    -11: 1,
    13: -1,
    -13: 1,
    22: 0,
    -22: 0,
    111: 0,
    211: 1,
    -211: -1,
    321: 1,
    -321: -1,
    2212: 1,
    -2212: -1,
    310: 0,
    3122: 0,
    -3122: 0,
}


def signed_reco_pid_class_order(
    target: str = SIGNED_RECO_PID_TARGET,
) -> tuple[str, ...]:
    if target == SIGNED_RECO_PID_TARGET:
        return SIGNED_RECO_PID_CLASS_ORDER
    if target == SIGNED_RECO_PID_TARGET_WITH_V0:
        return SIGNED_RECO_PID_CLASS_ORDER_WITH_V0
    raise ValueError(f"Unsupported signed reconstructed-PID target: {target}")


def signed_reco_pid_classes(
    reco_id: np.ndarray,
    target: str = SIGNED_RECO_PID_TARGET,
) -> np.ndarray:
    """Map signed reconstructed IDs to dense classes; unsupported IDs are -1."""
    values = np.asarray(reco_id)
    result = np.full(values.shape, -1, dtype=np.int8)
    result[np.abs(values) == 22] = 4
    mapping = (
        _SIGNED_ID_TO_CLASS_WITH_V0
        if target == SIGNED_RECO_PID_TARGET_WITH_V0
        else _SIGNED_ID_TO_CLASS
    )
    if target not in SIGNED_RECO_PID_TARGETS:
        raise ValueError(f"Unsupported signed reconstructed-PID target: {target}")
    for identifier, class_id in mapping.items():
        result[values == identifier] = class_id
    return result


def signed_reco_pid_expected_charge(reco_id: np.ndarray) -> np.ndarray:
    """Return the expected reconstructed charge, or 99 for unsupported IDs."""
    values = np.asarray(reco_id)
    result = np.full(values.shape, 99, dtype=np.int8)
    for identifier, charge in _EXPECTED_CHARGE.items():
        result[values == identifier] = charge
    return result


def signed_reco_pid_contract(
    target: str = SIGNED_RECO_PID_TARGET,
) -> dict[str, Any]:
    """Return the serialized target contract stored in manifests/checkpoints."""
    class_order = signed_reco_pid_class_order(target)
    mapping = {
        "11": "electron_minus",
        "-11": "electron_plus",
        "13": "muon_minus",
        "-13": "muon_plus",
        "abs(22)": "photon",
        "111": "pi0",
        "211": "pion_plus",
        "-211": "pion_minus",
        "321": "kaon_plus",
        "-321": "kaon_minus",
        "2212": "proton",
        "-2212": "antiproton",
    }
    if target == SIGNED_RECO_PID_TARGET_WITH_V0:
        mapping.update({"310": "k0s", "3122": "lambda", "-3122": "antilambda"})
    return {
        "name": target,
        "source": "target_reco_id (reconstructed Daughters_ID hypothesis)",
        "class_order": list(class_order),
        "mapping": mapping,
        "unsupported_class": -1,
        "eligibility": "physical particles with a supported reconstructed ID",
        "particle_charge_visible": True,
        "vertex_and_candidate_charge_visible": True,
        "generator_information": False,
    }


def pid_target_sha256(contract: Mapping[str, Any]) -> str:
    encoded = json.dumps(dict(contract), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def validate_signed_reco_pid_metadata(metadata: Mapping[str, Any]) -> None:
    """Fail closed if a signed-target manifest advertises a different mapping."""
    contract = metadata.get("pid_target_contract", {})
    if not isinstance(contract, Mapping):
        return
    target = contract.get("name")
    if target not in SIGNED_RECO_PID_TARGETS:
        return
    expected = signed_reco_pid_contract(str(target))
    if dict(contract) != expected:
        raise ValueError("Signed reconstructed-PID target contract does not match the canonical mapping")
    class_order = list(metadata.get("pid_class_order", []))
    if class_order != list(signed_reco_pid_class_order(str(target))):
        raise ValueError(
            "Signed reconstructed-PID class order differs from the canonical order: "
            f"{class_order}"
        )
    observed_hash = metadata.get("pid_target_sha256")
    expected_hash = pid_target_sha256(expected)
    if observed_hash != expected_hash:
        raise ValueError(
            "Signed reconstructed-PID target fingerprint is missing or inconsistent: "
            f"observed={observed_hash!r}, expected={expected_hash!r}"
        )
