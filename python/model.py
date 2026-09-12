#!/usr/bin/env python3
"""Heterogeneous, permutation-equivariant transformer for reconstructed jets.

Particle, PID-blind fitted-vertex, cascade/topology and jet tokens are processed
together. Track membership is injected through incidence matrices and also used
as a ParT-style pairwise attention bias. No generator label enters the encoder;
the optional flavour head is an auxiliary diagnostic probe only.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Sequence

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class ModelConfig:
    particle_dim: int
    pid_dim: int
    architecture: str = "heterogeneous_particle_transformer"
    global_dim: int = 0
    vertex_dim: int = 0
    candidate_dim: int = 0
    pairwise_dim: int = 0
    # Magnet polarity is an event-level detector condition. It is kept
    # separate from standardized physics globals and embedded categorically.
    # False preserves the parameter/state-dict contract of older checkpoints.
    use_polarity_conditioning: bool = False
    d_model: int = 128
    n_heads: int = 8
    n_layers: int = 4
    dim_feedforward: int = 512
    dropout: float = 0.1
    pid_num_classes: int = 6
    # Optional diagnostic head trained with tempered reconstructed-species
    # weights. The ordinary PID head remains the natural-prior likelihood used
    # for anomaly scoring.
    pid_balanced_auxiliary: bool = False
    # The natural-prior head can be calibrated without steering the shared
    # encoder toward common classes.  In that mode, only the exactly balanced
    # signed-species head contributes species gradients to the representation.
    pid_natural_head_detach_encoder: bool = False
    pid_bin_counts: list[int] = field(default_factory=list)
    pid_continuous_dim: int = 0
    origin_num_classes: int = 0
    vertex_num_classes: int = 0
    candidate_num_classes: int = 0
    flavour_num_classes: int = 3
    # Legacy probe/fine-tune checkpoints used the common two-layer MLP head.
    # Fully supervised flavour training opts into affine readouts so its head
    # capacity matches the external frozen-embedding probes.
    flavour_heads_affine: bool = False
    beauty_sign_num_classes: int = 0
    charm_sign_num_classes: int = 0
    # Five conditional binary generator-flavour readouts used only by the
    # fully supervised ceiling.  They are opt-in so all historical state-dict
    # contracts remain unchanged.
    b_vs_c_num_classes: int = 0
    b_vs_light_num_classes: int = 0
    c_vs_light_num_classes: int = 0
    b_vs_bbar_num_classes: int = 0
    c_vs_cbar_num_classes: int = 0
    topology_target_dim: int = 0
    # Opt-in missing-track completion.  Every default is disabled so models
    # built from older configuration dictionaries retain exactly their former
    # parameters, token order and strict state-dict contract.
    enable_semantic_token: bool = False
    enable_completeness_token: bool = False
    missing_track_max_queries: int = 0
    missing_track_count_classes: int = 0
    # Optional exactly class-balanced count head.  The natural-prior head stays
    # separate because its logits define the calibrated deployable score.
    missing_track_count_balanced_auxiliary: bool = False
    missing_track_count_natural_head_detach_encoder: bool = False
    missing_track_particle_dim: int = 0
    missing_track_pid_classes: int = 0
    missing_track_pid_balanced_auxiliary: bool = False
    # The natural-prior removed-particle PID readout is optional.  It defaults
    # to enabled so historical missing-track checkpoints retain their exact
    # state-dict contract, while streamlined models can train only the
    # class-balanced PID readout.
    missing_track_pid_natural_head_enabled: bool = True
    missing_track_pid_natural_head_detach_encoder: bool = False
    missing_track_charge_classes: int = 0
    missing_track_summary_dim: int = 0
    missing_track_corruption_classes: int = 0
    missing_track_topology_dim: int = 0
    missing_track_decoder_layers: int = 2
    # Label-free metric-learning projections. The jet projection consumes the
    # semantic token; the substructure projection consumes a masked mean of
    # contextual particle tokens. A zero dimension preserves old checkpoints.
    retrieval_dim: int = 0  # Retained for loading existing MC checkpoint configs.
    local_feature_indices: list[int] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ModelConfig":
        return cls(**dict(value))


class FeatureEncoder(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim), nn.Linear(input_dim, output_dim), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(output_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SetAttentionBlock(nn.Module):
    """Pre-norm transformer block accepting a per-head additive bias."""

    def __init__(self, d_model: int, n_heads: int, dim_ff: int, dropout: float) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.norm1, self.norm2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.attention = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_ff), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim_ff, d_model), nn.Dropout(dropout),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, valid: torch.Tensor, bias: torch.Tensor | None) -> torch.Tensor:
        normed = self.norm1(x)
        attn_mask = None
        if bias is not None:
            attn_mask = bias.to(normed.dtype).flatten(0, 1)
        attended = self.attention(normed, normed, normed, attn_mask=attn_mask, need_weights=False)[0]
        x = x + self.dropout(attended)
        x = x + self.ff(self.norm2(x))
        return x.masked_fill(~valid[..., None], 0.0)


class MissingTrackSetDecoder(nn.Module):
    """Decode an unordered set of absent constituents from visible tokens.

    The learned query ordering has no physical meaning.  Training is expected
    to match target constituents to queries with a permutation-invariant
    assignment (for example, Hungarian matching) and supervise unmatched
    queries as empty slots.
    """

    def __init__(
        self, d_model: int, n_heads: int, dim_ff: int, dropout: float,
        n_layers: int,
    ) -> None:
        super().__init__()
        if n_layers <= 0:
            raise ValueError("missing_track_decoder_layers must be positive")
        self.layers = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=dim_ff,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            for _ in range(n_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)

    def forward(
        self, queries: torch.Tensor, memory: torch.Tensor,
        memory_valid: torch.Tensor,
    ) -> torch.Tensor:
        if memory_valid.shape != memory.shape[:2]:
            raise ValueError(
                "memory_valid must match the first two memory dimensions: "
                f"valid={tuple(memory_valid.shape)}, memory={tuple(memory.shape)}"
            )
        value = queries
        padding_mask = ~memory_valid.bool()
        for layer in self.layers:
            value = layer(
                value,
                memory,
                memory_key_padding_mask=padding_mask,
            )
        return self.final_norm(value)


def _masked_incidence_mean(incidence: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    weights = incidence.to(values.dtype)
    return torch.bmm(weights, values) / weights.sum(-1, keepdim=True).clamp_min(1.0)


def _validated_polarity_indices(
    batch: Mapping[str, torch.Tensor], batch_size: int,
) -> torch.Tensor:
    if "polarity" not in batch:
        raise KeyError(
            "Polarity conditioning is enabled, but the batch has no 'polarity' field"
        )
    polarity = batch["polarity"]
    if polarity.ndim != 1 or polarity.shape[0] != batch_size:
        raise ValueError(
            "polarity must have shape [batch] with values -1 (down) or +1 (up); "
            f"got {tuple(polarity.shape)}"
        )
    valid = (polarity == -1) | (polarity == 1)
    # Dataset/collation validate on CPU; this assertion also protects direct
    # model callers without synchronizing every CUDA forward with the host.
    message = "polarity must contain only -1 (down) and +1 (up)"
    if polarity.is_cuda:
        torch._assert_async(valid.all(), message)
    elif not bool(valid.all()):
        observed = torch.unique(polarity.detach()).tolist()
        raise ValueError(f"{message}; observed {observed}")
    return (polarity > 0).long()


def build_pairwise_features(inputs: torch.Tensor) -> torch.Tensor:
    """Build ParT-style pair observables on the accelerator.

    Input fields are eta/phi relative to the jet, log momentum, charge, track
    state, unit direction and track availability. The output is
    log(delta-R), common-pion log mass squared, charge product, log DOCA and a
    DOCA-valid bit. No PID hypothesis enters this calculation.
    """
    if inputs.shape[-1] != 11:
        raise ValueError(f"pairwise_inputs must have 11 fields, got {inputs.shape[-1]}")
    eta, phi = inputs[..., 0], inputs[..., 1]
    delta_eta = eta[:, :, None] - eta[:, None, :]
    delta_phi_raw = phi[:, :, None] - phi[:, None, :]
    delta_phi = torch.atan2(torch.sin(delta_phi_raw), torch.cos(delta_phi_raw))
    log_delta_r = torch.log(torch.sqrt(delta_eta.square() + delta_phi.square()).clamp_min(1e-6))

    momentum_magnitude = torch.expm1(inputs[..., 2]).clamp_min(0.0)
    direction = inputs[..., 7:10]
    momentum = momentum_magnitude[..., None] * direction
    pion_mass_mev = 139.57039
    energy = torch.sqrt(momentum_magnitude.square() + pion_mass_mev * pion_mass_mev)
    pair_energy = energy[:, :, None] + energy[:, None, :]
    pair_momentum = momentum[:, :, None, :] + momentum[:, None, :, :]
    mass2 = (pair_energy.square() - pair_momentum.square().sum(-1)).clamp_min(0.0)
    log_mass2 = torch.log1p(mass2 / 1.0e6)

    charge = inputs[..., 3]
    charge_product = charge[:, :, None] * charge[:, None, :]
    state, track_valid = inputs[..., 4:7], inputs[..., 10] > 0.5
    state_delta = state[:, None, :, :] - state[:, :, None, :]
    first_direction = direction[:, :, None, :].expand_as(state_delta)
    second_direction = direction[:, None, :, :].expand_as(state_delta)
    direction_cross = torch.linalg.cross(first_direction, second_direction, dim=-1)
    cross_norm = torch.linalg.vector_norm(direction_cross, dim=-1)
    skew_doca = (state_delta * direction_cross).sum(-1).abs() / cross_norm.clamp_min(1e-6)
    parallel_doca = torch.linalg.vector_norm(torch.linalg.cross(state_delta, first_direction, dim=-1), dim=-1)
    doca = torch.where(cross_norm > 1e-5, skew_doca, parallel_doca)
    doca_valid = track_valid[:, :, None] & track_valid[:, None, :]
    log_doca = torch.where(doca_valid, torch.log1p(doca.clamp_min(0.0)), torch.zeros_like(doca))
    return torch.stack((log_delta_r, log_mass2, charge_product, log_doca, doca_valid.to(inputs.dtype)), dim=-1)


class JetTopologyTransformer(nn.Module):
    """Context model for masked PID, topology modelling and flavour mapping."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        if config.d_model % config.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        nonnegative = {
            "missing_track_max_queries": config.missing_track_max_queries,
            "missing_track_count_classes": config.missing_track_count_classes,
            "missing_track_particle_dim": config.missing_track_particle_dim,
            "missing_track_pid_classes": config.missing_track_pid_classes,
            "missing_track_charge_classes": config.missing_track_charge_classes,
            "missing_track_summary_dim": config.missing_track_summary_dim,
            "missing_track_corruption_classes": config.missing_track_corruption_classes,
            "missing_track_topology_dim": config.missing_track_topology_dim,
        }
        invalid = {name: value for name, value in nonnegative.items() if value < 0}
        if invalid:
            raise ValueError(f"Missing-track model dimensions must be nonnegative: {invalid}")
        query_outputs = (
            config.missing_track_particle_dim
            or config.missing_track_pid_classes
            or config.missing_track_charge_classes
            or config.missing_track_topology_dim
        )
        if query_outputs and not config.missing_track_max_queries:
            raise ValueError(
                "Per-query missing-track outputs require missing_track_max_queries > 0"
            )
        if config.retrieval_dim:
            raise ValueError("This trainer supports the MC architecture only")
        self.config = config
        d, drop = config.d_model, config.dropout
        self.particle_encoder = FeatureEncoder(config.particle_dim, d, drop)
        # Availability is concatenated so detector acceptance is distinguishable
        # from a numerical response of zero. Masked channels receive neither.
        self.pid_encoder = FeatureEncoder(2 * config.pid_dim, d, drop)
        self.vertex_encoder = FeatureEncoder(config.vertex_dim, d, drop) if config.vertex_dim else None
        self.candidate_encoder = FeatureEncoder(config.candidate_dim, d, drop) if config.candidate_dim else None
        self.global_encoder = FeatureEncoder(config.global_dim, d, drop) if config.global_dim else None
        self.polarity_embedding = nn.Embedding(2, d) if config.use_polarity_conditioning else None
        self.jet_token = nn.Parameter(torch.empty(1, 1, d))
        self.semantic_token = (
            nn.Parameter(torch.empty(1, 1, d))
            if config.enable_semantic_token else None
        )
        self.completeness_token = (
            nn.Parameter(torch.empty(1, 1, d))
            if config.enable_completeness_token else None
        )
        self.pid_mask_embedding = nn.Parameter(torch.empty(1, 1, d))
        self.token_type = nn.Embedding(4, d)  # jet, particle, vertex, cascade
        self.graph_particle = nn.Linear(d, d, bias=False)
        self.graph_vertex = nn.Linear(d, d, bias=False)
        self.graph_candidate = nn.Linear(d, d, bias=False)
        self.pairwise_projection = nn.Linear(config.pairwise_dim, config.n_heads, bias=False) if config.pairwise_dim else None
        # Shared-vertex particle pairs, shared-cascade particle pairs, direct
        # particle-vertex, particle-cascade, and cascade-vertex relations.
        self.relation_bias = nn.Parameter(torch.zeros(5, config.n_heads))
        self.particle_feature_mask_embedding = nn.Parameter(torch.empty(1, 1, d))
        self.vertex_feature_mask_embedding = nn.Parameter(torch.empty(1, 1, d))
        self.candidate_feature_mask_embedding = nn.Parameter(torch.empty(1, 1, d))
        self.blocks = nn.ModuleList([
            SetAttentionBlock(d, config.n_heads, config.dim_feedforward, drop)
            for _ in range(config.n_layers)
        ])
        self.final_norm = nn.LayerNorm(d)
        self.missing_query_tokens = (
            nn.Parameter(torch.empty(1, config.missing_track_max_queries, d))
            if config.missing_track_max_queries else None
        )
        self.missing_query_decoder = (
            MissingTrackSetDecoder(
                d, config.n_heads, config.dim_feedforward, drop,
                config.missing_track_decoder_layers,
            )
            if config.missing_track_max_queries else None
        )

        def head(out: int) -> nn.Module:
            return nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.LayerNorm(d), nn.Linear(d, out))


        self.pid_class_head = head(config.pid_num_classes) if config.pid_num_classes else None
        self.pid_balanced_class_head = (
            head(config.pid_num_classes)
            if config.pid_num_classes and config.pid_balanced_auxiliary else None
        )
        self.pid_bin_heads = nn.ModuleList([head(size) for size in config.pid_bin_counts])
        self.pid_continuous_head = head(2 * config.pid_continuous_dim) if config.pid_continuous_dim else None
        self.origin_head = head(config.origin_num_classes) if config.origin_num_classes else None
        self.vertex_class_head = head(config.vertex_num_classes) if config.vertex_num_classes else None
        self.candidate_class_head = head(config.candidate_num_classes) if config.candidate_num_classes else None
        flavour_head = (lambda out: nn.Linear(d, out)) if config.flavour_heads_affine else head
        self.flavour_head = flavour_head(config.flavour_num_classes) if config.flavour_num_classes else None
        self.beauty_sign_head = (
            flavour_head(config.beauty_sign_num_classes)
            if config.beauty_sign_num_classes else None
        )
        self.charm_sign_head = (
            flavour_head(config.charm_sign_num_classes)
            if config.charm_sign_num_classes else None
        )
        self.b_vs_c_head = (
            flavour_head(config.b_vs_c_num_classes)
            if config.b_vs_c_num_classes else None
        )
        self.b_vs_light_head = (
            flavour_head(config.b_vs_light_num_classes)
            if config.b_vs_light_num_classes else None
        )
        self.c_vs_light_head = (
            flavour_head(config.c_vs_light_num_classes)
            if config.c_vs_light_num_classes else None
        )
        self.b_vs_bbar_head = (
            flavour_head(config.b_vs_bbar_num_classes)
            if config.b_vs_bbar_num_classes else None
        )
        self.c_vs_cbar_head = (
            flavour_head(config.c_vs_cbar_num_classes)
            if config.c_vs_cbar_num_classes else None
        )
        self.topology_head = head(config.topology_target_dim) if config.topology_target_dim else None
        self.particle_feature_head = head(config.particle_dim)
        self.vertex_feature_head = head(config.vertex_dim) if config.vertex_dim else None
        self.candidate_feature_head = head(config.candidate_dim) if config.candidate_dim else None
        self.missing_count_head = (
            head(config.missing_track_count_classes)
            if config.missing_track_count_classes else None
        )
        self.missing_count_balanced_head = (
            head(config.missing_track_count_classes)
            if (
                config.missing_track_count_classes
                and config.missing_track_count_balanced_auxiliary
            ) else None
        )
        self.corruption_type_head = (
            head(config.missing_track_corruption_classes)
            if config.missing_track_corruption_classes else None
        )
        self.removed_summary_head = (
            head(2 * config.missing_track_summary_dim)
            if config.missing_track_summary_dim else None
        )
        self.missing_query_existence_head = (
            head(1) if config.missing_track_max_queries else None
        )
        self.missing_query_feature_head = (
            head(2 * config.missing_track_particle_dim)
            if config.missing_track_particle_dim else None
        )
        self.missing_pid_head = (
            head(config.missing_track_pid_classes)
            if (
                config.missing_track_pid_classes
                and config.missing_track_pid_natural_head_enabled
            ) else None
        )
        self.missing_pid_balanced_head = (
            head(config.missing_track_pid_classes)
            if (
                config.missing_track_pid_classes
                and config.missing_track_pid_balanced_auxiliary
            ) else None
        )
        self.missing_charge_head = (
            head(config.missing_track_charge_classes)
            if config.missing_track_charge_classes else None
        )
        self.missing_topology_head = (
            head(config.missing_track_topology_dim)
            if config.missing_track_topology_dim else None
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.jet_token, std=0.02)
        if self.semantic_token is not None:
            nn.init.normal_(self.semantic_token, std=0.02)
        if self.completeness_token is not None:
            nn.init.normal_(self.completeness_token, std=0.02)
        if self.missing_query_tokens is not None:
            nn.init.normal_(self.missing_query_tokens, std=0.02)
        nn.init.normal_(self.pid_mask_embedding, std=0.02)
        nn.init.normal_(self.particle_feature_mask_embedding, std=0.02)
        nn.init.normal_(self.vertex_feature_mask_embedding, std=0.02)
        nn.init.normal_(self.candidate_feature_mask_embedding, std=0.02)
        nn.init.normal_(self.token_type.weight, std=0.02)
        if self.polarity_embedding is not None:
            nn.init.normal_(self.polarity_embedding.weight, std=0.02)

    def _polarity_context(
        self, batch: Mapping[str, torch.Tensor], batch_size: int,
    ) -> torch.Tensor | None:
        if self.polarity_embedding is None:
            return None
        # Map magnet down/up to categorical indices 0/1. The same event-level
        # condition is added to every token before contextual attention.
        return self.polarity_embedding(
            _validated_polarity_indices(batch, batch_size)
        )[:, None]

    def _graph_exchange(
        self, particles: torch.Tensor, vertices: torch.Tensor | None,
        candidates: torch.Tensor | None, batch: Mapping[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        particle_context = torch.zeros_like(particles)
        if vertices is not None and "vertex_track_mask" in batch:
            incidence = batch["vertex_track_mask"].to(particles.dtype)
            particle_context += torch.bmm(incidence.transpose(1, 2), vertices) / incidence.sum(1)[..., None].clamp_min(1.0)
            vertices = vertices + self.graph_vertex(_masked_incidence_mean(incidence, particles))
        if candidates is not None and "candidate_track_mask" in batch:
            incidence = batch["candidate_track_mask"].to(particles.dtype)
            particle_context += torch.bmm(incidence.transpose(1, 2), candidates) / incidence.sum(1)[..., None].clamp_min(1.0)
            candidates = candidates + self.graph_candidate(_masked_incidence_mean(incidence, particles))
        particles = particles + self.graph_particle(particle_context)
        return particles, vertices, candidates

    def _attention_bias(
        self, batch: Mapping[str, torch.Tensor], token_valid: torch.Tensor,
        n_particles: int, n_vertices: int, n_candidates: int,
        n_prefix_tokens: int = 1,
    ) -> torch.Tensor:
        bsz, length = token_valid.shape
        bias = torch.zeros(bsz, self.config.n_heads, length, length, device=token_valid.device)
        particle_bias = torch.zeros(bsz, self.config.n_heads, n_particles, n_particles, device=token_valid.device)
        if self.pairwise_projection is not None:
            pairwise = batch.get("pairwise_features")
            if pairwise is None and "pairwise_inputs" in batch:
                pairwise = build_pairwise_features(batch["pairwise_inputs"])
            if pairwise is None:
                raise KeyError("Model expects pairwise_features or pairwise_inputs")
            particle_bias += self.pairwise_projection(pairwise).permute(0, 3, 1, 2)
        if "vertex_track_mask" in batch:
            shared = torch.bmm(batch["vertex_track_mask"].float().transpose(1, 2), batch["vertex_track_mask"].float()).clamp_max(1)
            particle_bias += shared[:, None] * self.relation_bias[0][None, :, None, None]
        if "candidate_track_mask" in batch:
            shared = torch.bmm(batch["candidate_track_mask"].float().transpose(1, 2), batch["candidate_track_mask"].float()).clamp_max(1)
            particle_bias += shared[:, None] * self.relation_bias[1][None, :, None, None]
        particle_start = n_prefix_tokens
        vertex_start = particle_start + n_particles
        bias[:, :, particle_start:vertex_start, particle_start:vertex_start] = particle_bias
        candidate_start = vertex_start + n_vertices

        def add_symmetric(left: slice, right: slice, relation: torch.Tensor, weight: torch.Tensor) -> None:
            direct = relation[:, None].to(bias.dtype) * weight[None, :, None, None]
            bias[:, :, left, right] += direct
            bias[:, :, right, left] += direct.transpose(-1, -2)

        if n_vertices and "vertex_track_mask" in batch:
            add_symmetric(
                slice(particle_start, vertex_start), slice(vertex_start, candidate_start),
                batch["vertex_track_mask"].transpose(1, 2), self.relation_bias[2],
            )
        if n_candidates and "candidate_track_mask" in batch:
            add_symmetric(
                slice(particle_start, vertex_start), slice(candidate_start, candidate_start+n_candidates),
                batch["candidate_track_mask"].transpose(1, 2), self.relation_bias[3],
            )
        if n_vertices and n_candidates and "candidate_vertex_mask" in batch:
            add_symmetric(
                slice(vertex_start, candidate_start), slice(candidate_start, candidate_start+n_candidates),
                batch["candidate_vertex_mask"].transpose(1, 2), self.relation_bias[4],
            )
        return bias.masked_fill(~token_valid[:, None, None, :], float("-inf"))

    def forward(self, batch: Mapping[str, torch.Tensor], pid_mask: torch.Tensor | None = None) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        """Encode a reconstructed jet and emit enabled prediction heads.

        Missing-track outputs are opt-in and use the following stable shapes,
        where ``Q = missing_track_max_queries``:

        * ``missing_count_logits``: ``[batch, count_classes]``;
        * ``corruption_type_logits``: ``[batch, corruption_classes]``;
        * ``removed_summary_mean/log_scale``: ``[batch, summary_dim]``;
        * ``missing_query_embedding``: ``[batch, Q, d_model]``;
        * ``missing_query_existence_logits``: ``[batch, Q]``;
        * ``missing_query_features_mean/log_scale``: ``[batch, Q, particle_dim]``;
        * ``missing_pid[_balanced]_logits``: ``[batch, Q, pid_classes]``;
        * ``missing_charge_logits``: ``[batch, Q, charge_classes]``;
        * ``missing_topology_prediction``: ``[batch, Q, topology_dim]``.

        Query-to-visible-topology compatibility is additionally returned as
        ``missing_vertex_logits`` and ``missing_candidate_logits`` whenever
        those token families exist.  Targets and matching are deliberately a
        trainer concern; the decoder's learned slot ordering is not physical.
        """
        particle_valid = batch["particle_mask"].bool()
        if pid_mask is None: pid_mask = torch.zeros_like(particle_valid)
        pid_mask = pid_mask & particle_valid
        particle_feature_mask = batch.get("particle_feature_mask", torch.zeros_like(particle_valid)).bool() & particle_valid
        particle_input = batch["particle_features"].masked_fill(particle_feature_mask[..., None], 0)
        particles = self.particle_encoder(particle_input)
        particles = particles + particle_feature_mask[..., None] * self.particle_feature_mask_embedding
        if "pid_available" in batch:
            pid_available = batch["pid_available"].bool()
        elif "pid_target_valid" in batch and batch["pid_target_valid"].shape == batch["pid_features"].shape:
            pid_available = batch["pid_target_valid"].bool()
        else:
            pid_available = torch.isfinite(batch["pid_features"]) & (batch["pid_features"].abs() < 900)
        visible = pid_available & ~pid_mask[..., None]
        pid_value = torch.nan_to_num(batch["pid_features"]) * visible
        particles = particles + self.pid_encoder(torch.cat((pid_value, visible.to(pid_value.dtype)), -1))
        particles = particles + pid_mask[..., None] * self.pid_mask_embedding

        vertices = None
        if self.vertex_encoder is not None and "vertex_features" in batch:
            vertex_feature_mask = batch.get("vertex_feature_mask", torch.zeros_like(batch["vertex_mask"])).bool() & batch["vertex_mask"].bool()
            vertices = self.vertex_encoder(batch["vertex_features"].masked_fill(vertex_feature_mask[..., None], 0))
            vertices = vertices + vertex_feature_mask[..., None] * self.vertex_feature_mask_embedding
        candidates = None
        if self.candidate_encoder is not None and "candidate_features" in batch:
            candidate_feature_mask = batch.get("candidate_feature_mask", torch.zeros_like(batch["candidate_mask"])).bool() & batch["candidate_mask"].bool()
            candidates = self.candidate_encoder(batch["candidate_features"].masked_fill(candidate_feature_mask[..., None], 0))
            candidates = candidates + candidate_feature_mask[..., None] * self.candidate_feature_mask_embedding
        particles, vertices, candidates = self._graph_exchange(particles, vertices, candidates, batch)
        bsz, n_particles = particles.shape[:2]
        jet = self.jet_token.expand(bsz, -1, -1)
        global_context = None
        if self.global_encoder is not None and "global_features" in batch:
            global_context = self.global_encoder(batch["global_features"])[:, None]
            jet = jet + global_context

        prefix = [jet]
        if self.semantic_token is not None:
            semantic = self.semantic_token.expand(bsz, -1, -1)
            if global_context is not None:
                semantic = semantic + global_context
            prefix.append(semantic)
        if self.completeness_token is not None:
            completeness = self.completeness_token.expand(bsz, -1, -1)
            if global_context is not None:
                completeness = completeness + global_context
            prefix.append(completeness)
        prefix_tokens = torch.cat(prefix, 1)
        n_prefix_tokens = prefix_tokens.shape[1]
        groups = [prefix_tokens, particles]
        masks = [
            torch.ones(
                bsz, n_prefix_tokens, dtype=torch.bool, device=jet.device,
            ),
            particle_valid,
        ]
        types = [0, 1]
        if vertices is not None:
            groups.append(vertices); masks.append(batch["vertex_mask"].bool()); types.append(2)
        if candidates is not None:
            groups.append(candidates); masks.append(batch["candidate_mask"].bool()); types.append(3)
        polarity_context = self._polarity_context(batch, bsz)
        groups = [
            value + self.token_type.weight[k] + (polarity_context if polarity_context is not None else 0)
            for value, k in zip(groups, types)
        ]
        tokens, token_valid = torch.cat(groups, 1), torch.cat(masks, 1)
        n_vertices = vertices.shape[1] if vertices is not None else 0
        n_candidates = candidates.shape[1] if candidates is not None else 0
        bias = self._attention_bias(
            batch, token_valid, n_particles, n_vertices, n_candidates,
            n_prefix_tokens,
        )
        for block in self.blocks: tokens = block(tokens, token_valid, bias)
        tokens = self.final_norm(tokens)

        jet_embedding = tokens[:, 0]
        prefix_cursor = 1
        semantic_embedding = None
        if self.semantic_token is not None:
            semantic_embedding = tokens[:, prefix_cursor]
            prefix_cursor += 1
        completeness_embedding = None
        if self.completeness_token is not None:
            completeness_embedding = tokens[:, prefix_cursor]
        particle_embedding = tokens[:, n_prefix_tokens:n_prefix_tokens+n_particles]
        cursor = n_prefix_tokens + n_particles
        vertex_embedding = tokens[:, cursor:cursor+vertices.shape[1]] if vertices is not None else None
        if vertices is not None: cursor += vertices.shape[1]
        candidate_embedding = tokens[:, cursor:cursor+candidates.shape[1]] if candidates is not None else None
        result: dict[str, torch.Tensor | list[torch.Tensor]] = {
            "jet_embedding": jet_embedding, "particle_embedding": particle_embedding,
        }
        if semantic_embedding is not None:
            result["semantic_embedding"] = semantic_embedding
        if completeness_embedding is not None:
            result["completeness_embedding"] = completeness_embedding
        if vertex_embedding is not None: result["vertex_embedding"] = vertex_embedding
        if candidate_embedding is not None: result["candidate_embedding"] = candidate_embedding
        scale = self.config.d_model ** -0.5
        if vertex_embedding is not None:
            result["vertex_track_logits"] = torch.einsum("bpd,bvd->bvp", particle_embedding, vertex_embedding) * scale
        if candidate_embedding is not None:
            result["candidate_track_logits"] = torch.einsum("bpd,bcd->bcp", particle_embedding, candidate_embedding) * scale
        if candidate_embedding is not None and vertex_embedding is not None:
            result["candidate_vertex_logits"] = torch.einsum("bvd,bcd->bcv", vertex_embedding, candidate_embedding) * scale
        natural_pid_embedding = (
            particle_embedding.detach()
            if self.config.pid_natural_head_detach_encoder else particle_embedding
        )
        if self.pid_class_head is not None:
            result["pid_logits"] = self.pid_class_head(natural_pid_embedding)
        if self.pid_balanced_class_head is not None:
            result["pid_balanced_logits"] = self.pid_balanced_class_head(particle_embedding)
        if self.pid_bin_heads: result["pid_bin_logits"] = [head(particle_embedding) for head in self.pid_bin_heads]
        if self.pid_continuous_head is not None:
            mean, log_scale = self.pid_continuous_head(particle_embedding).chunk(2, -1)
            result["pid_mean"], result["pid_log_scale"] = mean, log_scale.clamp(-7, 5)
        if self.origin_head is not None: result["origin_logits"] = self.origin_head(particle_embedding)
        if self.vertex_class_head is not None and vertex_embedding is not None: result["vertex_logits"] = self.vertex_class_head(vertex_embedding)
        if self.candidate_class_head is not None and candidate_embedding is not None: result["candidate_logits"] = self.candidate_class_head(candidate_embedding)
        if self.flavour_head is not None: result["flavour_logits"] = self.flavour_head(jet_embedding)
        if self.beauty_sign_head is not None:
            result["beauty_sign_logits"] = self.beauty_sign_head(jet_embedding)
        if self.charm_sign_head is not None:
            result["charm_sign_logits"] = self.charm_sign_head(jet_embedding)
        for name in (
            "b_vs_c", "b_vs_light", "c_vs_light", "b_vs_bbar", "c_vs_cbar",
        ):
            classifier = getattr(self, f"{name}_head")
            if classifier is not None:
                result[f"{name}_logits"] = classifier(jet_embedding)
        if self.topology_head is not None: result["topology_prediction"] = self.topology_head(jet_embedding)
        result["particle_feature_prediction"] = self.particle_feature_head(particle_embedding)
        if self.vertex_feature_head is not None and vertex_embedding is not None:
            result["vertex_feature_prediction"] = self.vertex_feature_head(vertex_embedding)
        if self.candidate_feature_head is not None and candidate_embedding is not None:
            result["candidate_feature_prediction"] = self.candidate_feature_head(candidate_embedding)

        completion_context = (
            completeness_embedding
            if completeness_embedding is not None else jet_embedding
        )
        if self.missing_count_head is not None:
            natural_count_context = (
                completion_context.detach()
                if self.config.missing_track_count_natural_head_detach_encoder
                else completion_context
            )
            result["missing_count_logits"] = self.missing_count_head(
                natural_count_context
            )
        if self.missing_count_balanced_head is not None:
            result["missing_count_balanced_logits"] = (
                self.missing_count_balanced_head(completion_context)
            )
        if self.corruption_type_head is not None:
            result["corruption_type_logits"] = self.corruption_type_head(completion_context)
        if self.removed_summary_head is not None:
            summary_mean, summary_log_scale = self.removed_summary_head(
                completion_context
            ).chunk(2, -1)
            result["removed_summary_mean"] = summary_mean
            result["removed_summary_log_scale"] = summary_log_scale.clamp(-7, 5)

        if self.missing_query_decoder is not None:
            assert self.missing_query_tokens is not None
            query_seed = self.missing_query_tokens.expand(bsz, -1, -1)
            query_seed = query_seed + completion_context[:, None]
            missing_queries = self.missing_query_decoder(
                query_seed, tokens, token_valid,
            )
            result["missing_query_embedding"] = missing_queries
            if self.missing_query_existence_head is not None:
                existence = self.missing_query_existence_head(
                    missing_queries
                ).squeeze(-1)
                result["missing_query_existence_logits"] = existence
                # Temporary compatibility alias for early trainer prototypes.
                result["missing_existence_logits"] = existence
            if self.missing_query_feature_head is not None:
                feature_mean, feature_log_scale = self.missing_query_feature_head(
                    missing_queries
                ).chunk(2, -1)
                result["missing_query_features_mean"] = feature_mean
                result["missing_query_features_log_scale"] = (
                    feature_log_scale.clamp(-7, 5)
                )
            if self.missing_pid_head is not None:
                natural_missing_queries = (
                    missing_queries.detach()
                    if self.config.missing_track_pid_natural_head_detach_encoder
                    else missing_queries
                )
                result["missing_pid_logits"] = self.missing_pid_head(
                    natural_missing_queries
                )
            if self.missing_pid_balanced_head is not None:
                result["missing_pid_balanced_logits"] = (
                    self.missing_pid_balanced_head(missing_queries)
                )
            if self.missing_charge_head is not None:
                result["missing_charge_logits"] = self.missing_charge_head(
                    missing_queries
                )
            if self.missing_topology_head is not None:
                result["missing_topology_prediction"] = (
                    self.missing_topology_head(missing_queries)
                )
            if vertex_embedding is not None:
                result["missing_vertex_logits"] = torch.einsum(
                    "bqd,bvd->bqv", missing_queries, vertex_embedding,
                ) * scale
            if candidate_embedding is not None:
                result["missing_candidate_logits"] = torch.einsum(
                    "bqd,bcd->bqc", missing_queries, candidate_embedding,
                ) * scale
        return result




def build_model(config: ModelConfig) -> nn.Module:
    if config.architecture == "heterogeneous_particle_transformer":
        return JetTopologyTransformer(config)
    raise ValueError(f"Unknown model architecture: {config.architecture}")




