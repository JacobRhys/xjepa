"""The seven training objectives of the controlled study.

Every condition sees an identical token budget and identical data; the objective
is the only thing that differs (RESEARCH_PLAN.md sec. 3.3). The conditions are:

===============  ====  =========  ================  ==================================
name             mask  predictor  loss positions    isolates
===============  ====  =========  ================  ==================================
c1_mlm            yes   --        masked            MLM baseline (ESM-2 80/10/10)
c2_jepa_ema       yes   yes       masked            EMA target; EXPECTED TO COLLAPSE
c3_jepa_frozen    yes   yes       masked            the treatment (frozen ESM-IF1)
c4_mlm_jepa       yes   yes       masked            MLM + lambda * JEPA
c5_distil         no    no        all               joint baseline
c5b_masked_distil yes   no        masked            value of the predictor
c5c_predictor_nomask no yes       all               value of masking
===============  ====  =========  ================  ==================================

Contract rules obeyed here (``docs/CONTRACTS.md``):

* ``loss()`` returns ``(scalar_loss, metrics)`` where every metric value stays a
  **GPU tensor**. Nothing in this module calls ``.item()``, ``.cpu()``,
  ``float()`` or ``print()`` on a tensor.
* No boolean indexing on masks. Selecting masked positions with ``x[mask]``
  produces a *data-dependent* output shape, which would defeat
  ``torch.compile(dynamic=False)`` and trigger a recompile per unique mask count.
  Instead every reduction is a fixed-shape weighted mean.

Note on ``c2_jepa_ema``: no anti-collapse trick (no centring, no whitening, no
variance term) is applied. Collapse is hypothesis H1a and is the replication
result we want to observe, not a bug to be patched.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import torch
import torch.nn.functional as F
from torch import Tensor, nn

__all__ = [
    "ObjectiveConfig",
    "Objective",
    "TrainableModel",
    "OBJECTIVE_NAMES",
    "build_objective",
    "MlmObjective",
    "JepaEmaObjective",
    "JepaFrozenObjective",
    "MlmJepaObjective",
    "DistilObjective",
    "MaskedDistilObjective",
    "PredictorNomaskObjective",
]

OBJECTIVE_NAMES: tuple[str, ...] = (
    "c1_mlm",
    "c2_jepa_ema",
    "c3_jepa_frozen",
    "c4_mlm_jepa",
    "c5_distil",
    "c5b_masked_distil",
    "c5c_predictor_nomask",
)

IGNORE_INDEX = -100


@dataclass(frozen=True)
class ObjectiveConfig:
    """Objective hyper-parameters (the only block that differs between configs).

    Attributes:
        name: One of :data:`OBJECTIVE_NAMES`.
        lambda_jepa: Weight of the latent term in ``c4_mlm_jepa`` (sec. 1.5 sweep).
        cos_weight: Weight of the ``1 - cos`` term in the latent loss.
        l1_weight: Weight of the smooth-L1 term in the latent loss.
        smooth_l1_beta: Transition point of the smooth-L1 (Huber) term.
        ema_tau_base: Initial EMA momentum for ``c2_jepa_ema``.
        ema_tau_final: Final EMA momentum for ``c2_jepa_ema``.
    """

    name: str = "c1_mlm"
    lambda_jepa: float = 1.0
    cos_weight: float = 1.0
    l1_weight: float = 1.0
    smooth_l1_beta: float = 1.0
    ema_tau_base: float = 0.996
    ema_tau_final: float = 1.0

    def __post_init__(self) -> None:
        if self.name not in OBJECTIVE_NAMES:
            raise ValueError(f"unknown objective {self.name!r}; expected one of {OBJECTIVE_NAMES}")
        if self.cos_weight < 0 or self.l1_weight < 0 or (self.cos_weight + self.l1_weight) <= 0:
            raise ValueError("cos_weight/l1_weight must be non-negative and not both zero")


@runtime_checkable
class TrainableModel(Protocol):
    """The composite module an objective is handed.

    ``xjepa.train.trainer.XJepaModel`` is the concrete implementation; the tests
    supply a tiny fake with the same attributes. Sub-modules are ``None`` when the
    condition does not use them, so that unused parameters never join the
    optimiser and can never receive a gradient.

    Attributes:
        encoder: ``forward(tokens, pad_mask) -> [B, L, d_model]``.
        predictor: ``forward(h, mask_sel) -> [B, L, target_dim]`` or ``None``.
        mlm_head: ``forward(h) -> [B, L, vocab]`` or ``None``.
        latent_head: ``nn.Linear(d_model, target_dim)`` or ``None``; the
            "no predictor" path of C5 / C5b.
        target_encoder: EMA copy of ``encoder`` (C2 only) or ``None``.
        target_head: EMA copy of ``latent_head`` mapping the EMA encoder output to
            the target dimension (C2 only) or ``None``.
    """

    encoder: nn.Module
    predictor: nn.Module | None
    mlm_head: nn.Module | None
    latent_head: nn.Module | None
    target_encoder: nn.Module | None
    target_head: nn.Module | None


# --------------------------------------------------------------------------- #
# shared tensor helpers -- all fixed-shape, all sync-free
# --------------------------------------------------------------------------- #


def original_tokens(batch) -> Tensor:
    """Reconstruct the *uncorrupted* token ids from a :class:`Batch`.

    The contract's ``Batch`` carries corrupted ``tokens`` (ESM-2 80/10/10) and
    ``labels`` holding the original id at every masked position and
    ``-100`` elsewhere. The unmasked-view conditions (C5, C5c) need the clean
    sequence, so it is recovered as ``where(labels >= 0, labels, tokens)``.

    This is exact for 80/10/10: the 10% "keep" and 10% "random" positions are
    still labelled, so their original id is restored too.

    Args:
        batch: A ``xjepa.data.store.Batch``.

    Returns:
        int64 ``[B, L]`` clean token ids on the batch's device.
    """
    labels = batch.labels
    return torch.where(labels >= 0, labels, batch.tokens)


def _mask_weights(batch, masked_only: bool) -> Tensor:
    """Per-position loss weights: real residues, optionally restricted to masked ones.

    Args:
        batch: A ``Batch``.
        masked_only: If True, weight only positions with ``mask_sel``.

    Returns:
        float ``[B, L]`` weights (1.0 = counted).
    """
    pad = batch.pad_mask
    sel = (pad & batch.mask_sel) if masked_only else pad
    return sel.to(torch.float32)


def _weighted_mean(per_position: Tensor, weights: Tensor) -> Tensor:
    """Mean of ``per_position`` over the positions selected by ``weights``.

    ``clamp_min(1.0)`` keeps the result finite for an (unlikely) all-pad batch
    without a data-dependent branch, which would force a graph break.

    Args:
        per_position: float ``[B, L]``.
        weights: float ``[B, L]``.

    Returns:
        Scalar tensor.
    """
    return (per_position * weights).sum() / weights.sum().clamp_min(1.0)


def latent_terms(
    pred: Tensor,
    target: Tensor,
    weights: Tensor,
    cfg: ObjectiveConfig,
) -> tuple[Tensor, Tensor, Tensor]:
    """Bounded latent-regression loss on L2-normalised vectors.

    RESEARCH_PLAN.md sec. 1.5: MLM cross-entropy and a raw L2-on-embeddings term
    have incomparable scales, which makes ``lambda`` in C4 impossible to set
    without a sweep we cannot afford. Normalising both vectors and combining
    ``1 - cos`` with a smooth-L1 (I-JEPA / DINO practice) makes the term bounded
    in ``[0, 2]``:

    * ``1 - cos(a, b)`` lies in ``[0, 2]`` for unit vectors;
    * ``mean_d smooth_l1(a_d, b_d)`` lies in ``[0, 2/sqrt(D)] subset [0, 2]``;
    * the returned loss is the **convex combination** of the two, hence in ``[0, 2]``.

    Args:
        pred: float ``[B, L, D]`` predicted latents (fp32 recommended).
        target: float ``[B, L, D]`` target latents; gradients must already be cut.
        weights: float ``[B, L]`` position weights.
        cfg: Objective config supplying the term weights.

    Returns:
        ``(loss, cosine_similarity, target_dim_std)`` -- all scalar GPU tensors.
    """
    p = F.normalize(pred.float(), dim=-1, eps=1e-6)
    t = F.normalize(target.float(), dim=-1, eps=1e-6)

    cos = (p * t).sum(-1)  # [B, L] in [-1, 1]
    sl1 = F.smooth_l1_loss(p, t, beta=cfg.smooth_l1_beta, reduction="none").mean(-1)  # [B, L]

    total_w = cfg.cos_weight + cfg.l1_weight
    per_position = (cfg.cos_weight * (1.0 - cos) + cfg.l1_weight * sl1) / total_w
    loss = _weighted_mean(per_position, weights)

    with torch.no_grad():
        mean_cos = _weighted_mean(cos, weights)
        # Per-dimension std of the (weighted) targets: the VICReg variance
        # criterion, a cheap early-warning signal for collapse of the target side.
        w = weights.unsqueeze(-1)
        denom = w.sum().clamp_min(1.0)
        mean_t = (t * w).sum(dim=(0, 1)) / denom
        var_t = ((t - mean_t) ** 2 * w).sum(dim=(0, 1)) / denom
        tgt_std = var_t.clamp_min(0.0).sqrt().mean()
    return loss, mean_cos, tgt_std


def mlm_terms(logits: Tensor, labels: Tensor) -> tuple[Tensor, Tensor]:
    """Masked-LM cross-entropy plus masked accuracy.

    Args:
        logits: float ``[B, L, vocab]``.
        labels: int64 ``[B, L]`` with ``-100`` at unsupervised positions.

    Returns:
        ``(cross_entropy, accuracy)`` -- scalar GPU tensors.
    """
    flat_logits = logits.reshape(-1, logits.shape[-1]).float()
    flat_labels = labels.reshape(-1)
    ce = F.cross_entropy(flat_logits, flat_labels, ignore_index=IGNORE_INDEX)
    with torch.no_grad():
        supervised = (flat_labels != IGNORE_INDEX).to(torch.float32)
        correct = (flat_logits.argmax(-1) == flat_labels).to(torch.float32) * supervised
        acc = correct.sum() / supervised.sum().clamp_min(1.0)
    return ce, acc


def _require(module: nn.Module | None, what: str, name: str) -> nn.Module:
    """Raise a clear error when a condition's required sub-module is missing."""
    if module is None:
        raise ValueError(f"objective {name!r} requires model.{what}, which is None")
    return module


# --------------------------------------------------------------------------- #
# objectives
# --------------------------------------------------------------------------- #


class Objective:
    """Base class implementing the ``Objective`` protocol from the contract.

    Subclasses implement :meth:`loss`, which must return a scalar loss and a dict
    of metrics whose values are GPU tensors (never Python numbers). The trainer
    accumulates those tensors into a preallocated buffer and syncs on its own
    cadence.
    """

    name: str = "base"
    metric_names: tuple[str, ...] = ()
    #: Whether the condition needs a predictor / MLM head / linear latent head /
    #: EMA target encoder. The trainer uses these to build only what is used, so
    #: an unused sub-module can never silently receive gradients.
    needs_predictor: bool = False
    needs_mlm_head: bool = False
    needs_latent_head: bool = False
    # c5c predicts at *every* position, so its predictor must not substitute a
    # learned mask query at masked ones. Every other predictor condition does.
    predictor_replaces_masked: bool = True
    needs_ema: bool = False
    #: Whether the condition ever feeds corrupted tokens to the encoder.
    uses_masking: bool = True

    def __init__(self, cfg: ObjectiveConfig) -> None:
        self.cfg = cfg

    def loss(self, model: TrainableModel, batch) -> tuple[Tensor, dict[str, Tensor]]:
        """Compute the training loss for one batch.

        Args:
            model: The composite model (see :class:`TrainableModel`).
            batch: A ``xjepa.data.store.Batch`` resident on device.

        Returns:
            ``(scalar_loss, metrics)``. Metric values stay GPU tensors.
        """
        raise NotImplementedError

    def __repr__(self) -> str:  # pragma: no cover - debugging convenience
        return f"{type(self).__name__}(name={self.name!r})"


class MlmObjective(Objective):
    """C1 -- masked amino-acid cross-entropy on ESM-2 80/10/10 corruption."""

    name = "c1_mlm"
    metric_names = ("mlm_loss", "mlm_acc")
    needs_mlm_head = True

    def loss(self, model: TrainableModel, batch) -> tuple[Tensor, dict[str, Tensor]]:
        head = _require(model.mlm_head, "mlm_head", self.name)
        h = model.encoder(batch.tokens, batch.pad_mask)
        ce, acc = mlm_terms(head(h), batch.labels)
        return ce, {"mlm_loss": ce.detach(), "mlm_acc": acc}


class JepaEmaObjective(Objective):
    """C2 -- predict EMA-target-encoder latents at masked positions.

    This condition is *expected* to collapse (hypothesis H1a): with a learnable
    target and no anti-collapse term, the trivial constant solution is a global
    optimum. Nothing here counteracts that on purpose -- observing the collapse
    is the replication result.
    """

    name = "c2_jepa_ema"
    metric_names = ("jepa_loss", "cos_sim", "target_std")
    needs_predictor = True
    needs_latent_head = True
    needs_ema = True

    def loss(self, model: TrainableModel, batch) -> tuple[Tensor, dict[str, Tensor]]:
        predictor = _require(model.predictor, "predictor", self.name)
        t_enc = _require(model.target_encoder, "target_encoder", self.name)
        t_head = _require(model.target_head, "target_head", self.name)

        h = model.encoder(batch.tokens, batch.pad_mask)
        pred = predictor(h, batch.mask_sel, pad_mask=batch.pad_mask)

        with torch.no_grad():
            clean = original_tokens(batch)
            target = t_head(t_enc(clean, batch.pad_mask)).detach()

        weights = _mask_weights(batch, masked_only=True)
        loss, cos, tgt_std = latent_terms(pred, target, weights, self.cfg)
        return loss, {"jepa_loss": loss.detach(), "cos_sim": cos, "target_std": tgt_std}


class JepaFrozenObjective(Objective):
    """C3 -- predict frozen ESM-IF1 latents at masked positions (the treatment).

    Targets come from the corpus' precomputed, PCA-128 target bank
    (``batch.targets``); the target side is frozen by construction, so rank is
    capped by the target bank's own rank (hypothesis H1b).
    """

    name = "c3_jepa_frozen"
    metric_names = ("jepa_loss", "cos_sim", "target_std")
    needs_predictor = True

    def loss(self, model: TrainableModel, batch) -> tuple[Tensor, dict[str, Tensor]]:
        predictor = _require(model.predictor, "predictor", self.name)
        h = model.encoder(batch.tokens, batch.pad_mask)
        pred = predictor(h, batch.mask_sel, pad_mask=batch.pad_mask)
        weights = _mask_weights(batch, masked_only=True)
        loss, cos, tgt_std = latent_terms(pred, batch.targets, weights, self.cfg)
        return loss, {"jepa_loss": loss.detach(), "cos_sim": cos, "target_std": tgt_std}


class MlmJepaObjective(Objective):
    """C4 -- ``mlm + lambda * jepa`` sharing a single encoder forward pass."""

    name = "c4_mlm_jepa"
    metric_names = ("mlm_loss", "mlm_acc", "jepa_loss", "cos_sim", "target_std")
    needs_predictor = True
    needs_mlm_head = True

    def loss(self, model: TrainableModel, batch) -> tuple[Tensor, dict[str, Tensor]]:
        head = _require(model.mlm_head, "mlm_head", self.name)
        predictor = _require(model.predictor, "predictor", self.name)

        h = model.encoder(batch.tokens, batch.pad_mask)
        ce, acc = mlm_terms(head(h), batch.labels)
        pred = predictor(h, batch.mask_sel, pad_mask=batch.pad_mask)
        weights = _mask_weights(batch, masked_only=True)
        jepa, cos, tgt_std = latent_terms(pred, batch.targets, weights, self.cfg)

        total = ce + self.cfg.lambda_jepa * jepa
        return total, {
            "mlm_loss": ce.detach(),
            "mlm_acc": acc,
            "jepa_loss": jepa.detach(),
            "cos_sim": cos,
            "target_std": tgt_std,
        }


class DistilObjective(Objective):
    """C5 -- regress all positions from the *unmasked* sequence, linear head only.

    No masking, no predictor: the joint baseline of the C3/C5b/C5c ladder
    (RESEARCH_PLAN.md sec. 1.4).
    """

    name = "c5_distil"
    metric_names = ("distil_loss", "cos_sim", "target_std")
    needs_latent_head = True
    uses_masking = False

    def loss(self, model: TrainableModel, batch) -> tuple[Tensor, dict[str, Tensor]]:
        head = _require(model.latent_head, "latent_head", self.name)
        tokens = original_tokens(batch)
        h = model.encoder(tokens, batch.pad_mask)
        pred = head(h)
        weights = _mask_weights(batch, masked_only=False)
        loss, cos, tgt_std = latent_terms(pred, batch.targets, weights, self.cfg)
        return loss, {"distil_loss": loss.detach(), "cos_sim": cos, "target_std": tgt_std}


class MaskedDistilObjective(Objective):
    """C5b -- masked positions, linear head, **no predictor**.

    C3 minus the predictor: isolates the predictor's contribution.
    """

    name = "c5b_masked_distil"
    metric_names = ("distil_loss", "cos_sim", "target_std")
    needs_latent_head = True

    def loss(self, model: TrainableModel, batch) -> tuple[Tensor, dict[str, Tensor]]:
        head = _require(model.latent_head, "latent_head", self.name)
        h = model.encoder(batch.tokens, batch.pad_mask)
        pred = head(h)
        weights = _mask_weights(batch, masked_only=True)
        loss, cos, tgt_std = latent_terms(pred, batch.targets, weights, self.cfg)
        return loss, {"distil_loss": loss.detach(), "cos_sim": cos, "target_std": tgt_std}


class PredictorNomaskObjective(Objective):
    """C5c -- all positions, **with** predictor, no masking.

    C3 minus the masking: isolates masking's contribution. The predictor is
    built with ``replace_masked=False`` so it transforms the encoder output at
    every position rather than substituting a learned query, which keeps the
    shape fixed exactly as in C3 while leaving the encoder in the gradient path.
    """

    name = "c5c_predictor_nomask"
    metric_names = ("distil_loss", "cos_sim", "target_std")
    needs_predictor = True
    uses_masking = False
    # Predicting at every position means the predictor must NOT substitute its
    # learned mask query anywhere -- doing so discards the encoder output and
    # severs the gradient path to the encoder entirely.
    predictor_replaces_masked = False

    def loss(self, model: TrainableModel, batch) -> tuple[Tensor, dict[str, Tensor]]:
        predictor = _require(model.predictor, "predictor", self.name)
        tokens = original_tokens(batch)
        h = model.encoder(tokens, batch.pad_mask)
        # replace_masked=False on this condition's predictor, so mask_sel is
        # inert; pass the (all-false) real one rather than the pad mask.
        pred = predictor(h, batch.mask_sel, pad_mask=batch.pad_mask)
        weights = _mask_weights(batch, masked_only=False)
        loss, cos, tgt_std = latent_terms(pred, batch.targets, weights, self.cfg)
        return loss, {"distil_loss": loss.detach(), "cos_sim": cos, "target_std": tgt_std}


_REGISTRY: dict[str, type[Objective]] = {
    MlmObjective.name: MlmObjective,
    JepaEmaObjective.name: JepaEmaObjective,
    JepaFrozenObjective.name: JepaFrozenObjective,
    MlmJepaObjective.name: MlmJepaObjective,
    DistilObjective.name: DistilObjective,
    MaskedDistilObjective.name: MaskedDistilObjective,
    PredictorNomaskObjective.name: PredictorNomaskObjective,
}


def build_objective(cfg: ObjectiveConfig) -> Objective:
    """Instantiate the objective named by ``cfg.name``.

    Args:
        cfg: Objective configuration.

    Returns:
        The matching :class:`Objective` instance.

    Raises:
        ValueError: If ``cfg.name`` is not one of :data:`OBJECTIVE_NAMES`.
    """
    try:
        cls = _REGISTRY[cfg.name]
    except KeyError as exc:  # pragma: no cover - guarded by ObjectiveConfig
        raise ValueError(f"unknown objective {cfg.name!r}") from exc
    return cls(cfg)
