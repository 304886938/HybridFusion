import math
from typing import Optional, Tuple

import numpy as np

try:
    import torch
    _HAS_TORCH = True
except Exception:
    torch = None
    _HAS_TORCH = False

from .temporal_memory_aggregation import TemporalMemoryAggregation  



def _to_numpy(x):
    """Accept np.ndarray or torch.Tensor; return np.ndarray (float32)."""
    if isinstance(x, np.ndarray):
        return x.astype(np.float32, copy=False)
    if _HAS_TORCH and isinstance(x, torch.Tensor):
        return x.detach().float().cpu().numpy()
    raise TypeError(f"Unsupported type: {type(x)}. Expected np.ndarray or torch.Tensor.")


def _l2_normalize(x: np.ndarray, axis: int = -1, eps: float = 1e-6) -> np.ndarray:
    n = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / (n + eps)


def transformer_qk_similarity_matrix(
    det_app,  # [D, C]
    tra_app,  # [T, C]
    temperature: float = 1.0,
    use_softmax: bool = True,
) -> np.ndarray:
    """
    Compute Transformer-style similarity:
        logits = (Q @ K^T) / sqrt(C) / temperature

    Here we treat:
        Q = det_app, K = tra_app
    so the output is [D, T].

    If use_softmax=True:
        sim = softmax(logits, axis=1)  # each detection attends over trajectories
    else:
        sim = logits  # unnormalized similarity

    Returns:
        sim: np.ndarray, shape [D, T]
    """
    Q = _to_numpy(det_app)
    K = _to_numpy(tra_app)

    if Q.ndim != 2 or K.ndim != 2 or Q.shape[1] != K.shape[1]:
        raise ValueError(f"Shape mismatch: det_app {Q.shape}, tra_app {K.shape}. Both must be [N, C].")

    C = Q.shape[1]
    logits = (Q @ K.T) / (math.sqrt(C) * max(temperature, 1e-6))

    if not use_softmax:
        return logits.astype(np.float32, copy=False)

    # numerically stable softmax over trajectories
    logits = logits - logits.max(axis=1, keepdims=True)
    expv = np.exp(logits)
    sim = expv / (expv.sum(axis=1, keepdims=True) + 1e-9)
    return sim.astype(np.float32, copy=False)


def cosine_similarity_matrix(
    det_app,
    tra_app,
) -> np.ndarray:
    """
    Cosine similarity in [-1, 1].
    Returns shape [D, T].
    """
    Q = _to_numpy(det_app)
    K = _to_numpy(tra_app)
    Q = _l2_normalize(Q, axis=1)
    K = _l2_normalize(K, axis=1)
    return (Q @ K.T).astype(np.float32, copy=False)


def compute_second_level_cost_from_appearance(
    det_app,
    tra_app,
    valid_mask_2d: Optional[np.ndarray] = None,  # [D, T] True means valid match
    mode: str = "qk_softmax",  # "qk_softmax" | "qk_logits" | "cosine"
    temperature: float = 1.0,
) -> np.ndarray:
    if mode == "qk_softmax":
        sim = transformer_qk_similarity_matrix(det_app, tra_app, temperature=temperature, use_softmax=True)  # [D,T], (0,1)
        cost = 1.0 - sim
    elif mode == "qk_logits":
        logits = transformer_qk_similarity_matrix(det_app, tra_app, temperature=temperature, use_softmax=False)
        # map logits -> (0,1) similarity using sigmoid for a bounded cost
        sim = 1.0 / (1.0 + np.exp(-logits))
        cost = 1.0 - sim
    elif mode == "cosine":
        cos = cosine_similarity_matrix(det_app, tra_app)  # [-1,1]
        sim = (cos + 1.0) * 0.5  # [0,1]
        cost = 1.0 - sim
    else:
        raise ValueError(f"Unknown mode: {mode}")

    cost = cost.astype(np.float32, copy=False)

    if valid_mask_2d is not None:
        vm = np.asarray(valid_mask_2d).astype(bool)
        if vm.shape != cost.shape:
            raise ValueError(f"valid_mask_2d shape {vm.shape} != cost shape {cost.shape}")
        cost[~vm] = np.inf

    return cost


def same_class_valid_mask_2d(det_labels: np.ndarray, tra_labels: np.ndarray) -> np.ndarray:
    """
    Build a [D, T] mask for same-class matching.

    det_labels: [D]
    tra_labels: [T]
    """
    det_labels = np.asarray(det_labels).astype(int)
    tra_labels = np.asarray(tra_labels).astype(int)
    return (det_labels[:, None] == tra_labels[None, :])


def _split_query_embed(query_embed, hidden_dim: int, use_dab: bool = False):
    """Extract appearance embedding from a query embedding.

    Many DETR-like trackers store `query_embed` as:
      - non-DAB: [pos_embed (H), app_embed (H)] => total 2H
      - DAB:     a single embedding is used; treat it as appearance.

    This helper returns the *appearance* part as shape [N, H].
    It supports both torch.Tensor and np.ndarray.
    """
    if query_embed is None:
        raise ValueError("query_embed is None")

    if _HAS_TORCH and isinstance(query_embed, torch.Tensor):
        if use_dab:
            return query_embed
        if query_embed.shape[-1] == hidden_dim:
            return query_embed
        return query_embed[..., hidden_dim:]

    qe = np.asarray(query_embed)
    if use_dab:
        return qe
    if qe.shape[-1] == hidden_dim:
        return qe
    return qe[..., hidden_dim:]


class AppearanceAssociator:
    """A lightweight wrapper that plugs `TemporalMemoryAggregation` into association.

    Typical flow in a runtime tracker:
      1) After first-stage association, update track memories with
         `associator.update_track_memory(tracks)`.
      2) For second-stage association, build cost matrix using query embeddings:
         `cost2 = associator.build_cost(det_query_embed, tra_query_embed, ...)`.
    """

    def __init__(
        self,
        hidden_dim: int,
        temporal_aggregator: Optional[object] = None,
        use_dab: bool = False,
        sim_mode: str = "qk_softmax",
        temperature: float = 1.0,
        same_class_only: bool = True,
    ):
        self.hidden_dim = int(hidden_dim)
        self.use_dab = bool(use_dab)
        self.sim_mode = sim_mode
        self.temperature = float(temperature)
        self.same_class_only = bool(same_class_only)
        self.temporal_aggregator = temporal_aggregator

    def update_track_memory(self, tracks_list):
        """Update track query embeddings using temporal aggregation.

        The aggregator is expected to mutate the tracks in-place and return them.
        """
        if self.temporal_aggregator is None:
            return tracks_list

        if hasattr(self.temporal_aggregator, "update_tracks_embedding"):
            return self.temporal_aggregator.update_tracks_embedding(tracks_list)

        if callable(self.temporal_aggregator):
            return self.temporal_aggregator(tracks_list)

        return tracks_list

    def build_cost(
        self,
        det_query_embed,
        tra_query_embed,
        det_labels: Optional[np.ndarray] = None,
        tra_labels: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Build appearance-based cost matrix [D, T]."""

        det_app = _split_query_embed(det_query_embed, self.hidden_dim, use_dab=self.use_dab)
        tra_app = _split_query_embed(tra_query_embed, self.hidden_dim, use_dab=self.use_dab)

        valid_mask = None
        if self.same_class_only and det_labels is not None and tra_labels is not None:
            valid_mask = same_class_valid_mask_2d(det_labels, tra_labels)

        return compute_second_level_cost_from_appearance(
            det_app=det_app,
            tra_app=tra_app,
            valid_mask_2d=valid_mask,
            mode=self.sim_mode,
            temperature=self.temperature,
        )

def compute_cost_for_appearance_feature(tracker, appearance_cfg: dict):
    """
    Example of how to integrate into an existing `Tracker` instance at runtime:
        patch_tracker_compute_cost_for_appearance(tracker, {...})

    This replaces tracker.compute_cost with a wrapped version that:
      - keeps one_stage cost unchanged
      - replaces two_stage cost with appearance cost

    Requirements on tracker:
      - tracker.det_infos must contain 'det_app' and 'np_dets' (for labels)
      - tracker.tra_infos must contain 'tra_app' and 'np_tras' (for labels)

    NOTE: This is optional. You can also directly edit nusc_tracker.py's compute_cost().
    """
    orig_compute_cost = tracker.compute_cost

    def _compute_cost_wrapped():
        cost_mats = orig_compute_cost()
        # build appearance cost2
        det_app = tracker.det_infos.get("det_app", None)
        tra_app = tracker.tra_infos.get("tra_app", None)
        if det_app is None or tra_app is None:
            raise KeyError("Missing det_app or tra_app. Please add them before calling compute_cost().")

        det_labels = tracker.det_infos["np_dets"][:, -1]
        tra_labels = tracker.tra_infos["np_tras"][:, -4]
        vm2d = same_class_valid_mask_2d(det_labels, tra_labels)

        cost2 = compute_second_level_cost_from_appearance(
            det_app=det_app,
            tra_app=tra_app,
            valid_mask_2d=vm2d,
            mode=appearance_cfg.get("mode", "qk_softmax"),
            temperature=float(appearance_cfg.get("temperature", 1.0)),
        )
        cost_mats["two_stage"] = cost2
        return cost_mats

    tracker.compute_cost = _compute_cost_wrapped
    return tracker
