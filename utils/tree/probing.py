"""Distance probe training and evaluation helpers."""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    import geoopt
except ImportError:  # pragma: no cover
    geoopt = None

from .encoding import DEVICE
from .trees import DistanceProbeConfig

# =============================================================================
# Public API
# =============================================================================

__all__ = [
    "DistanceProbe",
    "pairwise_euclidean",
    "pairwise_poincare",
    "pairwise_distance",
    "transform_probe_space",
    "fit_distance_probe",
    "fit_depth_probe",
    "evaluate_probes",
    "safe_float",
]


# =============================================================================
# Projection modules
# =============================================================================

class DistanceProbe(torch.nn.Module):
    """Linear projection trained to match tree distances."""

    def __init__(self, in_dim: int, proj_dim: int, device: torch.device) -> None:
        super().__init__()
        init_scale = 1.0 / math.sqrt(max(in_dim, 1))
        self.B = torch.nn.Parameter(torch.randn(in_dim, proj_dim, device=device) * init_scale)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs @ self.B


def pairwise_euclidean(points: torch.Tensor) -> torch.Tensor:
    """Compute pairwise Euclidean distances with numeric stability."""

    squared = (points ** 2).sum(dim=1, keepdim=True)
    distances_sq = squared + squared.T - 2 * (points @ points.T)
    distances_sq = torch.clamp(distances_sq, min=0.0)
    return torch.sqrt(distances_sq + 1e-9)


# =============================================================================
# Hyperbolic utilities
# =============================================================================

POINCARE_EPS = 1e-5


def _project_to_poincare_ball_torch(points: torch.Tensor, eps: float = POINCARE_EPS) -> torch.Tensor:
    """Project points to the open Poincaré ball to guarantee finite distances."""

    norms = torch.linalg.norm(points, dim=-1, keepdim=True)
    max_norm = 1.0 - eps
    scale = torch.ones_like(norms)
    mask = norms >= max_norm
    safe_norms = torch.where(mask, norms, torch.ones_like(norms))
    scale[mask] = max_norm / safe_norms[mask]
    return points * scale


def _project_to_ball_torch(points: torch.Tensor, curvature: torch.Tensor, eps: float = POINCARE_EPS) -> torch.Tensor:
    """Project points into the open Poincaré ball with curvature ``curvature``."""

    if points.dim() == 1:
        points = points.unsqueeze(0)
        squeeze_back = True
    else:
        squeeze_back = False
    curvature = torch.as_tensor(curvature, dtype=points.dtype, device=points.device)
    radius = 1.0 / torch.sqrt(torch.clamp(curvature, min=eps))
    max_norm = radius - eps
    norms = torch.linalg.norm(points, dim=-1, keepdim=True)
    scale = torch.ones_like(norms)
    mask = norms >= max_norm
    safe_norms = torch.where(mask, norms, torch.ones_like(norms))
    scale[mask] = max_norm / torch.clamp(safe_norms[mask], min=eps)
    projected = points * scale
    if squeeze_back:
        projected = projected.squeeze(0)
    return projected


def _mobius_add_torch(x: torch.Tensor, y: torch.Tensor, curvature: torch.Tensor, eps: float = POINCARE_EPS) -> torch.Tensor:
    """Möbius addition on the Poincaré ball."""

    c = torch.as_tensor(curvature, dtype=x.dtype, device=x.device)
    # Broadcast x to the shape of y if needed.
    while x.dim() < y.dim():
        x = x.unsqueeze(0)
    if x.shape != y.shape:
        x = x.expand_as(y)
    xy = torch.sum(x * y, dim=-1, keepdim=True)
    x2 = torch.sum(x * x, dim=-1, keepdim=True)
    y2 = torch.sum(y * y, dim=-1, keepdim=True)
    numerator = (1 + 2 * c * xy + c * y2) * x + (1 - c * x2) * y
    denominator = 1 + 2 * c * xy + (c ** 2) * x2 * y2
    return numerator / torch.clamp(denominator, min=eps)


def _transform_probe_space_torch(
    points: torch.Tensor,
    *,
    center: Optional[torch.Tensor],
    curvature: torch.Tensor,
    eps: float = POINCARE_EPS,
) -> torch.Tensor:
    projected = _project_to_ball_torch(points, curvature, eps=eps)
    if center is not None:
        center_projected = _project_to_ball_torch(center, curvature, eps=eps)
        center_expanded = center_projected.unsqueeze(0).expand_as(projected)
        projected = _mobius_add_torch(center_expanded, projected, curvature, eps=eps)
    return projected


def transform_probe_space(points: np.ndarray, info: Dict[str, Any], *, eps: float = POINCARE_EPS) -> np.ndarray:
    """Apply geometry-specific post-processing to probe embeddings (returns numpy)."""

    geometry = (info.get("geometry") or "euclidean").lower()
    if geometry != "hyperbolic":
        return points
    curvature = float(info.get("curvature", 1.0))
    center = info.get("center")
    center_tensor = torch.as_tensor(center, dtype=torch.float32) if center is not None else None
    transformed = _transform_probe_space_torch(
        torch.as_tensor(points, dtype=torch.float32),
        center=center_tensor,
        curvature=torch.tensor(curvature, dtype=torch.float32),
        eps=eps,
    )
    return transformed.cpu().numpy()


def pairwise_poincare(
    points: torch.Tensor,
    *,
    center: Optional[torch.Tensor] = None,
    curvature: Optional[torch.Tensor] = None,
    eps: float = POINCARE_EPS,
) -> torch.Tensor:
    """Compute pairwise Poincaré distances with optional center/curvature."""

    if curvature is None and center is None:
        projected = _project_to_poincare_ball_torch(points, eps=eps)
        squared_norms = (projected ** 2).sum(dim=1, keepdim=True)
        diff_sq = torch.clamp(
            squared_norms + squared_norms.T - 2 * (projected @ projected.T),
            min=0.0,
        )
        denom = torch.clamp((1.0 - squared_norms) * (1.0 - squared_norms.T), min=eps)
        arg = 1.0 + 2.0 * diff_sq / denom
        arg = torch.clamp(arg, min=1.0 + eps)
        diag_mask = torch.eye(arg.shape[0], dtype=torch.bool, device=arg.device)
        arg = torch.where(diag_mask, torch.ones_like(arg), arg)
        distances = torch.acosh(arg)
        distances = distances.clone()
        distances.fill_diagonal_(0.0)
        return distances

    curvature_tensor = torch.as_tensor(
        1.0 if curvature is None else curvature,
        dtype=points.dtype,
        device=points.device,
    )
    transformed = _transform_probe_space_torch(
        points,
        center=center,
        curvature=curvature_tensor,
        eps=eps,
    )
    sqrt_c = torch.sqrt(curvature_tensor)
    unit_points = _project_to_poincare_ball_torch(transformed * sqrt_c, eps=eps)
    distances = pairwise_poincare(unit_points, eps=eps)
    return distances / sqrt_c


def pairwise_distance(
    points: np.ndarray,
    geometry: str,
    *,
    center: Optional[np.ndarray] = None,
    curvature: Optional[float] = None,
    eps: float = POINCARE_EPS,
) -> np.ndarray:
    """Compute pairwise distances using the torch backend and return numpy arrays."""

    tensor = torch.as_tensor(points, dtype=torch.float32)
    geometry = geometry.lower()
    if geometry == "euclidean":
        distances = pairwise_euclidean(tensor)
    elif geometry == "hyperbolic":
        center_tensor = torch.as_tensor(center, dtype=torch.float32) if center is not None else None
        curvature_tensor = torch.as_tensor(curvature, dtype=torch.float32) if curvature is not None else None
        distances = pairwise_poincare(tensor, center=center_tensor, curvature=curvature_tensor, eps=eps)
    else:
        raise ValueError(f"Unsupported geometry '{geometry}'")
    return distances.cpu().numpy()


def _distance_weights(targets: np.ndarray, mask: Optional[np.ndarray], mode: str) -> Optional[np.ndarray]:
    """Return a per-entry weight matrix for distance targets."""

    if mode == "none":
        return None
    if mode != "inverse_freq":
        raise ValueError(f"Unsupported pair_weighting mode '{mode}'")
    if targets.size == 0:
        return None
    # Only count entries that participate in the loss.
    if mask is not None:
        valid = mask.astype(bool)
        if not valid.any():
            return None
        vals = targets[valid]
    else:
        vals = targets
    # Distances are integers in this setting; a small round guards against fp drift.
    vals_rounded = np.round(vals, decimals=6)
    unique, counts = np.unique(vals_rounded, return_counts=True)
    inv_counts = {val: 1.0 / max(cnt, 1) for val, cnt in zip(unique, counts)}
    weights = np.ones_like(targets, dtype=np.float32)
    for val, w in inv_counts.items():
        weights[np.isclose(targets, val)] = w
    return weights


# =============================================================================
# Probe training & evaluation
# =============================================================================

def _resolve_geometry_schedule(cfg: DistanceProbeConfig) -> Dict[str, Any]:
    """Return optimizer hyperparameters tailored to the probe geometry."""

    geometry = cfg.fit_geometry.lower()
    schedule: Dict[str, Any] = {
        "steps": max(cfg.steps, 600),
        "lr": cfg.lr,
        "weight_decay": cfg.weight_decay,
        "optimizer_cls": torch.optim.AdamW,
        "optimizer_kwargs": {},
        "grad_clip": None,
        "scheduler_factory": None,
    }
    if geometry == "hyperbolic":
        schedule.update(
            {
                "steps": max(cfg.steps, 6_000),
                "lr": max(cfg.lr, 1.2e-2),
                "weight_decay": 0.0,
                "optimizer_cls": torch.optim.Adam,
                "optimizer_kwargs": {"betas": (0.9, 0.95)},
                "grad_clip": 5.0,
                "scheduler_factory": lambda opt, total_steps: torch.optim.lr_scheduler.CosineAnnealingLR(
                    opt,
                    T_max=max(total_steps, 1),
                    eta_min=1e-3,
                ),
            }
        )
    return schedule


def fit_distance_probe(
    embeddings: np.ndarray,
    targets: np.ndarray,
    cfg: DistanceProbeConfig,
    *,
    device: str = DEVICE,
    mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Fit the distance probe and return projection matrix plus diagnostics.

    If ``mask`` is provided, it must be a square matrix matching ``targets``.
    Entries with mask==0 are excluded from the loss/metrics.
    """

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch_device = torch.device(device)

    geometry = cfg.fit_geometry.lower()
    if geometry not in {"euclidean", "hyperbolic"}:
        raise ValueError(f"Unsupported fit_geometry '{cfg.fit_geometry}'")

    schedule = _resolve_geometry_schedule(cfg)
    effective_steps = int(schedule["steps"])
    effective_lr = float(schedule["lr"])
    effective_weight_decay = float(schedule["weight_decay"])
    optimizer_kwargs = dict(schedule["optimizer_kwargs"])
    optimizer_cls = schedule["optimizer_cls"]
    grad_clip = schedule["grad_clip"]
    scheduler_factory = schedule.get("scheduler_factory")

    probe = DistanceProbe(embeddings.shape[1], cfg.proj_dim, torch_device)

    X = torch.from_numpy(embeddings).to(torch_device)
    T = torch.from_numpy(targets).to(torch_device)
    W_np = _distance_weights(targets, mask, cfg.pair_weighting)
    base_weights = torch.from_numpy(W_np).to(torch_device) if W_np is not None else torch.ones_like(T)
    M: Optional[torch.Tensor] = None
    if mask is not None:
        mask_tensor = np.asarray(mask)
        if mask_tensor.shape != targets.shape:
            raise ValueError("mask must have the same shape as targets")
        np.fill_diagonal(mask_tensor, 0.0)
        M = torch.from_numpy(mask_tensor.astype(np.float32)).to(torch_device)

    warm_start_history: List[Tuple[int, float, float]] = []
    warm_centroid = torch.zeros(cfg.proj_dim, device=torch_device)
    if geometry == "hyperbolic":
        warm_steps = max(300, min(effective_steps // 3, 1_200))
        if warm_steps > 0:
            warm_lr = min(max(effective_lr * 0.5, 1e-3), 5e-3)
            warm_optimizer = torch.optim.AdamW(probe.parameters(), lr=warm_lr, weight_decay=1e-4)
            loss_fn_warm = torch.nn.MSELoss()
            for step in range(warm_steps):
                warm_optimizer.zero_grad()
                Z_warm = probe(X)
                D_warm = pairwise_euclidean(Z_warm)
                weight_mat = base_weights * M if M is not None else base_weights
                denom = torch.sum(weight_mat)
                warm_loss = (torch.sum((D_warm - T).pow(2) * weight_mat) / max(denom, torch.tensor(1.0, device=denom.device)))
                warm_loss.backward()
                warm_optimizer.step()
                if (step + 1) % 50 == 0 or step == warm_steps - 1:
                    with torch.no_grad():
                        corr = _pair_corr(
                            T.detach().cpu().numpy(),
                            D_warm.detach().cpu().numpy(),
                        )
                    warm_start_history.append((step + 1, float(warm_loss.item()), corr))
            probe.zero_grad(set_to_none=True)
            with torch.no_grad():
                warm_centroid = probe(X).mean(dim=0)

    use_rgeom = geometry == "hyperbolic" and geoopt is not None
    curvature_init = 0.48 if geometry == "hyperbolic" else None

    def _curvature_log_init(curv: float) -> float:
        base = max(curv - 1e-3, 1e-6)
        return float(np.log(np.expm1(base)))

    if use_rgeom:
        probe.B = geoopt.ManifoldParameter(
            probe.B.detach(),
            manifold=geoopt.manifolds.Euclidean(),
        )

    trainable_params: List[torch.nn.Parameter] = [probe.B]

    radial_scale_param: Optional[torch.nn.Parameter] = None
    log_curvature_param: Optional[torch.nn.Parameter] = None
    center_param: Optional[torch.nn.Parameter] = None
    if geometry == "hyperbolic":
        log_init = _curvature_log_init(curvature_init)
        if use_rgeom:
            radial_scale_param = geoopt.ManifoldParameter(
                torch.tensor(0.0, device=torch_device),
                manifold=geoopt.manifolds.Euclidean(),
            )
            log_curvature_param = geoopt.ManifoldParameter(
                torch.tensor(log_init, device=torch_device),
                manifold=geoopt.manifolds.Euclidean(),
            )
        else:
            radial_scale_param = torch.nn.Parameter(torch.tensor(0.0, device=torch_device))
            log_curvature_param = torch.nn.Parameter(torch.tensor(log_init, device=torch_device))

        curvature_tensor_init = torch.tensor(curvature_init, device=torch_device)
        center_init = -warm_centroid
        center_init = _project_to_ball_torch(center_init, curvature_tensor_init, eps=POINCARE_EPS)
        if use_rgeom:
            center_param = geoopt.ManifoldParameter(
                center_init.detach(),
                manifold=geoopt.manifolds.PoincareBall(c=float(curvature_init)),
            )
        else:
            center_param = torch.nn.Parameter(center_init.detach().clone())
        trainable_params.extend([radial_scale_param, log_curvature_param, center_param])

    if use_rgeom:
        optimizer = geoopt.optim.RiemannianAdam(
            trainable_params,
            lr=effective_lr,
            weight_decay=effective_weight_decay,
            betas=(0.9, 0.95),
        )
        scheduler = None
    else:
        optimizer = optimizer_cls(
            trainable_params,
            lr=effective_lr,
            weight_decay=effective_weight_decay,
            **optimizer_kwargs,
        )
        scheduler = scheduler_factory(optimizer, effective_steps) if scheduler_factory else None
    loss_fn = torch.nn.MSELoss()

    history: List[Tuple[int, float, float]] = []

    for step in range(effective_steps):
        optimizer.zero_grad()
        if center_param is not None and center_param.grad is not None:
            center_param.grad.zero_()
        Z = probe(X)
        if radial_scale_param is not None:
            Z = Z * torch.exp(radial_scale_param)
        if geometry == "euclidean":
            D_hat = pairwise_euclidean(Z)
            curvature = None
            center_projected = None
        else:
            curvature = torch.nn.functional.softplus(log_curvature_param) + 1e-3
            center_projected = _project_to_ball_torch(center_param, curvature, eps=POINCARE_EPS)
            D_hat = pairwise_poincare(Z, center=center_projected, curvature=curvature)
        weight_mat = base_weights * M if M is not None else base_weights
        denom = torch.sum(weight_mat)
        loss = (torch.sum((D_hat - T).pow(2) * weight_mat) / max(denom, torch.tensor(1.0, device=denom.device)))
        loss.backward()
        if grad_clip is not None and not use_rgeom:
            clip_grad_norm_(trainable_params, grad_clip)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        if geometry == "hyperbolic" and center_param is not None and not use_rgeom:
            with torch.no_grad():
                curvature_eval = torch.nn.functional.softplus(log_curvature_param) + 1e-3
                curvature_eval = torch.clamp(curvature_eval, min=1e-6)
                current_center = _project_to_ball_torch(center_param.data, curvature_eval, eps=POINCARE_EPS)
                center_param.data.copy_(current_center)
                if center_param.grad is not None:
                    norm_sq = torch.sum(current_center ** 2)
                    factor = ((1 - curvature_eval * norm_sq) ** 2) / 4
                    r_grad = factor * center_param.grad
                    center_param.data.add_(-effective_lr * r_grad)
                    center_param.data.copy_(
                        _project_to_ball_torch(center_param.data, curvature_eval, eps=POINCARE_EPS)
                    )
                    center_param.grad.zero_()

        if (step + 1) % 50 == 0 or step == effective_steps - 1:
            with torch.no_grad():
                corr = _pair_corr(
                    T.detach().cpu().numpy(),
                    D_hat.detach().cpu().numpy(),
                )
            history.append((step + 1, float(loss.item()), corr))

    projection = probe.B.detach().cpu().numpy()
    scale_value = None
    center_value = None
    curvature_value = None
    if radial_scale_param is not None:
        scale_value = float(torch.exp(radial_scale_param.detach()).cpu())
        projection = projection * scale_value
    if geometry == "hyperbolic":
        curvature_tensor = torch.nn.functional.softplus(log_curvature_param.detach()) + 1e-3
        curvature_value = float(curvature_tensor.cpu().item())
        center_tensor = _project_to_ball_torch(
            center_param.detach(),
            torch.as_tensor(curvature_value, dtype=center_param.dtype, device=center_param.device),
            eps=POINCARE_EPS,
        )
        center_value = center_tensor.cpu().numpy()

    return projection, {
        "history": history,
        "geometry": geometry,
        "effective_steps": effective_steps,
        "effective_lr": effective_lr,
        "effective_weight_decay": effective_weight_decay,
        "optimizer": optimizer.__class__.__name__,
        "scheduler": scheduler.__class__.__name__ if scheduler is not None else None,
        "radial_scale": scale_value,
        "warm_start_history": warm_start_history if warm_start_history else None,
        "center": center_value,
        "curvature": curvature_value,
        "pair_weighting": cfg.pair_weighting,
    }


def fit_depth_probe(
    *,
    features: np.ndarray,
    depths: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    exact_mask: Optional[np.ndarray] = None,
    alpha: float = 1e-2,
    normalize_tree: Optional[float] = None,
    seed: int = 0,
) -> Tuple[Dict[str, Any], Pipeline]:
    """Train a weighted linear depth probe on activations and report split metrics."""

    def _mse(a: np.ndarray, b: np.ndarray) -> Optional[float]:
        if a.size == 0 or b.size == 0:
            return None
        return safe_float(float(np.mean((a - b) ** 2)))

    def _pearson(a: np.ndarray, b: np.ndarray) -> Optional[float]:
        if a.size < 2 or b.size < 2:
            return None
        if np.allclose(a, a[0]) or np.allclose(b, b[0]):
            return None
        corr = np.corrcoef(a, b)[0, 1]
        return safe_float(float(corr))

    def _spearman(a: np.ndarray, b: np.ndarray) -> Optional[float]:
        if a.size < 2 or b.size < 2:
            return None
        # Simple average-rank approximation without external deps.
        def ranks(arr: np.ndarray) -> np.ndarray:
            order = np.argsort(arr)
            ranks = np.empty_like(order, dtype=np.float64)
            ranks[order] = np.arange(len(arr), dtype=np.float64)
            return ranks
        return _pearson(ranks(a), ranks(b))

    def _depth_metrics(true_vals: np.ndarray, pred_vals: np.ndarray) -> Dict[str, Optional[float]]:
        if true_vals.size == 0 or pred_vals.size == 0:
            return {"mse": None, "pearson": None, "spearman": None, "r2": None}
        mse_val = _mse(true_vals, pred_vals)
        pearson_val = _pearson(true_vals, pred_vals)
        spearman_val = _spearman(true_vals, pred_vals)
        r2_val = None
        try:
            if true_vals.size >= 2:
                from sklearn.metrics import r2_score
                r2_val = safe_float(float(r2_score(true_vals, pred_vals)))
        except Exception:
            r2_val = None
        return {
            "mse": mse_val,
            "pearson": pearson_val,
            "spearman": spearman_val,
            "r2": r2_val,
        }

    if normalize_tree:
        depths = depths / normalize_tree

    if exact_mask is None:
        exact_mask = np.ones_like(depths, dtype=bool)

    unique, counts = np.unique(depths[train_idx], return_counts=True)
    inv_counts = {val: 1.0 / max(cnt, 1) for val, cnt in zip(unique, counts)}
    sample_weight = np.array([inv_counts[val] for val in depths[train_idx]], dtype=np.float64)

    model = Pipeline(
        steps=[
            ("scale", StandardScaler()),
            ("reg", Ridge(alpha=alpha)),
        ]
    ).fit(features[train_idx], depths[train_idx], reg__sample_weight=sample_weight)

    def _eval(model: Pipeline, indices: np.ndarray) -> Dict[str, Optional[float]]:
        if indices.size == 0:
            return {"mse": None, "pearson": None, "spearman": None, "r2": None}
        preds = model.predict(features[indices])
        return _depth_metrics(depths[indices], preds)

    test_exact_idx = test_idx[exact_mask[test_idx]] if test_idx.size and exact_mask.size else np.zeros((0,), dtype=int)
    test_inexact_idx = test_idx[~exact_mask[test_idx]] if test_idx.size and exact_mask.size else np.zeros((0,), dtype=int)

    rng = np.random.default_rng(seed)
    shuffled_depths = depths[train_idx].copy()
    rng.shuffle(shuffled_depths)
    unique_shuf, counts_shuf = np.unique(shuffled_depths, return_counts=True)
    inv_counts_shuf = {val: 1.0 / max(cnt, 1) for val, cnt in zip(unique_shuf, counts_shuf)}
    sample_weight_shuf = np.array([inv_counts_shuf[val] for val in shuffled_depths], dtype=np.float64)
    shuf_model = Pipeline(
        steps=[
            ("scale", StandardScaler()),
            ("reg", Ridge(alpha=alpha)),
        ]
    ).fit(features[train_idx], shuffled_depths, reg__sample_weight=sample_weight_shuf)

    return (
        {
            "train": _eval(model, train_idx),
            "test": _eval(model, test_idx),
            "test_exact": _eval(model, test_exact_idx),
            "test_inexact": _eval(model, test_inexact_idx),
            "shuf_train": _eval(shuf_model, train_idx),
            "shuf_test": _eval(shuf_model, test_idx),
            "alpha": alpha,
            "normalized_tree": bool(normalize_tree),
            "normalization_factor": float(normalize_tree) if normalize_tree else None,
        },
        model,
    )


def _pair_corr(true_mat: np.ndarray, pred_mat: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
    if true_mat.size == 0 or pred_mat.size == 0:
        return float("nan")
    if mask is not None:
        mask_tri = np.triu(mask, k=1).astype(bool)
        if not mask_tri.any():
            return float("nan")
        true_vals = true_mat[mask_tri]
        pred_vals = pred_mat[mask_tri]
    else:
        tri = np.triu_indices_from(true_mat, k=1)
        if tri[0].size == 0:
            return float("nan")
        true_vals = true_mat[tri]
        pred_vals = pred_mat[tri]
    if true_vals.size < 2:
        return float("nan")
    return float(np.corrcoef(true_vals, pred_vals)[0, 1])


def _pair_mse(true_mat: np.ndarray, pred_mat: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
    """Mean squared error over upper triangle (optionally masked)."""

    if true_mat.size == 0 or pred_mat.size == 0:
        return float("nan")
    if mask is not None:
        mask_tri = np.triu(mask, k=1).astype(bool)
        if not mask_tri.any():
            return float("nan")
        true_vals = true_mat[mask_tri]
        pred_vals = pred_mat[mask_tri]
    else:
        tri = np.triu_indices_from(true_mat, k=1)
        if tri[0].size == 0:
            return float("nan")
        true_vals = true_mat[tri]
        pred_vals = pred_mat[tri]
    if true_vals.size == 0:
        return float("nan")
    return float(np.mean((true_vals - pred_vals) ** 2))


def evaluate_probes(
    encoded: Dict[str, Dict[str, Any]],
    cfg: DistanceProbeConfig,
    *,
    device: str = DEVICE,
    normalize_tree: Optional[float] = None,
    depth_alpha: float = 1e-2,
) -> Dict[str, Dict[str, Any]]:
    """Compute probe performance plus baselines for each dataset."""

    results: Dict[str, Dict[str, Any]] = {}
    for name, data in encoded.items():
        X = data["X"]
        T = data["D"]
        depth = data["depth"]
        example_ids = np.asarray(data.get("example_ids", []), dtype=np.int64)
        train_idx = np.array(data["train_idx"], dtype=int)
        test_idx = np.array(data["test_idx"], dtype=int)

        X_train = X[train_idx]
        X_test = X[test_idx]
        depth_train = depth[train_idx]
        depth_test = depth[test_idx]
        T_train = T[np.ix_(train_idx, train_idx)]
        T_test = T[np.ix_(test_idx, test_idx)]
        ex_train = example_ids[train_idx] if example_ids.size else np.arange(len(train_idx))
        ex_test = example_ids[test_idx] if example_ids.size else np.arange(len(test_idx))
        train_mask = (ex_train[:, None] == ex_train[None, :])
        test_mask = (ex_test[:, None] == ex_test[None, :])
        np.fill_diagonal(train_mask, False)
        np.fill_diagonal(test_mask, False)

        if normalize_tree:
            T_train = T_train / normalize_tree
            T_test = T_test / normalize_tree
            depth_train = depth_train / normalize_tree
            depth_test = depth_test / normalize_tree
        exact_mask_full = np.asarray(data.get("example_is_exact", np.ones_like(depth, dtype=bool)), dtype=bool)
        test_exact_mask = exact_mask_full[test_idx]
        test_inexact_mask = ~test_exact_mask if test_exact_mask.size else np.array([], dtype=bool)
        exact_pair_mask = (
            (np.outer(test_exact_mask, test_exact_mask) & test_mask) if test_exact_mask.size else np.zeros_like(test_mask, dtype=bool)
        )
        inexact_pair_mask = (
            (np.outer(test_inexact_mask, test_inexact_mask) & test_mask) if test_inexact_mask.size else np.zeros_like(test_mask, dtype=bool)
        )

        B, info = fit_distance_probe(X_train, T_train, cfg, device=device, mask=train_mask)
        Z_train_raw = X_train @ B
        Z_test_raw = X_test @ B

        geometry = info.get("geometry", cfg.fit_geometry).lower()
        center = info.get("center")
        curvature = info.get("curvature")

        Dhat_train = (
            pairwise_distance(Z_train_raw, geometry, center=center, curvature=curvature)
            if len(train_idx) > 1
            else np.zeros_like(T_train)
        )
        Dhat_test = (
            pairwise_distance(Z_test_raw, geometry, center=center, curvature=curvature)
            if len(test_idx) > 1
            else np.zeros_like(T_test)
        )

        dist_corr_train = _pair_corr(T_train, Dhat_train, mask=train_mask)
        dist_corr_test = _pair_corr(T_test, Dhat_test, mask=test_mask)
        dist_mse_train = _pair_mse(T_train, Dhat_train, mask=train_mask)
        dist_mse_test = _pair_mse(T_test, Dhat_test, mask=test_mask)
        dist_mse_test_exact = _pair_mse(T_test, Dhat_test, mask=exact_pair_mask)
        dist_mse_test_inexact = _pair_mse(T_test, Dhat_test, mask=inexact_pair_mask)
        dist_corr_test_exact = _pair_corr(T_test, Dhat_test, mask=exact_pair_mask)
        dist_corr_test_inexact = _pair_corr(T_test, Dhat_test, mask=inexact_pair_mask)

        if geometry == "hyperbolic":
            Z_train = transform_probe_space(Z_train_raw, info)
            Z_test = transform_probe_space(Z_test_raw, info)
        else:
            Z_train = Z_train_raw
            Z_test = Z_test_raw

        if len(train_idx) > 0 and min(X_train.shape[0], X_train.shape[1]) > 0:
            n_components = min(2, X_train.shape[0], X_train.shape[1])
            pca = PCA(n_components=n_components)
            Zp_train_fit = pca.fit_transform(X_train)
            Zp_train = np.zeros((len(train_idx), 2), dtype=np.float32)
            Zp_train[:, :n_components] = Zp_train_fit.astype(np.float32)
            if len(test_idx) > 0:
                Zp_test_fit = pca.transform(X_test)
                Zp_test = np.zeros((len(test_idx), 2), dtype=np.float32)
                Zp_test[:, :n_components] = Zp_test_fit.astype(np.float32)
            else:
                Zp_test = np.zeros((0, 2), dtype=np.float32)
        else:
            Zp_train = np.zeros((len(train_idx), 2), dtype=np.float32)
            Zp_test = np.zeros((len(test_idx), 2), dtype=np.float32)

        Dpca_train = (
            pairwise_distance(Zp_train, geometry, center=center, curvature=curvature)
            if len(train_idx) > 1
            else np.zeros_like(T_train)
        )
        Dpca_test = (
            pairwise_distance(Zp_test, geometry, center=center, curvature=curvature)
            if len(test_idx) > 1
            else np.zeros_like(T_test)
        )

        dist_corr_pca_train = _pair_corr(T_train, Dpca_train, mask=train_mask)
        dist_corr_pca_test = _pair_corr(T_test, Dpca_test, mask=test_mask)
        dist_mse_pca_train = _pair_mse(T_train, Dpca_train, mask=train_mask)
        dist_mse_pca_test = _pair_mse(T_test, Dpca_test, mask=test_mask)

        # Depth probe: optionally use precomputed PCA features when provided.
        depth_features = np.asarray(data.get("depth_X", X))
        X_train_raw = depth_features[train_idx]
        X_test_raw = depth_features[test_idx]
        all_features = np.vstack([X_train_raw, X_test_raw])
        all_depths = np.concatenate([depth_train, depth_test])
        train_idx_local = np.arange(len(X_train_raw))
        test_idx_local = np.arange(len(X_train_raw), len(X_train_raw) + len(X_test_raw))
        exact_mask_local = np.concatenate(
            [exact_mask_full[train_idx], exact_mask_full[test_idx]]
        ).astype(bool)
        depth_metrics, depth_model = fit_depth_probe(
            features=all_features,
            depths=all_depths,
            train_idx=train_idx_local,
            test_idx=test_idx_local,
            exact_mask=exact_mask_local,
            alpha=depth_alpha,
            normalize_tree=normalize_tree,
            seed=cfg.seed,
        )

        rng = np.random.default_rng(cfg.seed)
        B_rand = rng.normal(0, 1 / np.sqrt(max(X.shape[1], 1)), size=(X.shape[1], cfg.proj_dim)).astype(np.float32)
        Zr_train_raw = X_train @ B_rand
        Zr_test_raw = X_test @ B_rand
        Dhr_train = (
            pairwise_distance(Zr_train_raw, geometry, center=center, curvature=curvature)
            if len(train_idx) > 1
            else np.zeros_like(T_train)
        )
        Dhr_test = (
            pairwise_distance(Zr_test_raw, geometry, center=center, curvature=curvature)
            if len(test_idx) > 1
            else np.zeros_like(T_test)
        )

        if geometry == "hyperbolic":
            Zr_train = transform_probe_space(Zr_train_raw, info)
            Zr_test = transform_probe_space(Zr_test_raw, info)
        else:
            Zr_train = Zr_train_raw
            Zr_test = Zr_test_raw
        dist_corr_rand_train = _pair_corr(T_train, Dhr_train, mask=train_mask)
        dist_corr_rand_test = _pair_corr(T_test, Dhr_test, mask=test_mask)
        dist_mse_rand_train = _pair_mse(T_train, Dhr_train, mask=train_mask)
        dist_mse_rand_test = _pair_mse(T_test, Dhr_test, mask=test_mask)

        T_shuf = T_train.copy().reshape(-1)
        rng.shuffle(T_shuf)
        T_shuf = T_shuf.reshape(T_train.shape)
        B_shuf, _ = fit_distance_probe(
            X_train,
            T_shuf,
            DistanceProbeConfig(
                proj_dim=cfg.proj_dim,
                steps=min(cfg.steps, 600),
                lr=cfg.lr,
                weight_decay=cfg.weight_decay,
                seed=cfg.seed,
                fit_geometry=cfg.fit_geometry,
            ),
            device=device,
        )
        Zsh_train = X_train @ B_shuf
        Zsh_test = X_test @ B_shuf
        Dhsh_train = (
            pairwise_distance(Zsh_train, geometry, center=center, curvature=curvature)
            if len(train_idx) > 1
            else np.zeros_like(T_train)
        )
        Dhsh_test = (
            pairwise_distance(Zsh_test, geometry, center=center, curvature=curvature)
            if len(test_idx) > 1
            else np.zeros_like(T_test)
        )

        if normalize_tree:
            Dhsh_train = Dhsh_train / normalize_tree
            Dhsh_test = Dhsh_test / normalize_tree
        dist_corr_shuf_train = _pair_corr(T_train, Dhsh_train, mask=train_mask)
        dist_corr_shuf_test = _pair_corr(T_test, Dhsh_test, mask=test_mask)
        dist_mse_shuf_train = _pair_mse(T_train, Dhsh_train, mask=train_mask)
        dist_mse_shuf_test = _pair_mse(T_test, Dhsh_test, mask=test_mask)

        results[name] = {
            "dist_corr_train": safe_float(dist_corr_train),
            "dist_corr_test": safe_float(dist_corr_test),
            "dist_mse_train": safe_float(dist_mse_train),
            "dist_mse_test": safe_float(dist_mse_test),
            "projection": B,
            "depth": depth_metrics,
            "depth_model": depth_model,
            "dist_corr_pca_train": safe_float(dist_corr_pca_train),
            "dist_corr_pca_test": safe_float(dist_corr_pca_test),
            "dist_mse_pca_train": safe_float(dist_mse_pca_train),
            "dist_mse_pca_test": safe_float(dist_mse_pca_test),
            "dist_corr_rand_train": safe_float(dist_corr_rand_train),
            "dist_corr_rand_test": safe_float(dist_corr_rand_test),
            "dist_mse_rand_train": safe_float(dist_mse_rand_train),
            "dist_mse_rand_test": safe_float(dist_mse_rand_test),
            "dist_corr_shuf_test": safe_float(dist_corr_shuf_test),
            "dist_corr_shuf_train": safe_float(dist_corr_shuf_train),
            "dist_mse_shuf_test": safe_float(dist_mse_shuf_test),
            "dist_mse_shuf_train": safe_float(dist_mse_shuf_train),
            "dist_corr_test_exact": safe_float(dist_corr_test_exact),
            "dist_corr_test_inexact": safe_float(dist_corr_test_inexact),
            "dist_mse_test_exact": safe_float(dist_mse_test_exact),
            "dist_mse_test_inexact": safe_float(dist_mse_test_inexact),
            "history": info["history"],
            "train_idx": train_idx.tolist(),
            "test_idx": test_idx.tolist(),
            "geometry": geometry,
            "effective_steps": info.get("effective_steps"),
            "effective_lr": info.get("effective_lr"),
            "effective_weight_decay": info.get("effective_weight_decay"),
            "optimizer": info.get("optimizer"),
            "scheduler": info.get("scheduler"),
            "radial_scale": info.get("radial_scale"),
            "warm_start_history": info.get("warm_start_history"),
            "center": info.get("center"),
            "curvature": info.get("curvature"),
            "normalized_tree": bool(normalize_tree),
            "pair_weighting": cfg.pair_weighting,
        }
    return results


def safe_float(value: Any) -> Optional[float]:
    """Convert numeric values to plain floats, guarding against NaNs."""

    if isinstance(value, (float, np.floating)):
        if math.isnan(float(value)):
            return None
        return float(value)
    return value
