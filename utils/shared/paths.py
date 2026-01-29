"""Path helpers for the traversal pipeline.

Unified layout (per dataset tag):
    cutter/data/{dataset_tag}/
        dataset/traversal_paths.jsonl (tree) or gsm8k.jsonl (math)
        models/{model_tag}/
            responses.jsonl
            embeddings.npz
            embeddings_pca{N}.npz
            probes/probe[_normtree]_proj{proj_dim}_pca{pca}.npz
            interventions/interventions_proj{proj_dim}_pca{pca}_layer{layer}.npz

Tags are sanitized versions of model ids and dataset descriptors to keep
filesystem paths stable across scripts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple, Union
import re

from cutter.utils.shared.basic import sanitize_tag

REPO_ROOT = Path(__file__).resolve().parents[3]
BASE_DATA_DIR = REPO_ROOT / "cutter" / "data"
DEFAULT_TREE_DATASET_TAG = "depth1-2_n1000_steps1-2"


# ---------------------------------------------------------------------------
# Tag helpers


def _format_depth_tag(min_depth: int, max_depth: int) -> str:
    if min_depth == max_depth:
        return f"depth{min_depth}"
    return f"depth{min_depth}-{max_depth}"


def _format_steps_tag(num_steps: Union[int, Tuple[int, int]]) -> str:
    if isinstance(num_steps, tuple):
        if len(num_steps) != 2:
            raise ValueError("num_steps tuple must have length 2")
        min_steps, max_steps = num_steps
    else:
        min_steps = max_steps = int(num_steps)

    if min_steps == max_steps == 1:
        return ""
    if min_steps == max_steps:
        return f"_steps{min_steps}"
    return f"_steps{min_steps}-{max_steps}"


def dataset_tag(
    min_depth: int,
    max_depth: int,
    num_samples: int,
    num_steps: Union[int, Tuple[int, int]] = 1,
) -> str:
    """Compact, filesystem-safe identifier for dataset settings."""

    suffix = f"n{num_samples}" if num_samples > 0 else "nall"
    depth = _format_depth_tag(min_depth, max_depth)
    steps = _format_steps_tag(num_steps)
    return f"{depth}_{suffix}{steps}"


def dataset_tag_from_path(dataset_path: Path) -> str:
    """Infer dataset tag from the path.

    Expected: .../{dataset_tag}/dataset/traversal_paths.jsonl
    Fallback: parent name or stem.
    """

    resolved = dataset_path.resolve()
    if resolved.parent.name == "dataset":
        return resolved.parent.parent.name
    return resolved.parent.name or resolved.stem


def parse_tree_dataset_tag(tag: str) -> Tuple[int, int, int, Tuple[int, int]]:
    """Parse a traversal dataset tag into (min_depth, max_depth, num_samples, (min_steps, max_steps))."""

    pattern = re.compile(
        r"^depth(?P<dmin>\d+)(?:-(?P<dmax>\d+))?_n(?P<nsamp>all|\d+)"
        r"(?:_steps(?P<smin>\d+)(?:-(?P<smax>\d+))?)?$"
    )
    match = pattern.match(tag)
    if not match:
        raise ValueError(f"Unrecognized traversal dataset tag: {tag}")
    dmin = int(match.group("dmin"))
    dmax = int(match.group("dmax") or dmin)
    nsamp_raw = match.group("nsamp")
    num_samples = 0 if nsamp_raw == "all" else int(nsamp_raw)
    smin = int(match.group("smin") or 1)
    smax = int(match.group("smax") or smin)
    return dmin, dmax, num_samples, (smin, smax)


def model_tag(model_id: str) -> str:
    """Filesystem-safe model tag from a HF id."""

    return sanitize_tag(model_id.split("/")[-1])


# ---------------------------------------------------------------------------
# Dataset paths


def dataset_dir(tag: str) -> Path:
    """Directory that contains the dataset and any sidecars."""

    return BASE_DATA_DIR / tag / "dataset"


def dataset_path_from_tag(tag: str) -> Path:
    """JSONL path for a dataset tag."""

    return dataset_dir(tag) / "traversal_paths.jsonl"


def gsm8k_dataset_tag(num_samples: int, seed: int, split: str) -> str:
    """Compact, filesystem-safe identifier for GSM8K dataset settings."""

    split_tag = sanitize_tag((split or "all").lower())
    sample_tag = "nall" if num_samples <= 0 else f"n{num_samples}"
    if split_tag == "all":
        return f"gsm8k_{sample_tag}_seed{seed}"
    return f"gsm8k_{split_tag}_{sample_tag}_seed{seed}"


def parse_gsm8k_dataset_tag(tag: str) -> Tuple[str, int, int]:
    """Parse a GSM8K dataset tag into (split, num_samples, seed)."""

    pattern = re.compile(
        r"^gsm8k_(?:(?P<split>[a-z0-9]+)_)?(?P<sample>nall|n\d+)_seed(?P<seed>\d+)$"
    )
    match = pattern.match(tag)
    if not match:
        raise ValueError(f"Unrecognized GSM8K dataset tag: {tag}")
    split = match.group("split") or "all"
    sample_raw = match.group("sample")
    num_samples = 0 if sample_raw == "nall" else int(sample_raw[1:])
    seed = int(match.group("seed"))
    return split, num_samples, seed


def gsm8k_dataset_path_from_tag(tag: str) -> Path:
    """JSONL path for a GSM8K dataset tag."""

    return BASE_DATA_DIR / tag / "dataset" / "gsm8k.jsonl"


def default_gsm8k_dataset_path(num_samples: int, seed: int, split: str) -> Path:
    """Convenience wrapper for the default GSM8K dataset path."""

    return gsm8k_dataset_path_from_tag(gsm8k_dataset_tag(num_samples, seed, split))


def default_dataset_path(
    min_depth: int,
    max_depth: int,
    num_samples: int,
    num_steps: Union[int, Tuple[int, int]] = 1,
) -> Path:
    """Convenience wrapper for the default dataset path."""

    return dataset_path_from_tag(dataset_tag(min_depth, max_depth, num_samples, num_steps))


def resolve_dataset_tag(
    setting: str,
    dataset_tag_value: str | None,
    *,
    num_samples: int,
    seed: int,
    gsm8k_split: str,
) -> str:
    if dataset_tag_value:
        return dataset_tag_value
    if setting == "math":
        return gsm8k_dataset_tag(num_samples, seed, gsm8k_split)
    return DEFAULT_TREE_DATASET_TAG


def resolve_dataset_path(
    setting: str,
    dataset_tag_value: str | None,
    *,
    num_samples: int,
    seed: int,
    gsm8k_split: str,
) -> Path:
    dataset_tag_value = resolve_dataset_tag(
        setting,
        dataset_tag_value,
        num_samples=num_samples,
        seed=seed,
        gsm8k_split=gsm8k_split,
    )
    if setting == "math":
        return gsm8k_dataset_path_from_tag(dataset_tag_value)
    return dataset_path_from_tag(dataset_tag_value)


# ---------------------------------------------------------------------------
# Model output paths


def model_output_dir(dataset_tag: str, model_id: str) -> Path:
    """Directory for responses/embeddings for a given dataset/model pair."""

    return BASE_DATA_DIR / dataset_tag / "models" / model_tag(model_id)


def responses_path(dataset_tag: str, model_id: str) -> Path:
    """Default responses JSONL path for a dataset/model pair."""

    return model_output_dir(dataset_tag, model_id) / "responses.jsonl"


def embeddings_path(dataset_tag: str, model_id: str) -> Path:
    """Default embeddings NPZ path for a dataset/model pair."""

    return model_output_dir(dataset_tag, model_id) / "embeddings.npz"


def embeddings_pca_path(dataset_tag: str, model_id: str, pca_components: int) -> Path:
    """Default PCA-reduced embeddings NPZ path for a dataset/model pair."""

    if pca_components < 0:
        return embeddings_path(dataset_tag, model_id)
    return model_output_dir(dataset_tag, model_id) / f"embeddings_pca{pca_components}.npz"


def repo_relative(path: Path) -> str:
    """Return a repo-relative string if possible."""

    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def resolve_repo_path(path: Path) -> Path:
    """Resolve relative or relocated repo paths back into this checkout."""

    if path.is_absolute():
        if path.exists():
            return path
        parts = path.parts
        if "cutter" in parts:
            idx = parts.index("cutter")
            return REPO_ROOT.joinpath(*parts[idx:])
        return path
    return REPO_ROOT / path


# ---------------------------------------------------------------------------
# Probe paths


def probe_output_dir(dataset_tag: str, model_id: str) -> Path:
    """Directory for probe results for a dataset/model pair."""

    return model_output_dir(dataset_tag, model_id) / "probes"


def _pca_tag(pca_components: int) -> str:
    if pca_components < 0:
        return "all"
    return str(pca_components)


def probe_path(dataset_tag: str, model_id: str, proj_dim: int, normalize_tree: bool, pca_components: int) -> Path:
    """Default probe results path."""

    pca_tag = _pca_tag(pca_components)
    name = f"probe_proj{proj_dim}_pca{pca_tag}.npz"
    if normalize_tree:
        name = f"probe_normtree_proj{proj_dim}_pca{pca_tag}.npz"
    return probe_output_dir(dataset_tag, model_id) / name


# ---------------------------------------------------------------------------
# Intervention paths


def intervention_output_dir(dataset_tag: str, model_id: str) -> Path:
    """Directory for intervention outputs for a dataset/model pair."""

    return model_output_dir(dataset_tag, model_id) / "interventions"


def intervention_path(
    dataset_tag: str,
    model_id: str,
    proj_dim: int,
    pca_components: int,
    layer: int | str,
    *,
    tag: str | None = None,
) -> Path:
    """Default intervention results path."""

    pca_tag = _pca_tag(pca_components)
    name = f"interventions_proj{proj_dim}_pca{pca_tag}_layer{layer}"
    if tag:
        name = f"{name}_{tag}"
    return intervention_output_dir(dataset_tag, model_id) / f"{name}.npz"
