"""Experiment registry and path helpers.

This keeps partial experiment variants (for example ``3x3`` and ``4x3``)
explicit in code instead of scattering suffix logic across notebooks/scripts.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable


MODEL_FILE_BASENAMES: Dict[str, str] = {
    "xgb": "pixel_xgb.json",
    "catboost": "pixel_catboost.cbm",
    "pixel_mlp": "best_pixel_mlp.pth",
}


@dataclass(frozen=True)
class ExperimentSpec:
    """Metadata for an experiment variant."""

    name: str
    suffix: str
    notebook_file: str | None
    results_subdir: str

    def model_filename(self, model_family: str) -> str:
        """Return the canonical model filename for this experiment."""
        if model_family not in MODEL_FILE_BASENAMES:
            supported = ", ".join(sorted(MODEL_FILE_BASENAMES))
            raise ValueError(f"Unsupported model family '{model_family}'. Expected one of: {supported}")
        return MODEL_FILE_BASENAMES[model_family]


EXPERIMENTS: Dict[str, ExperimentSpec] = {
    "full": ExperimentSpec(
        name="full",
        suffix="",
        notebook_file=None,
        results_subdir="full",
    ),
    "3x3": ExperimentSpec(
        name="3x3",
        suffix="_3x3",
        notebook_file="pr_partial_3x3.ipynb",
        results_subdir="3x3",
    ),
    "4x3": ExperimentSpec(
        name="4x3",
        suffix="_4x3",
        notebook_file="pr_partial_4x3.ipynb",
        results_subdir="4x3",
    ),
    "4x1_lastcol": ExperimentSpec(
        name="4x1_lastcol",
        suffix="_4x1_lastcol",
        notebook_file="pr_partial_4x1_lastcol.ipynb",
        results_subdir="4x1_lastcol",
    ),
}


def list_experiment_names() -> Iterable[str]:
    """Return experiment names in stable order."""
    return EXPERIMENTS.keys()


def get_experiment(name: str) -> ExperimentSpec:
    """Return the registered experiment specification."""
    try:
        return EXPERIMENTS[name]
    except KeyError as exc:
        supported = ", ".join(EXPERIMENTS.keys())
        raise ValueError(f"Unknown experiment '{name}'. Expected one of: {supported}") from exc


def resolve_model_path(project_root: Path, experiment_name: str, model_family: str) -> Path:
    """Resolve a model path with support for organized and legacy layouts.

    Resolution order:
    1. ``model/experiments/<experiment>/<canonical_filename>``
    2. Legacy flat name in ``model/`` (for backward compatibility)
    """
    spec = get_experiment(experiment_name)
    canonical_name = spec.model_filename(model_family)

    organized_path = project_root / "model" / "experiments" / spec.name / canonical_name
    if organized_path.exists():
        return organized_path

    legacy_name = canonical_name if spec.suffix == "" else _add_suffix(canonical_name, spec.suffix)
    return project_root / "model" / legacy_name


def resolve_experiment_notebook(project_root: Path, experiment_name: str) -> Path | None:
    """Return the notebook path for an experiment, if one exists."""
    spec = get_experiment(experiment_name)
    if spec.notebook_file is None:
        return None
    return project_root / "notebooks" / "experiments" / spec.notebook_file


def resolve_experiment_results_dir(project_root: Path, experiment_name: str) -> Path:
    """Return the results directory for an experiment."""
    spec = get_experiment(experiment_name)
    if spec.name == "full":
        return project_root / "results"
    return project_root / "results" / "partial_experiments" / spec.results_subdir


def _add_suffix(filename: str, suffix: str) -> str:
    stem, ext = filename.rsplit(".", 1)
    return f"{stem}{suffix}.{ext}"
