"""Helpers for loading experiment-specific trained models."""

from __future__ import annotations

from pathlib import Path

from src.utils.experiments import (
    MODEL_FILE_BASENAMES,
    get_experiment,
    resolve_model_path,
)
from src.utils.file_paths import file_paths
from src.utils.testing_utils.trained_models import load_pixel_mlp_checkpoint


def get_experiment_model_paths(
    experiment_name: str,
    project_root: Path | None = None,
) -> dict[str, Path]:
    """Return resolved model paths for an experiment variant."""
    root = (project_root or file_paths.base_path).resolve()
    get_experiment(experiment_name)  # validates name
    return {
        model_family: resolve_model_path(root, experiment_name, model_family)
        for model_family in MODEL_FILE_BASENAMES
    }


def load_xgb_model(model_path: Path | str):
    """Load an XGBoost booster from disk."""
    import xgboost as xgb

    model = xgb.Booster()
    model.load_model(str(model_path))
    return model


def load_catboost_model(model_path: Path | str):
    """Load a CatBoost model from disk."""
    from catboost import CatBoostClassifier

    model = CatBoostClassifier()
    model.load_model(str(model_path))
    return model


def load_models_for_experiment(
    experiment_name: str,
    project_root: Path | None = None,
    include_mlp: bool = True,
    map_location: str = "cpu",
) -> dict[str, object]:
    """Load available models (XGBoost/CatBoost/MLP) for an experiment."""
    paths = get_experiment_model_paths(experiment_name, project_root=project_root)
    loaded: dict[str, object] = {}

    if paths["xgb"].exists():
        loaded["xgb"] = load_xgb_model(paths["xgb"])

    if paths["catboost"].exists():
        loaded["catboost"] = load_catboost_model(paths["catboost"])

    if include_mlp and paths["pixel_mlp"].exists():
        loaded["pixel_mlp"] = load_pixel_mlp_checkpoint(paths["pixel_mlp"], map_location=map_location)

    return loaded
