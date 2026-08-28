"""Project path utilities.

The module keeps legacy attribute names for notebook compatibility while adding
helpers for organized experiment-specific assets.
"""

from __future__ import annotations

import os
from pathlib import Path

from src.utils.experiments import (
    resolve_experiment_notebook,
    resolve_experiment_results_dir,
    resolve_model_path,
)


TISSUE_DIMENSIONS = {
    "cervix": {"num_rows": 600, "num_cols": 800},
    "brain": {"num_rows": 388, "num_cols": 516},
    "afmmm": {"num_rows": 500, "num_cols": 500},
    "simulated": {"num_rows": 100, "num_cols": 100},
}


class FilePaths:
    """Centralized paths used across notebooks and scripts."""

    def __init__(self, base_path: Path | None = None, processed_path: Path | None = None):
        self.base_path = (base_path or Path(__file__).resolve().parents[2]).resolve()

        self.data_path = self.base_path / "data"
        self.test_path = self.data_path / "test"
        self.raw_path = self.data_path / "raw"
        self.interim_path = self.data_path / "interim"
        self.combined_interim_path = self.interim_path / "combined"

        self.cervix_test_path = self.test_path / "cervix"
        self.brain_test_path = self.test_path / "brain"
        self.afmmm_test_path = self.test_path / "afmmm"
        self.simulated_test_path = self._resolve_simulated_path()

        self.cervix_raw_path = self.raw_path / "cervix"
        self.brain_raw_path = self.raw_path / "brain"
        self.afmmm_raw_path = self.raw_path / "afmmm"

        self.cervix_interim_path = self.interim_path / "cervix"
        self.brain_interim_path = self.interim_path / "brain"
        self.afmmm_interim_path = self.interim_path / "afmmm"

        self.processed_path = self._resolve_processed_path(processed_path)
        self.cervix_processed_path = self.processed_path / "cervix"
        self.brain_processed_path = self.processed_path / "brain"
        self.afmmm_processed_path = self.processed_path / "afmmm"

        self.results = self.base_path / "results"
        self.figures = self.results / "figures"

        self.model_save_path = self.base_path / "model"
        self.notebooks_path = self.base_path / "notebooks"

    def _resolve_simulated_path(self) -> Path:
        preferred = Path("/Volumes/ep_ssd/database/partialPr/data/test/simulated")
        if preferred.exists():
            return preferred
        return self.test_path / "simulated"

    def _resolve_processed_path(self, override: Path | None) -> Path:
        if override is not None:
            return Path(override).resolve()

        env_override = os.getenv("PARTIAL_PR_PROCESSED_PATH")
        if env_override:
            return Path(env_override).expanduser().resolve()

        preferred_external = Path("/Volumes/ep_ssd/database/partialPr/data/processed")
        if preferred_external.exists():
            return preferred_external

        return self.data_path / "processed"

    def experiment_model_path(self, experiment_name: str, model_family: str) -> Path:
        """Return model path for a specific experiment/model family pair."""
        return resolve_model_path(self.base_path, experiment_name, model_family)

    def experiment_results_path(self, experiment_name: str) -> Path:
        """Return results directory for an experiment."""
        return resolve_experiment_results_dir(self.base_path, experiment_name)

    def experiment_notebook_path(self, experiment_name: str) -> Path | None:
        """Return notebook path for an experiment if present."""
        return resolve_experiment_notebook(self.base_path, experiment_name)


# Backward-compatible names used by existing notebooks.
FilePath = FilePaths
file_paths = FilePaths()
