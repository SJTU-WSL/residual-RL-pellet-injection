"""Shared data models and configuration helpers for the visualization app."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from importlib import util as importlib_util
import json
from pathlib import Path
import pickle
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VISUALIZATION_ROOT = Path(__file__).resolve().parent
RUNS_ROOT = VISUALIZATION_ROOT / "runs"

SCALAR_FIELDS = [
    "Q_fusion",
    "H98",
    "q95",
    "q_min",
    "W_thermal_total",
    "fgw_n_e_volume_avg",
    "P_fusion",
    "tau_E",
    "P_external_total",
    "n_e_volume_avg",
    "T_e_volume_avg",
    "T_i_volume_avg",
    "n_e_core",
    "T_e_core",
    "T_i_core",
    "S_pellet",
    "injection_applied",
    "injection_commanded",
    "pellet_location",
    "pellet_width",
    "pellet_rate",
]

VECTOR_FIELDS = [
    "T_e",
    "T_i",
    "n_e",
    "n_i",
    "q_face",
    "s_face",
]

DEFAULT_SELECTED_FIELDS = [
    "Q_fusion",
    "H98",
    "q_min",
    "fgw_n_e_volume_avg",
    "n_e_volume_avg",
    "T_e_core",
    "T_i_core",
    "n_e",
    "T_e",
]

FIELD_LABELS = {
    "Q_fusion": "Q Fusion",
    "H98": "H98",
    "q95": "q95",
    "q_min": "q min",
    "W_thermal_total": "W Thermal",
    "fgw_n_e_volume_avg": "fGW",
    "P_fusion": "P Fusion",
    "tau_E": "Tau E",
    "P_external_total": "P External",
    "n_e_volume_avg": "n_e avg",
    "T_e_volume_avg": "T_e avg",
    "T_i_volume_avg": "T_i avg",
    "n_e_core": "n_e core",
    "T_e_core": "T_e core",
    "T_i_core": "T_i core",
    "S_pellet": "Pellet Source",
    "injection_applied": "Applied Injection",
    "injection_commanded": "Commanded Injection",
    "pellet_location": "Pellet Location",
    "pellet_width": "Pellet Width",
    "pellet_rate": "Pellet Rate",
    "T_e": "Electron Temperature",
    "T_i": "Ion Temperature",
    "n_e": "Electron Density",
    "n_i": "Ion Density",
    "q_face": "q Face",
    "s_face": "s Face",
}

FIELD_COLORS = {
    "Q_fusion": "#66d9ef",
    "H98": "#a6e22e",
    "q95": "#f8f8f2",
    "q_min": "#ff79c6",
    "W_thermal_total": "#fd971f",
    "fgw_n_e_volume_avg": "#50fa7b",
    "P_fusion": "#8be9fd",
    "tau_E": "#bd93f9",
    "P_external_total": "#ffb86c",
    "n_e_volume_avg": "#00d2d3",
    "T_e_volume_avg": "#ff9f43",
    "T_i_volume_avg": "#ee5253",
    "n_e_core": "#48dbfb",
    "T_e_core": "#ff6b6b",
    "T_i_core": "#f368e0",
    "S_pellet": "#feca57",
    "injection_applied": "#c8d6e5",
    "injection_commanded": "#8395a7",
    "pellet_location": "#5f27cd",
    "pellet_width": "#10ac84",
    "pellet_rate": "#feca57",
    "T_e": "#ff9f43",
    "T_i": "#ff6b6b",
    "n_e": "#48dbfb",
    "n_i": "#00d2d3",
    "q_face": "#a29bfe",
    "s_face": "#ffeaa7",
}

FIELD_UNITS = {
    "Q_fusion": "",
    "H98": "",
    "q95": "",
    "q_min": "",
    "W_thermal_total": "J",
    "fgw_n_e_volume_avg": "",
    "P_fusion": "W",
    "tau_E": "s",
    "P_external_total": "W",
    "n_e_volume_avg": "m^-3",
    "T_e_volume_avg": "keV",
    "T_i_volume_avg": "keV",
    "n_e_core": "m^-3",
    "T_e_core": "keV",
    "T_i_core": "keV",
    "S_pellet": "s^-1",
    "injection_applied": "",
    "injection_commanded": "",
    "pellet_location": "rho",
    "pellet_width": "rho",
    "pellet_rate": "s^-1",
    "T_e": "eV",
    "T_i": "eV",
    "n_e": "m^-3",
    "n_i": "m^-3",
    "q_face": "",
    "s_face": "",
}


def default_iter_path() -> str:
    return str(PROJECT_ROOT / "config" / "ITER.py")


def normalize_custom_fields(custom_text: str) -> list[str]:
    tokens = custom_text.replace("\n", ",").split(",")
    return [token.strip() for token in tokens if token.strip()]


def merge_selected_fields(selected_fields: list[str], custom_text: str) -> list[str]:
    merged: list[str] = []
    seen: set[str] = set()
    for field in list(selected_fields) + normalize_custom_fields(custom_text):
        if field not in seen:
            merged.append(field)
            seen.add(field)
    return merged


def split_supported_fields(fields: list[str]) -> tuple[list[str], list[str], list[str]]:
    scalars: list[str] = []
    vectors: list[str] = []
    unsupported: list[str] = []
    for field in fields:
        if field in SCALAR_FIELDS:
            scalars.append(field)
        elif field in VECTOR_FIELDS:
            vectors.append(field)
        else:
            unsupported.append(field)
    return scalars, vectors, unsupported


def load_config_dict(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path).resolve()
    spec = importlib_util.spec_from_file_location("visualization_iter_config", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load config file: {path}")
    module = importlib_util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = getattr(module, "CONFIG", None)
    if not isinstance(config, dict):
        raise ValueError(f"{path} does not define a CONFIG dictionary")
    return config


def load_visual_geometry(config_path: str | Path) -> dict[str, float]:
    config = load_config_dict(config_path)
    geometry = config.get("geometry", {})
    return {
        "R_major": float(geometry.get("R_major", 6.2)),
        "a_minor": float(geometry.get("a_minor", 2.0)),
        # ITER.py does not expose these directly; keep stable visualization defaults.
        "kappa": float(geometry.get("kappa", 1.8)),
        "delta": float(geometry.get("delta", 0.4)),
        "B_0": float(geometry.get("B_0", 5.3)),
    }


@dataclass
class SimulationSettings:
    config_path: str = field(default_factory=default_iter_path)
    batch_size: int = 4
    total_steps: int = 10_000
    injection_interval_ms: int = 100
    velocity_mps: float = 300.0
    thickness_mm: float = 4.0
    selected_fields: list[str] = field(default_factory=lambda: DEFAULT_SELECTED_FIELDS.copy())
    preview_every_steps: int = 10

    @property
    def thickness_m(self) -> float:
        return self.thickness_mm / 1000.0

    @property
    def dt_ms(self) -> int:
        return 1

    def to_metadata(self) -> dict[str, Any]:
        metadata = asdict(self)
        metadata["thickness_m"] = self.thickness_m
        metadata["dt_ms"] = self.dt_ms
        return metadata


def build_run_directory(run_id: str) -> Path:
    return RUNS_ROOT / run_id


def save_run_bundle(run_result: dict[str, Any], run_directory: Path) -> dict[str, str]:
    run_directory.mkdir(parents=True, exist_ok=True)
    metadata_path = run_directory / "run_metadata.json"
    results_path = run_directory / "run_results.pkl"

    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(run_result.get("metadata", {}), f, indent=2)

    with results_path.open("wb") as f:
        pickle.dump(run_result, f)

    return {
        "metadata": str(metadata_path),
        "results": str(results_path),
    }

